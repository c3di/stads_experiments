import os
import time
import threading
from typing import Optional, List

from tifffile import tifffile

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import logging
import traceback

from stads.debug_images import save_error_map, save_pixel_wise_psnr_plots
from stads.evaluation import calculate_psnr, calculate_ssim
from stads.read_images import get_frames_from_tif
from sem_noise_generator import SEMNoiseModel

from experiment_common import (
    GROUNDTRUTH_MAP, GROUNDTRUTH_NAMES, _ground_truth_path, log,
    debug_images_dict, RunConfig, run_sampler, BASE_CSV_FIELDNAMES,
    debug_output_dir, DEBUG_OUTPUT_ROOT, PublicationOptions,
    write_results, LINE_PROFILE_ENABLED,
)
from experiment_run_manager import (
    ExperimentRun, ExperimentRunManager, ExperimentStatus,
    create_experiments_from_parameter_lists,
)

logging.basicConfig(level=logging.INFO)

# --------------------
# CONFIG
# --------------------
INTERPOLATION_METHODS: List[str] = ["cubic"]

SCANNED_PIXELS_PERCENTAGES: List[float] = [0.5]
ALPHAS: List[Optional[float]] = [0.25, 0.5]
TEMPORAL_SAMPLING_OPTIONS: List[bool] = [True]
TEMPORAL_RECONSTRUCTION_OPTIONS: List[bool] = [True]

TEMPORAL_METHODS: List[str] = ["temporal_variance"]
TEMPORAL_RESIDUAL_CUTOFFS: List[float] = [12.5]#, 25.0, 50.0]
TEMPORAL_RESIDUAL_CONFIDENCE_SCALES: List[float] = [250.0]#, 500.0]
ADAPTIVE_REFINEMENT_FRACTIONS: List[float] = [0.3]
MIN_DENSITY_GAMMAS: List[float] = [0.1]

SAMPLE_SEQUENCES: List[str] = ["halton"] #["uniform", "stratified", "halton"]

DEBUG_IMAGES_ENABLED = True
DEBUG_IMAGES_DICT = (
    debug_images_dict({"reconstruction", "samples", "pdf", "pdf_spatial", "pdf_temporal", "flow", "temporal_variance"})
    # debug_images_dict({"reconstruction", "samples", "pdf"})
    if DEBUG_IMAGES_ENABLED else None
)

#: Run the dose-matched full-raster reference beside the sparse samplers, one
#: run per (dataset, sparsity). It shares the CSV with them, under
#: sampler="low_dwell".
RUN_LOW_DWELL_BASELINE = True

limit_number_of_frames_to = None
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
LOGFILE = "script_log.txt"
CSV_PATH = os.path.join(output_dir, "per_frame_results.csv")
STANDARD_WORKER_POOL_SIZE = 6

# JSON persistence configuration
# Set JSON_MODE directly here:
#   ExperimentRunManager.NO_JSON - No JSON persistence (original behavior)
#   ExperimentRunManager.USE_ONLY - Use only JSON file, skip assembly, run only unfinished
#   ExperimentRunManager.USE_AND_UPDATE - Merge assembly with JSON, filter finished, add new configs
JSON_MODE = ExperimentRunManager.USE_AND_UPDATE
JSON_PATH = os.path.join(output_dir, "experiments_state.json")

# Global experiment run manager
EXPERIMENT_MANAGER = None

# Publication ("_hr") debug streams: each enabled debug kind gets a second
# TIFF stack beside its plain one, supersampled by PUBLICATION_SCALE and
# carrying the dataset's own ROI from GROUNDTRUTH_MAP as a zoomed inset.
# None disables them entirely.
#
# frames is worth naming explicitly: a scale-4 page of a 1024x1024 frame is
# ~50 MB before compression, and a figure needs one frame, not a run's worth.
# PUBLICATION_IMAGES = None
PUBLICATION_IMAGES = PublicationOptions(scale=4, frames=(100,101,102,103,104,105,106,107,108,109,110))

RUN_CONFIG = RunConfig(
    limit_number_of_frames_to=limit_number_of_frames_to,
    debug_images_dict=DEBUG_IMAGES_DICT,
    log_path=LOGFILE,
    line_profile_enabled=LINE_PROFILE_ENABLED,
    publication_images=PUBLICATION_IMAGES,
)


# --------------------
# Load noise model
# --------------------
semNoiseModel = SEMNoiseModel()
semNoiseModel.load_model("sem_noise_model.pkl")

# --------------------
# Load video
# --------------------
def load_video(gt_name, limit_number_of_frames_to=None, scanned_pixel_percent=None):
    total_dwell_time = GROUNDTRUTH_MAP[gt_name][1]
    video = get_frames_from_tif(_ground_truth_path(gt_name), frame_limit=limit_number_of_frames_to)
    if video.ndim == 4 and video.shape[-1] == 1:
        video = video.squeeze(-1)
    if scanned_pixel_percent is not None:
        t_high = total_dwell_time
        t_target = (scanned_pixel_percent / 100.0) * t_high
        noisy_video = []
        for frame in video:
            noisy_frame = semNoiseModel.generate_low_dwell_time_image(
                frame, t_high=t_high, t_target=t_target)
            noisy_video.append(noisy_frame)
        video = np.array(noisy_video)
    return video


def run_low_dwell_time_sampler(gt_name, scanned_pixel_percent):
    local_results = []
    log(LOGFILE, f"Starting: LOW-DWELL | {gt_name} | S={scanned_pixel_percent}%")
    try:
        gt_video = load_video(gt_name, limit_number_of_frames_to)
        t_high = GROUNDTRUTH_MAP[gt_name][1]
        s = scanned_pixel_percent / 100.0
        t_target = s * t_high
        rec_video = []
        PSNRs = []
        SSIMs = []
        example_dir = os.path.join(DEBUG_OUTPUT_ROOT, "low_dwell", gt_name,
                                   f"sparsity_{scanned_pixel_percent}")
        os.makedirs(example_dir, exist_ok=True)
        for i, frame in enumerate(gt_video):
            noisy_frame = semNoiseModel.generate_low_dwell_time_image(
                frame, t_high=t_high, t_target=t_target)
            rec_video.append(noisy_frame)
            psnr = calculate_psnr(frame, noisy_frame)
            ssim, _grad, _full = calculate_ssim(frame, noisy_frame)
            PSNRs.append(psnr)
            SSIMs.append(ssim)
            tifffile.imwrite(os.path.join(example_dir, f"frame_{i:03d}_low_dwell.tiff"), noisy_frame)
            tifffile.imwrite(os.path.join(example_dir, f"frame_{i:03d}_abs_error_map.tiff"),
                             save_error_map(frame, noisy_frame))
            tifffile.imwrite(os.path.join(example_dir, f"frame_{i:03d}_pixelwise_psnr.tiff"),
                             save_pixel_wise_psnr_plots(frame, noisy_frame))
        rec_video = np.array(rec_video)
        T = rec_video.shape[0]
        for frame_idx in range(T):
            local_results.append({
                "sampler": "low_dwell", "withTemporalSampler": None,
                "withTemporalReconstruction": None, "gt_name": gt_name,
                "scanned_pixel_percent": scanned_pixel_percent, "frame_idx": frame_idx,
                "PSNR": PSNRs[frame_idx], "SSIM": SSIMs[frame_idx], "alpha": None,
                "beta": None, "adaptiveFraction": None, "minDensityGamma": None,
                "sampleSequence": None, "temporalMethod": None,
                "temporalResidualCutoff": None, "temporalResidualConfidenceScale": None,
            })
        log(LOGFILE, f"[DONE] LOW-DWELL | {gt_name} | S={scanned_pixel_percent}%")
    except Exception as e:
        log(LOGFILE, f"[ERROR] LOW-DWELL | {gt_name} | S={scanned_pixel_percent}% | {e}\n"
                     f"{traceback.format_exc()}")
    return local_results


# --------------------
# Simple worker function - no status updates (happens in main thread)
# --------------------
def run_sampler_worker(config, experiment):
    """Worker function that just runs the sampler and returns result."""
    task = experiment.to_tuple()
    example_dir = debug_output_dir(
        experiment.gt_name, experiment.scanned_pixel_percent,
        experiment.alpha, experiment.adaptive_fraction,
        experiment.temporal_residual_cutoff,
        experiment.temporal_residual_confidence_scale,
        experiment.sample_sequence)
    try:
        result = run_sampler(config, *task)
        return (experiment, result, example_dir, None)
    except Exception as e:
        error_msg = f"{e}\n{traceback.format_exc()}"
        return (experiment, None, None, error_msg)


# --------------------
# Main
# --------------------
def build_experiment_list():
    experiments = []
    experiments.extend(create_experiments_from_parameter_lists(
        gt_names=GROUNDTRUTH_NAMES,
        scanned_pixel_percentages=SCANNED_PIXELS_PERCENTAGES,
        sampler_types=["adaptive"],
        interpol_methods=INTERPOLATION_METHODS,
        has_temporal_samplers=TEMPORAL_SAMPLING_OPTIONS,
        has_temporal_reconstructions=TEMPORAL_RECONSTRUCTION_OPTIONS,
        alphas=ALPHAS,
        adaptive_fractions=ADAPTIVE_REFINEMENT_FRACTIONS,
        min_density_gammas=MIN_DENSITY_GAMMAS,
        temporal_methods=TEMPORAL_METHODS,
        temporal_residual_cutoffs=TEMPORAL_RESIDUAL_CUTOFFS,
        temporal_residual_confidence_scales=TEMPORAL_RESIDUAL_CONFIDENCE_SCALES,
        sample_sequences=SAMPLE_SEQUENCES,
    ))
    return experiments


def main():
    global EXPERIMENT_MANAGER
    t_experiment_start = time.perf_counter()

    if os.path.exists(LOGFILE):
        os.remove(LOGFILE)
    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)

    # Initialize experiment run manager
    EXPERIMENT_MANAGER = ExperimentRunManager(json_path=JSON_PATH, mode=JSON_MODE)

    # Build experiment list
    assembled_experiments = build_experiment_list()
    experiments_to_run = EXPERIMENT_MANAGER.initialize(assembled_experiments)

    log(LOGFILE, f"===== Starting Experiment Run =====")
    log(LOGFILE, f"JSON Mode: {JSON_MODE}")
    log(LOGFILE, f"JSON Path: {JSON_PATH}")
    log(LOGFILE, f"Total experiments: {len(assembled_experiments)}, To run: {len(experiments_to_run)}")
    log(LOGFILE, "===== Starting Parallel Runs =====")

    # Run experiments with status updates in main thread
    # Only submit STANDARD_WORKER_POOL_SIZE at a time to track running state accurately
    with ProcessPoolExecutor(max_workers=STANDARD_WORKER_POOL_SIZE) as executor:
        futures = {}
        remaining_experiments = list(experiments_to_run)
        
        # Submit initial batch
        for experiment in remaining_experiments[:STANDARD_WORKER_POOL_SIZE]:
            future = executor.submit(run_sampler_worker, RUN_CONFIG, experiment)
            futures[future] = experiment
            EXPERIMENT_MANAGER.mark_experiment_started(experiment.experiment_id)
        
        remaining_experiments = remaining_experiments[STANDARD_WORKER_POOL_SIZE:]
        EXPERIMENT_MANAGER.save_if_dirty()

        while futures:
            # Wait for next future to complete
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            
            for future in done:
                experiment = futures[future]
                
                try:
                    exp_result, result, example_dir, error_msg = future.result()
                    if error_msg:
                        EXPERIMENT_MANAGER.mark_experiment_error(experiment.experiment_id, error_msg)
                        log(LOGFILE, f"[WORKER ERROR] {experiment.experiment_id}")
                    elif result:
                        write_results(result, CSV_PATH, BASE_CSV_FIELDNAMES, LOGFILE)
                        EXPERIMENT_MANAGER.mark_experiment_finished(experiment.experiment_id, example_dir)
                    else:
                        log(LOGFILE, f"[WORKER WARNING] No result for {experiment.experiment_id}")
                        EXPERIMENT_MANAGER.mark_experiment_error(experiment.experiment_id, "No result")
                except Exception as e:
                    EXPERIMENT_MANAGER.mark_experiment_error(experiment.experiment_id, str(e))
                    log(LOGFILE, f"[WORKER ERROR] {experiment.experiment_id} | {e}")
                
                # Remove completed future
                del futures[future]
            
            # Submit new experiments to maintain pool size
            while remaining_experiments and len(futures) < STANDARD_WORKER_POOL_SIZE:
                next_experiment = remaining_experiments.pop(0)
                next_future = executor.submit(run_sampler_worker, RUN_CONFIG, next_experiment)
                futures[next_future] = next_experiment
                EXPERIMENT_MANAGER.mark_experiment_started(next_experiment.experiment_id)
            
            EXPERIMENT_MANAGER.save_if_dirty()

    # Save final state (belt and suspenders)
    EXPERIMENT_MANAGER.finalize()
    log(LOGFILE, f"[JSON] Final state saved to {JSON_PATH}")

    # The dose-matched baseline: the same electron budget spent on a full
    # raster scan instead of a sparse one, so every pixel is measured at
    # scanned_pixel_percent of the ground truth's dwell time.
    low_dwell_tasks = ([(gt_name, scanned_pixel_percent)
                        for gt_name in GROUNDTRUTH_NAMES
                        for scanned_pixel_percent in SCANNED_PIXELS_PERCENTAGES]
                       if RUN_LOW_DWELL_BASELINE else [])
    log(LOGFILE, f"===== Low-dwell baseline: {len(low_dwell_tasks)} run(s) =====")
    with ProcessPoolExecutor(max_workers=STANDARD_WORKER_POOL_SIZE) as executor:
        futures = {executor.submit(run_low_dwell_time_sampler, *task): task for task in low_dwell_tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                if result:
                    write_results(result, CSV_PATH, BASE_CSV_FIELDNAMES, LOGFILE)
            except Exception as e:
                log(LOGFILE, f"[LOW DWELL ERROR] {task} | {e}\n{traceback.format_exc()}")

    log(LOGFILE, "===== All Runs Completed =====")
    log(LOGFILE, f"Saved results to {CSV_PATH}")
    t_experiment_end = time.perf_counter()
    log(LOGFILE, f"[TIMING] Total: {t_experiment_end - t_experiment_start:.2f}s")

    # Print summary
    all_exp = EXPERIMENT_MANAGER.get_all_experiments()
    completed = sum(1 for e in all_exp if e.status == ExperimentStatus.FINISHED)
    errors = sum(1 for e in all_exp if e.status == ExperimentStatus.ERROR)
    not_started = sum(1 for e in all_exp if e.status == ExperimentStatus.NOT_STARTED)
    log(LOGFILE, f"[SUMMARY] Total: {len(all_exp)}, Finished: {completed}, Errors: {errors}, Not Started: {not_started}")


if __name__ == "__main__":
    main()
