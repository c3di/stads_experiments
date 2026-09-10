import os
import time
import threading

from tifffile import tifffile

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import traceback

from stads.debug_images import save_error_map, save_pixel_wise_psnr_plots
from stads.evaluation import calculate_psnr, calculate_ssim
from stads.read_images import get_frames_from_tif
from sem_noise_generator import SEMNoiseModel

from experiment_common import (
    GROUNDTRUTH_MAP, GROUNDTRUTH_NAMES, _ground_truth_path, log,
    debug_images_dict, RunConfig, run_sampler, BASE_CSV_FIELDNAMES,
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
INTERPOLATION_METHODS = ["cubic"]

SCANNED_PIXELS_PERCENTAGES = [0.1, 1.0]
ALPHAS = [0.25, 0.5, 1.0]
TEMPORAL_SAMPLING_OPTIONS = [True]
TEMPORAL_RECONSTRUCTION_OPTIONS = [True]

TEMPORAL_METHODS = ["temporal_variance"]
TEMPORAL_RESIDUAL_CUTOFFS = [12.0, 25.0, 50.0]
TEMPORAL_RESIDUAL_CONFIDENCE_SCALES = [100.0, 250.0, 500.0]
ADAPTIVE_REFINEMENT_FRACTIONS = [0.0, 0.1, 0.3, 0.5]
MIN_DENSITY_GAMMAS = [0.1]

SAMPLE_SEQUENCES = ["uniform", "stratified", "halton"]

DEBUG_IMAGES_ENABLED = True
DEBUG_IMAGES_DICT = (
    debug_images_dict({"reconstruction", "samples", "pdf", "pdf_spatial", "pdf_temporal", "flow", "temporal_variance"})
    if DEBUG_IMAGES_ENABLED else None
)

limit_number_of_frames_to = None
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
LOGFILE = "script_log.txt"
CSV_PATH = os.path.join(output_dir, "per_frame_results.csv")
STANDARD_WORKER_POOL_SIZE = 2

# JSON persistence configuration
# Set JSON_MODE directly here:
#   ExperimentRunManager.NO_JSON - No JSON persistence (original behavior)
#   ExperimentRunManager.USE_ONLY - Use only JSON file, skip assembly, run only unfinished
#   ExperimentRunManager.USE_AND_UPDATE - Merge assembly with JSON, filter finished, add new configs
JSON_MODE = ExperimentRunManager.NO_JSON
JSON_PATH = os.path.join(output_dir, "experiments_state.json")

# Global experiment run manager
EXPERIMENT_MANAGER = None

RUN_CONFIG = RunConfig(
    output_dir=output_dir,
    limit_number_of_frames_to=limit_number_of_frames_to,
    debug_images_dict=DEBUG_IMAGES_DICT,
    log_path=LOGFILE,
    line_profile_enabled=LINE_PROFILE_ENABLED,
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
    _, total_dwell_time = GROUNDTRUTH_MAP[gt_name]
    video = get_frames_from_tif(_ground_truth_path(gt_name), frame_limit=limit_number_of_frames_to)
    if video.ndim == 4 and video.shape[-1] == 1:
        video = video.squeeze(-1)
    if scanned_pixel_percent is not None:
        t_high = total_dwell_time
        t_target = (scanned_pixel_percent / 100.0) * t_high
        noisy_video = []
        for frame in video:
            noisy_frame = semNoiseModel.generate_low_dwell_time_image(frame, t_high=t_high, t_target=t_target)
            noisy_video.append(noisy_frame)
        video = np.array(noisy_video)
    return video


def run_low_dwell_time_sampler(gt_name, scanned_pixel_percent):
    local_results = []
    log(LOGFILE, f"Starting: LOW-DWELL | {gt_name} | S={scanned_pixel_percent}%")
    try:
        gt_video = load_video(gt_name, limit_number_of_frames_to)
        _, t_high = GROUNDTRUTH_MAP[gt_name]
        s = scanned_pixel_percent / 100.0
        t_target = s * t_high
        rec_video = []
        PSNRs = []
        SSIMs = []
        example_dir = os.path.join(output_dir, "examples", "low_dwell", f"sparsity_{scanned_pixel_percent}", gt_name)
        os.makedirs(example_dir, exist_ok=True)
        for i, frame in enumerate(gt_video):
            noisy_frame = semNoiseModel.generate_low_dwell_time_image(frame, t_high=t_high, t_target=t_target)
            rec_video.append(noisy_frame)
            psnr = calculate_psnr(frame, noisy_frame)
            ssim = calculate_ssim(frame, noisy_frame)
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
        log(LOGFILE, f"[ERROR] LOW-DWELL | {gt_name} | S={scanned_pixel_percent}% | {e}")
    return local_results


# --------------------
# Simple worker function - no status updates (happens in main thread)
# --------------------
def run_sampler_worker(config, experiment):
    """Worker function that just runs the sampler and returns result."""
    task = experiment.to_tuple()
    example_dir = os.path.join(
        config.output_dir, "examples", experiment.sampler_type,
        f"interpol_{experiment.interpol_method}",
        f"sparsity_{experiment.scanned_pixel_percent}", experiment.gt_name,
        f"sampler_{experiment.has_temporal_sampler}_reconstruction_{experiment.has_temporal_reconstruction}",
        f"temporalMethod_{experiment.temporal_method}",
        f"temporalResidualCutoff_{experiment.temporal_residual_cutoff}",
        f"temporalResidualConfidenceScale_{experiment.temporal_residual_confidence_scale}",
        f"sampleSequence_{experiment.sample_sequence}",
        f"alpha_{experiment.alpha}", f"adaptive_{experiment.adaptive_fraction}"
    )
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
    with ProcessPoolExecutor(max_workers=STANDARD_WORKER_POOL_SIZE) as executor:
        futures = {}
        for experiment in experiments_to_run:
            # Mark as started BEFORE submitting to worker
            EXPERIMENT_MANAGER.mark_experiment_started(experiment.experiment_id)
            future = executor.submit(run_sampler_worker, RUN_CONFIG, experiment)
            futures[future] = experiment

        for future in as_completed(futures):
            experiment = futures[future]
            try:
                exp_result, result, example_dir, error_msg = future.result()
                if error_msg:
                    # Worker had an exception
                    EXPERIMENT_MANAGER.mark_experiment_error(experiment.experiment_id, error_msg)
                    log(LOGFILE, f"[WORKER ERROR] {experiment.experiment_id} | {error_msg[:100]}")
                elif result:
                    write_results(result, CSV_PATH, BASE_CSV_FIELDNAMES, LOGFILE)
                    EXPERIMENT_MANAGER.mark_experiment_finished(experiment.experiment_id, example_dir)
                else:
                    log(LOGFILE, f"[WORKER WARNING] No result for {experiment.experiment_id}")
                    EXPERIMENT_MANAGER.mark_experiment_error(experiment.experiment_id, "No result")
            except Exception as e:
                EXPERIMENT_MANAGER.mark_experiment_error(experiment.experiment_id, str(e))
                log(LOGFILE, f"[WORKER ERROR] {experiment.experiment_id} | {e}")
            
            # Periodically save state
            if len(experiments_to_run) > 10:
                EXPERIMENT_MANAGER._save_to_json()

    # Save final state
    EXPERIMENT_MANAGER._save_to_json()

    # Low-dwell tasks (currently disabled)
    low_dwell_tasks = []
    with ProcessPoolExecutor(max_workers=STANDARD_WORKER_POOL_SIZE) as executor:
        futures = {executor.submit(run_low_dwell_time_sampler, *task): task for task in low_dwell_tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                if result:
                    write_results(result, CSV_PATH, BASE_CSV_FIELDNAMES, LOGFILE)
            except Exception as e:
                log(LOGFILE, f"[LOW DWELL ERROR] {task}")

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
