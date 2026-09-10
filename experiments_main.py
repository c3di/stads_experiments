import os
import time
import threading
from typing import Optional, List

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
    create_experiments_from_parameter_lists, get_json_mode_from_env, get_json_path_from_env,
)

logging.basicConfig(level=logging.INFO)

# --------------------
# CONFIG
# --------------------
# Interpolation backends swept as separate experiments (adaptive sampler only).
#   "linear" -> barycentric linear      (stads GpuLinearInterpolator)
#   "cubic"  -> Clough-Tocher cubic     (stads GpuCloughTocherInterpolator)
# Both run on the GPU via AdaptiveSampler.interpolate_sparse_image_grid.
# NOTE: this multiplies the adaptive task count by len(INTERPOLATION_METHODS).
INTERPOLATION_METHODS = ["cubic"]

SCANNED_PIXELS_PERCENTAGES = [0.1, 1.0] # [0.1, 0.5, 1.0, 2.0, 5.0]
ALPHAS = [0.25, 0.5, 1.0] # [0.25, 0.5, 1.0, 5.0, 10.0]
TEMPORAL_SAMPLING_OPTIONS = [True]
TEMPORAL_RECONSTRUCTION_OPTIONS = [True]

TEMPORAL_METHODS = ["temporal_variance"]#, "optical_flow"]  # ["optical_flow", "temporal_variance"]
TEMPORAL_RESIDUAL_CUTOFFS = [12.0, 25.0, 50.0]
TEMPORAL_RESIDUAL_CONFIDENCE_SCALES = [100.0, 250.0, 500.0]
ADAPTIVE_REFINEMENT_FRACTIONS = [0.0, 0.1, 0.3, 0.5]
MIN_DENSITY_GAMMAS = [0.1]

SAMPLE_SEQUENCES = ["uniform", "stratified", "halton"]

DEBUG_IMAGES_ENABLED = True
DEBUG_IMAGES_DICT = (
        debug_images_dict({"reconstruction", "samples", "pdf", "pdf_spatial", "pdf_temporal", "flow", "temporal_variance"})
        # debug_images_dict({"reconstruction", "samples", "pdf"})
    if DEBUG_IMAGES_ENABLED else None
)

limit_number_of_frames_to = None
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
LOGFILE = "script_log.txt"
CSV_PATH = os.path.join(output_dir, "per_frame_results.csv")
STANDARD_WORKER_POOL_SIZE = 2 #11 seems good value for RTX 5000 ada laptop - but not for the 'big' GT (aerospace + titanium)

# JSON persistence configuration
# Set JSON_MODE directly here, or use environment variables:
#   "no_json" - No JSON persistence (original behavior)
#   "use_only" - Use only JSON file, skip assembly, run only unfinished
#   "use_and_update" - Merge assembly with JSON, filter finished, add new configs
JSON_MODE = ExperimentRunManager.USE_AND_UPDATE  # Change this line to configure
JSON_PATH = os.path.join(output_dir, "experiments_state.json")

# Override from environment variables if set
if os.environ.get("STADS_JSON_MODE"):
    JSON_MODE = get_json_mode_from_env()
if os.environ.get("STADS_JSON_PATH"):
    JSON_PATH = get_json_path_from_env(JSON_PATH)

# Global experiment run manager (will be initialized in main)
EXPERIMENT_MANAGER: Optional[ExperimentRunManager] = None
EXPERIMENT_LOCK = threading.Lock()  # For thread-safe status updates

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
            noisy_frame = semNoiseModel.generate_low_dwell_time_image(frame,t_high=t_high,t_target=t_target)
            noisy_video.append(noisy_frame)
        video = np.array(noisy_video)
    return video


def run_low_dwell_time_sampler(gt_name, scanned_pixel_percent):
    local_results = []

    log(LOGFILE,
        f"Starting: LOW-DWELL | "
        f"{gt_name} | "
        f"S={scanned_pixel_percent}%")

    try:

        gt_video = load_video(gt_name, limit_number_of_frames_to)
        _, t_high = GROUNDTRUTH_MAP[gt_name]
        s = scanned_pixel_percent / 100.0
        t_target = s * t_high
        rec_video = []

        PSNRs = []
        SSIMs = []
        # Save figures
        example_dir = os.path.join(output_dir, "examples", "low_dwell", f"sparsity_{scanned_pixel_percent}", gt_name)
        os.makedirs(example_dir, exist_ok=True)

        for i,frame in enumerate(gt_video):

            noisy_frame = semNoiseModel.generate_low_dwell_time_image(frame,t_high=t_high,t_target=t_target)
            rec_video.append(noisy_frame)

            psnr = calculate_psnr(frame, noisy_frame)
            ssim = calculate_ssim(frame, noisy_frame)
            PSNRs.append(psnr)
            SSIMs.append(ssim)
            tifffile.imwrite(os.path.join(example_dir,f"frame_{i:03d}_low_dwell.tiff"),noisy_frame)
            tifffile.imwrite(os.path.join(example_dir, f"frame_{i:03d}_abs_error_map.tiff"),
                             save_error_map(frame, noisy_frame))
            tifffile.imwrite(os.path.join(example_dir, f"frame_{i:03d}_pixelwise_psnr.tiff"),
                             save_pixel_wise_psnr_plots(frame, noisy_frame))

        rec_video = np.array(rec_video)
        T = rec_video.shape[0]

        for frame_idx in range(T):

            local_results.append({
                "sampler": "low_dwell",
                "withTemporalSampler": None,
                "withTemporalReconstruction": None,
                "gt_name": gt_name,
                "scanned_pixel_percent": scanned_pixel_percent,
                "frame_idx": frame_idx,
                "PSNR": PSNRs[frame_idx],
                "SSIM": SSIMs[frame_idx],
                "alpha": None,
                # Every key in BASE_CSV_FIELDNAMES has to be present:
                # write_results writes with to_csv(columns=fieldnames), and a
                # batch containing only these rows would otherwise have no
                # such column at all and raise KeyError.
                "beta": None,
                "adaptiveFraction": None,
                "minDensityGamma": None,
                "sampleSequence": None,
                "temporalMethod": None,
                "temporalResidualCutoff": None,
                "temporalResidualConfidenceScale": None,
            })

        log(LOGFILE,
            f"[DONE] LOW-DWELL | "
            f"{gt_name} | "
            f"S={scanned_pixel_percent}%")

    except Exception as e:

        log(LOGFILE,
            f"[ERROR] LOW-DWELL | "
            f"{gt_name} | "
            f"S={scanned_pixel_percent}% | "
            f"{e}\n{traceback.format_exc()}")

    return local_results


# --------------------
# Helper function to update experiment status (called from worker threads)
# --------------------
def update_experiment_status(experiment_id: str, status: str, error_message: str | None = None, result_path: str | None = None, log_path: str | None = None):
    """Update experiment status in a thread-safe manner.
    
    This function is called at the same level as write_results to ensure
    consistency and prevent race conditions with ProcessPoolExecutor.
    
    Args:
        experiment_id: The experiment ID
        status: One of 'started', 'finished', 'error'
        error_message: Error message if status is 'error'
        result_path: Result path if status is 'finished'
        log_path: Path to log file
    """
    global EXPERIMENT_MANAGER
    
    if EXPERIMENT_MANAGER is None:
        return
    
    with EXPERIMENT_LOCK:
        if status == 'started':
            EXPERIMENT_MANAGER.mark_experiment_started(experiment_id)
        elif status == 'finished':
            EXPERIMENT_MANAGER.mark_experiment_finished(experiment_id, result_path)
        elif status == 'error':
            EXPERIMENT_MANAGER.mark_experiment_error(experiment_id, error_message or "Unknown error")
        
        # Periodically save to prevent data loss
        if EXPERIMENT_MANAGER._dirty:
            try:
                EXPERIMENT_MANAGER._save_to_json()
            except Exception as e:
                if log_path:
                    log(log_path, f"[WARNING] Could not save experiment state: {e}")


# --------------------
# Wrapper for run_sampler that handles experiment status updates
# --------------------
def run_sampler_with_status_update(config: RunConfig, experiment: ExperimentRun):
    """Run a sampler with experiment status tracking.
    
    This wrapper calls run_sampler and updates the experiment status
    at the same level to prevent race conditions.
    """
    global EXPERIMENT_MANAGER
    
    # Convert experiment to tuple for run_sampler
    task = experiment.to_tuple()
    
    # Mark as started
    update_experiment_status(
        experiment.experiment_id, 
        'started',
        log_path=config.log_path
    )
    
    try:
        # Create example directory based on experiment parameters
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
        
        result = run_sampler(config, *task)
        
        # Mark as finished
        update_experiment_status(
            experiment.experiment_id,
            'finished',
            result_path=example_dir,
            log_path=config.log_path
        )
        
        return result
        
    except Exception as e:
        # Mark as error
        error_msg = f"{e}\n{traceback.format_exc()}"
        update_experiment_status(
            experiment.experiment_id,
            'error',
            error_message=error_msg,
            log_path=config.log_path
        )
        raise  # Re-raise so the error is handled by the caller


# --------------------
# Main
# --------------------
def build_experiment_list() -> List[ExperimentRun]:
    """Build the list of experiments using the factory method.
    
    This replaces the nested for-loop tuple generation with a cleaner
    approach that returns ExperimentRun objects.
    """
    experiments = []
    
    # Build experiments using the factory method
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
    
    # Add stratified sampler tasks (no temporal options, no alpha)
    # Not swept over INTERPOLATION_METHODS: this baseline goes through
    # ImageInterpolator (scipy), where "cubic" means griddata's cubic, not
    # Clough-Tocher, so sweeping it here would not compare like with like.
    # Required baseline -- currently disabled only to narrow this particular
    # run to the adaptive sampler; not dead code, don't delete.
    '''
    for gt_name in GROUNDTRUTH_NAMES:
        for scanned_pixel_percent in SCANNED_PIXELS_PERCENTAGES:
            experiments.append(create_experiment_from_parameters(
                gt_name=gt_name,
                scanned_pixel_percent=scanned_pixel_percent,
                sampler_type="stratified",
                interpol_method="linear",
                has_temporal_sampler=False,
                has_temporal_reconstruction=False,
                alpha=None,
                adaptive_fraction=0.0,
                min_density_gamma=0.0,
                temporal_method="optical_flow",
                temporal_residual_cutoff=12.0,
                temporal_residual_confidence_scale=100.0,
                sample_sequence="stratified",
            ))
    '''
    
    return experiments


def main():
    global EXPERIMENT_MANAGER
    
    t_experiment_start = time.perf_counter()

    # Initialize experiment run manager based on JSON mode
    EXPERIMENT_MANAGER = ExperimentRunManager(json_path=JSON_PATH, mode=JSON_MODE)
    
    if os.path.exists(LOGFILE):
        os.remove(LOGFILE)

    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)

    # Build experiment list using the new factory approach
    assembled_experiments = build_experiment_list()
    
    # Initialize the manager and get the experiments to run
    experiments_to_run = EXPERIMENT_MANAGER.initialize(assembled_experiments)
    
    # Log the configuration
    log(LOGFILE, f"===== Starting Experiment Run =====")
    log(LOGFILE, f"JSON Mode: {JSON_MODE}")
    log(LOGFILE, f"JSON Path: {JSON_PATH}")
    log(LOGFILE, f"Total assembled experiments: {len(assembled_experiments)}")
    log(LOGFILE, f"Experiments to run: {len(experiments_to_run)}")
    log(LOGFILE, f"Worker pool size: {STANDARD_WORKER_POOL_SIZE}")

    log(LOGFILE, "===== Starting Parallel Runs =====")

    # Sampler tasks run in separate processes (bypasses GIL for CPU-bound work).
    # Use the new experiment objects instead of tuples
    with ProcessPoolExecutor(max_workers=STANDARD_WORKER_POOL_SIZE) as executor:
        # Create a mapping from future to experiment
        futures = {}
        for experiment in experiments_to_run:
            # Use the wrapper that handles status updates
            future = executor.submit(
                run_sampler_with_status_update, 
                RUN_CONFIG, 
                experiment
            )
            futures[future] = experiment
        
        for future in as_completed(futures):
            experiment = futures[future]
            try:
                result = future.result()
                if result:
                    write_results(result, CSV_PATH, BASE_CSV_FIELDNAMES, LOGFILE)
                else:
                    log(LOGFILE, f"[WORKER WARNING] No result for experiment {experiment.experiment_id}")
                    # Ensure status is updated if no result
                    update_experiment_status(
                        experiment.experiment_id,
                        'error',
                        error_message="No result returned",
                        log_path=LOGFILE
                    )
            except Exception as e:
                # The wrapper should have already updated the status to error
                # But we log it here as well for completeness
                log(LOGFILE, f"[WORKER ERROR] Experiment {experiment.experiment_id} | {e}")

    # Low-dwell tasks run in separate processes using the same completion handling.
    # Required baseline -- currently disabled only to narrow this particular
    # run to the adaptive sampler; not dead code, don't delete.
    '''
    low_dwell_tasks = []
    for gt_name in GROUNDTRUTH_NAMES:
        for scanned_pixel_percent in SCANNED_PIXELS_PERCENTAGES:
            low_dwell_tasks.append((gt_name, scanned_pixel_percent))
    
    with ProcessPoolExecutor(max_workers=STANDARD_WORKER_POOL_SIZE) as executor:
        futures = {executor.submit(run_low_dwell_time_sampler, *task): task for task in low_dwell_tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                if result:
                    write_results(result, CSV_PATH, BASE_CSV_FIELDNAMES, LOGFILE)
                else:
                    log(LOGFILE, f"[LOW DWELL WARNING] No result for {task}")
            except Exception as e:
                log(LOGFILE, f"[LOW DWELL ERROR] {task} | {e}\n{traceback.format_exc()}")
    '''
    low_dwell_tasks = []

    with ProcessPoolExecutor(max_workers=STANDARD_WORKER_POOL_SIZE) as executor:
        futures = {executor.submit(run_low_dwell_time_sampler, *task): task for task in low_dwell_tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                if result:
                    write_results(result, CSV_PATH, BASE_CSV_FIELDNAMES, LOGFILE)
                else:
                    log(LOGFILE, f"[LOW DWELL WARNING] No result for {task}")
            except Exception as e:
                log(LOGFILE, f"[LOW DWELL ERROR] {task} | {e}\n{traceback.format_exc()}")

    # Finalize and save experiment state
    if EXPERIMENT_MANAGER:
        EXPERIMENT_MANAGER.finalize()
    
    log(LOGFILE, "===== All Runs Completed =====")
    log(LOGFILE, f"Saved per-frame results to {CSV_PATH}")
    t_experiment_end = time.perf_counter()
    log(LOGFILE, f"[TIMING] Experiment total: {t_experiment_end - t_experiment_start:.2f}s using {STANDARD_WORKER_POOL_SIZE} sampler workers")
    
    # Print summary statistics
    if EXPERIMENT_MANAGER:
        all_exp = EXPERIMENT_MANAGER.get_all_experiments()
        completed = sum(1 for e in all_exp if e.is_complete())
        not_started = sum(1 for e in all_exp if e.status == ExperimentStatus.NOT_STARTED)
        errors = sum(1 for e in all_exp if e.status == ExperimentStatus.ERROR)
        
        log(LOGFILE, f"[SUMMARY] Total: {len(all_exp)}, Completed: {completed}, Not Started: {not_started}, Errors: {errors}")


if __name__ == "__main__":
    main()
