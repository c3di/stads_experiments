import os
import time

from tifffile import tifffile

import numpy as np
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
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

SCANNED_PIXELS_PERCENTAGES = [0.5] # [0.05, 0.1, 0.5, 2.0, 5.0, 7.0]
ALPHAS = [2.0] #[0.25,0.5, 1.0, 2.0, 4.0]#list(np.arange(0.5, 5.5, 0.5))
BETAS = []
TEMPORAL_SAMPLING_OPTIONS = [True]
TEMPORAL_RECONSTRUCTION_OPTIONS = [True]

# True adaptive sampling: the share of each frame's budget held back from the
# PDF draw and spent inside the frame, subdividing the triangulation edges this
# frame's own samples say are worst. 0.0 is the previous behaviour exactly.
# The total number of acquired pixels is unchanged either way, so runs at
# different fractions are comparable at equal budget.
# NOTE: this multiplies the adaptive task count by len(ADAPTIVE_REFINEMENT_FRACTIONS).
ADAPTIVE_REFINEMENT_FRACTIONS = [0.1] #[0.0, 0.1, 0.3, 0.5]

# Probability mass mixed into every pdf as a uniform floor -- see
# AdaptiveSampler's minDensityGamma / pdf_blend.apply_minimum_density. 0.0 is
# the previous behaviour exactly (no floor).
# NOTE: this multiplies the adaptive task count by len(MIN_DENSITY_GAMMAS).
MIN_DENSITY_GAMMAS = [0.0]

DEBUG_IMAGES_ENABLED = True
DEBUG_IMAGES_DICT = (
    debug_images_dict({"reconstruction", "samples", "pdf"})
    if DEBUG_IMAGES_ENABLED else None
)

limit_number_of_frames_to = 500
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
LOGFILE = "script_log.txt"
CSV_PATH = os.path.join(output_dir, "per_frame_results.csv")
STANDARD_WORKER_POOL_SIZE = 6 #6 probably best value for asr-ws-murdock

# Line-by-line profiling of the pdf and overlay_masks phases (see stads.py's
# [PHASE-TOTAL] log), opt-in via STADS_LINE_PROFILE=1 so a normal run's
# timing is unaffected. Runs inside the worker process, around the same
# sampler.run() call an unprofiled run makes -- not a separate harness.
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
                "minDensityGamma": None
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
# Main
# --------------------
def main():
    t_experiment_start = time.perf_counter()

    if os.path.exists(LOGFILE):
        os.remove(LOGFILE)

    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)

    # Build sampler task list
    #experimental conditions: Main method: adaptive, with/without temporal sampler, with/without temporal reconstruction
    sampler_tasks = []

    # Task tuples are unpacked positionally into run_sampler() (after the
    # leading RunConfig), so the order here must match its signature:
    #   (gt_name, scanned_pixel_percent, sampler_type, interpol_method,
    #    has_temporal_sampler, has_temporal_reconstruction, alpha,
    #    adaptive_fraction, min_density_gamma)
    for gt_name in GROUNDTRUTH_NAMES:
        for interpol_method in INTERPOLATION_METHODS:
            for use_temporal_sampler in TEMPORAL_SAMPLING_OPTIONS:
                for use_temporal_reconstruction in TEMPORAL_RECONSTRUCTION_OPTIONS:
                    for scanned_pixel_percent in SCANNED_PIXELS_PERCENTAGES:
                        for adaptive_fraction in ADAPTIVE_REFINEMENT_FRACTIONS:
                            for min_density_gamma in MIN_DENSITY_GAMMAS:
                                if use_temporal_reconstruction:
                                    for alpha in ALPHAS:
                                        sampler_tasks.append((gt_name, scanned_pixel_percent, "adaptive", interpol_method, use_temporal_sampler, use_temporal_reconstruction, alpha, adaptive_fraction, min_density_gamma))
                                else:
                                    sampler_tasks.append((gt_name, scanned_pixel_percent, "adaptive", interpol_method, use_temporal_sampler, use_temporal_reconstruction, 1.0, adaptive_fraction, min_density_gamma)) # alpha is irrelevant when temporal reconstruction is disabled


    # Add stratified sampler tasks (no temporal options, no alpha).
    # Not swept over INTERPOLATION_METHODS: this baseline goes through
    # ImageInterpolator (scipy), where "cubic" means griddata's cubic, not
    # Clough-Tocher, so sweeping it here would not compare like with like.
    # Required baseline -- currently disabled only to narrow this particular
    # run to the adaptive sampler; not dead code, don't delete.
    '''
    for gt_name in GROUNDTRUTH_NAMES:
        for scanned_pixel_percent in SCANNED_PIXELS_PERCENTAGES:
            sampler_tasks.append((gt_name, scanned_pixel_percent, "stratified", "linear", False, False, None))
    '''

    log(LOGFILE, "===== Starting Parallel Runs =====")

    # Sampler tasks run in separate processes (bypasses GIL for CPU-bound work).
    with ProcessPoolExecutor(max_workers=STANDARD_WORKER_POOL_SIZE) as executor:
        futures = {executor.submit(run_sampler, RUN_CONFIG, *task): task for task in sampler_tasks}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                if result:
                    write_results(result, CSV_PATH, BASE_CSV_FIELDNAMES, LOGFILE)
                else:
                    log(LOGFILE, f"[WORKER WARNING] No result for {task}")
            except Exception as e:
                log(LOGFILE, f"[WORKER ERROR] {task} | {e}\n{traceback.format_exc()}")

    # Low-dwell tasks run in separate processes using the same completion handling.
    # Required baseline -- currently disabled only to narrow this particular
    # run to the adaptive sampler; not dead code, don't delete.
    '''
    for gt_name in GROUNDTRUTH_NAMES:
        for scanned_pixel_percent in SCANNED_PIXELS_PERCENTAGES:
            low_dwell_tasks.append((gt_name, scanned_pixel_percent))
    '''
    low_dwell_tasks = []

    with ProcessPoolExecutor(max_workers=STANDARD_WORKER_POOL_SIZE) as executor:
        futures = {executor.submit(run_low_dwell_time_sampler, *task): task for task in low_dwell_tasks}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                if result:
                    write_results(result, CSV_PATH, BASE_CSV_FIELDNAMES, LOGFILE)
                else:
                    log(LOGFILE, f"[LOW DWELL WARNING] No result for {task}")
            except Exception as e:
                log(LOGFILE, f"[LOW DWELL ERROR] {task} | {e}\n{traceback.format_exc()}")

    log(LOGFILE, "===== All Runs Completed =====")
    log(LOGFILE, f"Saved per-frame results to {CSV_PATH}")
    t_experiment_end = time.perf_counter()
    log(LOGFILE, f"[TIMING] Experiment total: {t_experiment_end - t_experiment_start:.2f}s using {STANDARD_WORKER_POOL_SIZE} sampler workers")

if __name__ == "__main__":
    main()
