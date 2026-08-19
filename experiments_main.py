import os
import time

from tifffile import tifffile

import numpy as np
import pandas as pd
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import logging
import traceback
from datetime import datetime

# stads is a sibling repository installed with `pip install -e`, so import it
# as a normal package rather than reaching into a vendored copy.
from stads.stads import AdaptiveSampler
from stads.stratified_sampler import StratifiedSampler
from stads.debug_images import save_error_map, save_pixel_wise_psnr_plots
from stads.evaluation import calculate_psnr, calculate_ssim
from stads.read_images import get_frames_from_tif
# Datasets are not downloaded on demand any more: run
# `python -m stads.video_downloader <filename>` first. DEFAULT_SAVE_DIR is
# where that CLI puts them, and the directory GROUNDTRUTH_MAP's filenames are
# resolved against below.
from stads.video_downloader import DEFAULT_SAVE_DIR

from sem_noise_generator import SEMNoiseModel

logging.basicConfig(level=logging.INFO)

# --------------------
# CONFIG
# --------------------
# display name -> (filename under DEFAULT_SAVE_DIR, total dwell time)
GROUNDTRUTH_MAP = {
    # "HYDRATION_ONE": ("Hydration.tif", 25000),
    "LI_EXPULSION_ONE": ("Li_Expulsion_1.tif", 20000),
    # "LI_EXPULSION_TWO": ("Li_Expulsion_2.tif", 20000),
    # "SI_LITHIATION_ONE": ("Si_Lithiation.tif", 20000),
    # "EDS_AEROSPACE_ONE": ("EDS_aerospace_one.tif", 20000),
    # "EDS_AEROSPACE_TWO": ("EDS_aerospace_two.tif", 20000),
    # "TITANIUM_STRAIN_ONE": ("Titanium_strain.tif", 20000)
}

GROUNDTRUTH_NAMES = list(GROUNDTRUTH_MAP.keys())


def _ground_truth_path(gt_name):
    filename, _ = GROUNDTRUTH_MAP[gt_name]
    return str(DEFAULT_SAVE_DIR / filename)

# Interpolation backends swept as separate experiments (adaptive sampler only).
#   "linear" -> barycentric linear      (stads GpuLinearInterpolator)
#   "cubic"  -> Clough-Tocher cubic     (stads GpuCloughTocherInterpolator)
# Both run on the GPU via AdaptiveSampler.interpolate_sparse_image_grid.
# NOTE: this multiplies the adaptive task count by len(INTERPOLATION_METHODS).
INTERPOLATION_METHODS = ["cubic"]

SCANNED_PIXELS_PERCENTAGES = [2.0]#[0.5, 2.0, 5.0, 7.0]#list(np.arange(0.5, 5.5, 0.5)) + [0.1, 7.0, 10.0, 20.0]
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
ADAPTIVE_REFINEMENT_FRACTIONS = [0.3, 0.5] #[0.0, 0.1, 0.3, 0.5]

limit_number_of_frames_to = 20
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
LOGFILE = "script_log.txt"
CSV_PATH = os.path.join(output_dir, "per_frame_results.csv")
STANDARD_WORKER_POOL_SIZE = 6 #6 probably best value for asr-ws-murdock


# --------------------
# Logging helper
# --------------------
def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"{now} | {msg}"
    print(full_msg, flush=True)
    try:
        with open(LOGFILE, "a") as f:
            f.write(full_msg + "\n")
    except Exception:
        pass


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

    log(
        f"Starting: LOW-DWELL | "
        f"{gt_name} | "
        f"S={scanned_pixel_percent}%"
    )

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
            ssim = calculate_ssim(noisy_frame,frame)
            PSNRs.append(psnr)
            SSIMs.append(ssim)
            tifffile.imwrite(os.path.join(example_dir,f"frame_{i:03d}_low_dwell.tiff"),noisy_frame)
            save_error_map(frame,noisy_frame,savePlot=True,savePath=os.path.join(example_dir,
                    f"frame_{i:03d}_abs_error_map.tiff"))
            save_pixel_wise_psnr_plots(frame,noisy_frame,savePlot=True,savePath=os.path.join(example_dir
                    ,f"frame_{i:03d}_pixelwise_psnr.tiff"))

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
                # Every key in CSV_FIELDNAMES has to be present: write_results
                # writes with to_csv(columns=CSV_FIELDNAMES), and a batch
                # containing only these rows would otherwise have no such
                # column at all and raise KeyError.
                "beta": None,
                "adaptiveFraction": None
            })

        log(
            f"[DONE] LOW-DWELL | "
            f"{gt_name} | "
            f"S={scanned_pixel_percent}%"
        )

    except Exception as e:

        log(
            f"[ERROR] LOW-DWELL | "
            f"{gt_name} | "
            f"S={scanned_pixel_percent}% | "
            f"{e}\n{traceback.format_exc()}"
        )

    return local_results

# --------------------
# STADS wrapper
# --------------------
def run_sampler(gt_name, scanned_pixel_percent, sampler_type, interpol_method="linear", has_temporal_sampler=True, has_temporal_reconstruction=True, alpha=None, adaptive_fraction=0.0):
    local_results = []
    t_overall_start = time.perf_counter()

    log(f"Starting: {sampler_type} | interpol={interpol_method} | {gt_name} | S={scanned_pixel_percent}% | SamplerTemporal={has_temporal_sampler}| ReconstructionTemporal={has_temporal_reconstruction} | alpha={alpha} | adaptive={adaptive_fraction}")
    try:
        ground_truth_path = _ground_truth_path(gt_name)
        if sampler_type == "adaptive":
            sampler = AdaptiveSampler(
                initialSampling="stratified",
                boundaryPlacement="border",
                interpolMethod=interpol_method,
                sparsityPercent=scanned_pixel_percent,
                limit_number_of_frames_to=limit_number_of_frames_to,
                groundTruthPath=ground_truth_path,
                alpha=alpha,
                withTemporalSampling=has_temporal_sampler,
                withTemporalReconstruction=has_temporal_reconstruction,
                adaptiveRefinementFraction=adaptive_fraction
            )
        else:
            sampler = StratifiedSampler(
                interpolMethod=interpol_method,
                sparsityPercent=scanned_pixel_percent,
                limit_number_of_frames_to=limit_number_of_frames_to,
                groundTruthPath=ground_truth_path,
            )

        trueNumberOfFrames = sampler.numberOfFrames

        # interpol_method is part of the path: without it a cubic run would
        # overwrite the linear run's figures for the same configuration.
        # adaptive_fraction likewise, or a refined run would overwrite the
        # unrefined one it is meant to be compared against.
        example_dir = os.path.join(output_dir, "examples", sampler_type, f"interpol_{interpol_method}", f"sparsity_{scanned_pixel_percent}", gt_name, f"sampler_{has_temporal_sampler}_reconstruction_{has_temporal_reconstruction}", f"alpha_{alpha}", f"adaptive_{adaptive_fraction}")
        os.makedirs(example_dir, exist_ok=True)

        t_run_start = time.perf_counter()
        # Both sampler families write each frame's enabled debug images as
        # they're produced (see stads' debug_images/) rather than in a
        # separate pass afterward, so this includes that I/O.
        rec_video, PSNRs, SSIMs = sampler.run(save_path=example_dir)
        t_run_end = time.perf_counter()
        log(f"[TIMING] sampler.run(): {t_run_end - t_run_start:.2f}s | {sampler_type} | {gt_name} | S={scanned_pixel_percent}% | alpha={alpha}")

        # Collect results (LOCAL!)
        for frame_idx in range(trueNumberOfFrames):
            local_results.append({
                "sampler": sampler_type,
                "interpolation": interpol_method,
                "withTemporalSampler": has_temporal_sampler if sampler_type == "adaptive" else False,
                "withTemporalReconstruction": has_temporal_reconstruction if sampler_type == "adaptive" else False,
                "gt_name": gt_name,
                "scanned_pixel_percent": scanned_pixel_percent,
                "frame_idx": frame_idx,
                "PSNR": PSNRs[frame_idx],
                "SSIM": SSIMs[frame_idx],
                "alpha": alpha if (sampler_type == "adaptive" and has_temporal_reconstruction) else None,
                "beta": alpha if (sampler_type == "adaptive" and has_temporal_reconstruction) else None,
                "adaptiveFraction": adaptive_fraction if sampler_type == "adaptive" else None
            })

        log(f"[DONE] {sampler_type} | interpol={interpol_method} | {gt_name} | S={scanned_pixel_percent}%")

    except Exception as e:
        log(f"[ERROR] {sampler_type} | interpol={interpol_method} | {gt_name} | S={scanned_pixel_percent}% | {e}\n{traceback.format_exc()}")

    t_overall_end = time.perf_counter()
    log(f"[TIMING] run_sampler() total: {t_overall_end - t_overall_start:.2f}s | {sampler_type} | interpol={interpol_method} | {gt_name} | S={scanned_pixel_percent}% | alpha={alpha}")
    return local_results


CSV_FIELDNAMES = ["sampler", "withTemporalSampler", "withTemporalReconstruction", "gt_name",
                   "scanned_pixel_percent", "frame_idx", "PSNR", "SSIM", "alpha", "beta", "adaptiveFraction"]


def write_results(results):
    """Append a batch of per-frame result dicts to CSV_PATH, called directly
    from the main thread as each ProcessPoolExecutor future completes."""
    if not results:
        return
    df = pd.DataFrame(results)
    if not os.path.exists(CSV_PATH):
        log(f"[OUTPUT] Creating new CSV file: {CSV_PATH}")
        df.to_csv(CSV_PATH, index=False, columns=CSV_FIELDNAMES)
    else:
        df.to_csv(CSV_PATH, index=False, mode='a', header=False, columns=CSV_FIELDNAMES)

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
    
    # Task tuples are unpacked positionally into run_sampler(), so the order
    # here must match its signature:
    #   (gt_name, scanned_pixel_percent, sampler_type, interpol_method,
    #    has_temporal_sampler, has_temporal_reconstruction, alpha,
    #    adaptive_fraction)
    for gt_name in GROUNDTRUTH_NAMES:
        for interpol_method in INTERPOLATION_METHODS:
            for use_temporal_sampler in TEMPORAL_SAMPLING_OPTIONS:
                for use_temporal_reconstruction in TEMPORAL_RECONSTRUCTION_OPTIONS:
                    for scanned_pixel_percent in SCANNED_PIXELS_PERCENTAGES:
                        for adaptive_fraction in ADAPTIVE_REFINEMENT_FRACTIONS:
                            if use_temporal_reconstruction:
                                for alpha in ALPHAS:
                                    sampler_tasks.append((gt_name, scanned_pixel_percent, "adaptive", interpol_method, use_temporal_sampler, use_temporal_reconstruction, alpha, adaptive_fraction))
                            else:
                                sampler_tasks.append((gt_name, scanned_pixel_percent, "adaptive", interpol_method, use_temporal_sampler, use_temporal_reconstruction, 1.0, adaptive_fraction)) # alpha is irrelevant when temporal reconstruction is disabled
    
    
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

    log("===== Starting Parallel Runs =====")

    # Sampler tasks run in separate processes (bypasses GIL for CPU-bound work).
    with ProcessPoolExecutor(max_workers=STANDARD_WORKER_POOL_SIZE) as executor:
        futures = {executor.submit(run_sampler, *task): task for task in sampler_tasks}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                if result:
                    write_results(result)
                else:
                    log(f"[WORKER WARNING] No result for {task}")
            except Exception as e:
                log(f"[WORKER ERROR] {task} | {e}\n{traceback.format_exc()}")

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
                    write_results(result)
                else:
                    log(f"[LOW DWELL WARNING] No result for {task}")
            except Exception as e:
                log(f"[LOW DWELL ERROR] {task} | {e}\n{traceback.format_exc()}")

    log("===== All Runs Completed =====")
    log(f"Saved per-frame results to {CSV_PATH}")
    t_experiment_end = time.perf_counter()
    log(f"[TIMING] Experiment total: {t_experiment_end - t_experiment_start:.2f}s using {STANDARD_WORKER_POOL_SIZE} sampler workers")

if __name__ == "__main__":
    main()