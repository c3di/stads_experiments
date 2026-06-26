import os
import time
import pickle

from tifffile import tifffile

import numpy as np
import pandas as pd
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import queue
import threading
import logging
import traceback
from datetime import datetime
import delauney

from src.stadsadaptivesampler.src.stads.stads import AdaptiveSampler
from src.stadsadaptivesampler.src.stads.stratified_sampler import StratifiedSampler
from src.stadsadaptivesampler.src.stads.monitor import save_error_map, save_pixel_wise_psnr_plots
from src.stadsadaptivesampler.src.stads.config import HYDRATION_ONE, LI_EXPULSION_ONE, LI_EXPULSION_TWO, SI_LITHIATION_ONE, EDS_AEROSPACE_ONE, EDS_AEROSPACE_TWO, TITANIUM_STRAIN_ONE

from src.stadsadaptivesampler.src.stads.evaluation import calculate_psnr, calculate_ssim

import padis_fsr
from padis_fsr import generate_mask_for_frame, run_padis_fsr_video_with_masks
from noise_data_preparation import SEMNoiseDataset
from sem_noise_generator import compute_stats, SEMNoiseModel

logging.basicConfig(level=logging.INFO)

# --------------------
# CONFIG
# --------------------
GROUNDTRUTH_MAP = {
    "hydration_one": (HYDRATION_ONE,25000),
    "li_expulsion_one": (LI_EXPULSION_ONE,20000),
    "li_expulsion_two": (LI_EXPULSION_TWO,20000),
    #"si_lithiation_one": (SI_LITHIATION_ONE,20000),
    "eds_aerospace_one": (EDS_AEROSPACE_ONE,20000),
    "eds_aerospace_two": (EDS_AEROSPACE_TWO,20000),
    "titanium_strain_one": (TITANIUM_STRAIN_ONE,20000)
}

GROUNDTRUTH_NAMES = list(GROUNDTRUTH_MAP.keys())
SCANNED_PIXELS_PERCENTAGES = [0.5, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0]#list(np.arange(0.5, 5.5, 0.5)) + [0.1, 7.0, 10.0, 20.0]
ALPHAS = [0.25,0.5, 1.0, 2.0, 4.0]#list(np.arange(0.5, 5.5, 0.5))
TEMPORAL_SAMPLING_OPTIONS = [True, False]
TEMPORAL_RECONSTRUCTION_OPTIONS = [True, False]

numberOfFrames = 20
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
LOGFILE = "script_log.txt"
CSV_PATH = os.path.join(output_dir, "per_frame_results.csv")
STANDARD_WORKER_POOL_SIZE = 6 #6 probably best value for asr-ws-murdock
PADIS_WORKER_POOL_SIZE = 1


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
def load_video(gt_name, num_frames, scanned_pixel_percent=None):
    video, total_dwell_time = GROUNDTRUTH_MAP[gt_name]

    video = video[:num_frames]

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

        gt_video = load_video(gt_name, numberOfFrames)
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
                "alpha": None
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
def run_sampler(gt_name, scanned_pixel_percent, sampler_type, has_temporal_sampler=True, has_temporal_reconstruction=True, alpha=None):
    local_results = []
    t_overall_start = time.perf_counter()

    log(f"Starting: {sampler_type} | {gt_name} | S={scanned_pixel_percent}% | SamplerTemporal={has_temporal_sampler}| ReconstructionTemporal={has_temporal_reconstruction} | alpha={alpha}")
    try:
        gt_video = load_video(gt_name,numberOfFrames,100.0)
        trueNumberOfFrames = min(gt_video.shape[0],numberOfFrames)

        if sampler_type == "adaptive":
            sampler = AdaptiveSampler(
                initialSampling="stratified",
                interpolMethod="linear",
                sparsityPercent=scanned_pixel_percent,
                numberOfFrames=trueNumberOfFrames,
                groundTruthName=gt_name,
                alpha=alpha,
                withTemporalSampling=has_temporal_sampler,
                withTemporalReconstruction=has_temporal_reconstruction
            )
        else:
            sampler = StratifiedSampler(
                interpolMethod="linear",
                sparsityPercent=scanned_pixel_percent,
                numberOfFrames=trueNumberOfFrames,
                groundTruthName=gt_name,
            )

        t_run_start = time.perf_counter()
        rec_video, PSNRs, SSIMs = sampler.run()
        t_run_end = time.perf_counter()
        log(f"[TIMING] sampler.run(): {t_run_end - t_run_start:.2f}s | {sampler_type} | {gt_name} | S={scanned_pixel_percent}% | alpha={alpha}")

        # Save figures
        example_dir = os.path.join(output_dir, "examples", sampler_type, f"sparsity_{scanned_pixel_percent}", gt_name, f"sampler_{has_temporal_sampler}_reconstruction_{has_temporal_reconstruction}", f"alpha_{alpha}")
        os.makedirs(example_dir, exist_ok=True)

        t_save_start = time.perf_counter()
        failed_figures = False
        for frame_idx in range(trueNumberOfFrames):
            try:
                sampler.save_figures(frameNumber=frame_idx, save_path=example_dir)
            except Exception as e:
                log(f"[ERROR] Failed to save figures for frame {frame_idx} | {gt_name} | {scanned_pixel_percent}% | {sampler_type} | {e}")
                failed_figures = True
        t_save_end = time.perf_counter()
        log(f"[TIMING] save_figures loop: {t_save_end - t_save_start:.2f}s | {sampler_type} | {gt_name} | S={scanned_pixel_percent}% | alpha={alpha} | failed_figures={failed_figures}")

        # Collect results (LOCAL!)
        for frame_idx in range(trueNumberOfFrames):
            local_results.append({
                "sampler": sampler_type,
                "withTemporalSampler": has_temporal_sampler if sampler_type == "adaptive" else False,
                "withTemporalReconstruction": has_temporal_reconstruction if sampler_type == "adaptive" else False,
                "gt_name": gt_name,
                "scanned_pixel_percent": scanned_pixel_percent,
                "frame_idx": frame_idx,
                "PSNR": PSNRs[frame_idx],
                "SSIM": SSIMs[frame_idx],
                "alpha": alpha if (sampler_type == "adaptive" and has_temporal_reconstruction) else None
            })

        log(f"[DONE] {sampler_type} | {gt_name} | S={scanned_pixel_percent}%")

    except Exception as e:
        log(f"[ERROR] {sampler_type} | {gt_name} | S={scanned_pixel_percent}% | {e}\n{traceback.format_exc()}")

    t_overall_end = time.perf_counter()
    log(f"[TIMING] run_sampler() total: {t_overall_end - t_overall_start:.2f}s | {sampler_type} | {gt_name} | S={scanned_pixel_percent}% | alpha={alpha}")
    return local_results


class PadisWorker(threading.Thread):
    def __init__(self, padis_queue, result_queue, group = None, target = None, name = None, args = ..., kwargs = None, *, daemon = None):
        self.padis_queue = padis_queue
        self.result_queue = result_queue
        super().__init__(group, target, name, args, kwargs, daemon=daemon)

    def run(self):
        while True:
            try:
                gt_name, scanned_pixel_percent = self.padis_queue.get(timeout=5)
                result = run_padis(gt_name, scanned_pixel_percent)
                if result:
                    self.result_queue.put(result)
                else:
                    log(f"[PADIS WORKER WARNING] No result for {gt_name} | {scanned_pixel_percent}%")
                self.padis_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                log(f"[PADIS WORKER ERROR] {e}\n{traceback.format_exc()}")
                self.padis_queue.task_done()


class LowDwellWorker(threading.Thread):

    def __init__(self,low_dwell_queue,result_queue,group=None,target=None,name=None, args=..., kwargs=None, *, daemon=None):

        self.low_dwell_queue = low_dwell_queue
        self.result_queue = result_queue
        super().__init__(group,target,name,args,kwargs,daemon=daemon)

    def run(self):

        while True:

            try:
                gt_name, scanned_pixel_percent = self.low_dwell_queue.get(timeout=5)
                result = run_low_dwell_time_sampler(gt_name,scanned_pixel_percent)

                if result:
                    self.result_queue.put(result)

                else:
                    log(f"[LOW DWELL WARNING] No result for {gt_name} | {scanned_pixel_percent}%")
                self.low_dwell_queue.task_done()

            except queue.Empty:
                break

            except Exception as e:

                log(f"[LOW DWELL ERROR] {e}\n{traceback.format_exc()}")
                self.low_dwell_queue.task_done()


class OutputWorker(threading.Thread):
    def __init__(self, result_queue, group = None, target = None, name = None, args = ..., kwargs = None, *, daemon = None):
        self.result_queue = result_queue
        self.fieldnames = ["sampler", "withTemporalSampler", "withTemporalReconstruction", "gt_name", "scanned_pixel_percent", "frame_idx", "PSNR", "SSIM", "alpha"]
        self.csv_path = os.path.join(output_dir, "per_frame_results.csv")

        super().__init__(group, target, name, args, kwargs, daemon=daemon)

    def run(self):
        while True:
            try:
                results = self.result_queue.get(timeout=10)
                if results:
                    df = pd.DataFrame(results)
                    if not os.path.exists(self.csv_path):
                        log(f"[OUTPUT WORKER INFO] Creating new CSV file: {self.csv_path}")
                        df.to_csv(self.csv_path, index=False, columns=self.fieldnames)
                    else:
                        df.to_csv(self.csv_path, index=False, mode='a', header=False, columns=self.fieldnames)
                        log(f"[OUTPUT WORKER INFO] Appended results to CSV file: {self.csv_path}")
                else:
                    if not self.result_queue.empty():
                        log(f"[OUTPUT WORKER WARNING] Received empty result and non-empty queue.")
                    else:
                        log(f"[OUTPUT WORKER] Received empty result, queue is empty. Assuming all tasks are done. Terminating.")
                        break
                self.result_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                log(f"[OUTPUT WORKER ERROR] {e}\n{traceback.format_exc()}")
                self.result_queue.task_done()

# --------------------
# PADIS wrapper
# --------------------
def run_padis(gt_name, scanned_pixel_percent):
    local_results = []

    log(f"Starting: PADIS-FSR | {gt_name} | S={scanned_pixel_percent}%")
    try:
        gt_video = load_video(gt_name, numberOfFrames)
        T, H, W = gt_video.shape

        masks = [generate_mask_for_frame(gt_video[t], scanned_pixel_percent).astype(np.uint8) for t in range(T)]
        rec_video, psnrs, ssims = run_padis_fsr_video_with_masks(gt_video, masks)

        # Save figures
        example_dir = os.path.join(output_dir, "examples", "padis_fsr", f"sparsity_{scanned_pixel_percent}", gt_name)
        os.makedirs(example_dir, exist_ok=True)

        for frame_idx in range(T):
            save_error_map(
                gt_video[frame_idx], rec_video[frame_idx],
                savePlot=True,
                savePath=os.path.join(example_dir, f"frame_{frame_idx:03d}_abs_error_map.tiff")
            )

            save_pixel_wise_psnr_plots(
                gt_video[frame_idx], rec_video[frame_idx],
                savePlot=True,
                savePath=os.path.join(example_dir, f"frame_{frame_idx:03d}_pixelwise_psnr.tiff")
            )

        # Collect results
        for frame_idx in range(T):
            local_results.append({
                "sampler": "padis_fsr",
                "withTemporalSampler": False,
                "withTemporalReconstruction": False,
                "gt_name": gt_name,
                "scanned_pixel_percent": scanned_pixel_percent,
                "frame_idx": frame_idx,
                "PSNR": psnrs[frame_idx],
                "SSIM": ssims[frame_idx],
                "alpha": None
            })

        log(f"[DONE] PADIS-FSR | {gt_name} | S={scanned_pixel_percent}%")

    except Exception as e:
        log(f"[ERROR] PADIS-FSR | {gt_name} | S={scanned_pixel_percent}% | {e}\n{traceback.format_exc()}")

    return local_results

# --------------------
# Main
# --------------------
def main():
    t_experiment_start = time.perf_counter()
    """
    if os.path.exists(LOGFILE):
        os.remove(LOGFILE)

    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)
    """

    low_dwell_queue = queue.Queue()

    padis_queue = queue.Queue()
    result_queue = queue.Queue()

    # Build sampler task list
    #experimental conditions: Main method: adaptive, with/without temporal sampler, with/without temporal reconstruction
    sampler_tasks = []
    
    for gt_name in GROUNDTRUTH_NAMES:
        for use_temporal_sampler in TEMPORAL_SAMPLING_OPTIONS:
            for use_temporal_reconstruction in TEMPORAL_RECONSTRUCTION_OPTIONS:
                for scanned_pixel_percent in SCANNED_PIXELS_PERCENTAGES:
                    if use_temporal_reconstruction:
                        for alpha in ALPHAS:
                            sampler_tasks.append((gt_name, scanned_pixel_percent, "adaptive", use_temporal_sampler, use_temporal_reconstruction, alpha))
                    else:
                        sampler_tasks.append((gt_name, scanned_pixel_percent, "adaptive", use_temporal_sampler, use_temporal_reconstruction, 1.0)) # alpha is irrelevant when temporal reconstruction is disabled
    
    
    # Add stratified sampler tasks (no temporal options, no alpha)
    for gt_name in GROUNDTRUTH_NAMES:
        for scanned_pixel_percent in SCANNED_PIXELS_PERCENTAGES:
            sampler_tasks.append((gt_name, scanned_pixel_percent, "stratified", False, False, None))
    
    # Add low dwell time sampler tasks as baseline
    
    for gt_name in GROUNDTRUTH_NAMES:
        for scanned_pixel_percent in SCANNED_PIXELS_PERCENTAGES:
            low_dwell_queue.put((gt_name,scanned_pixel_percent))
    """
    for gt_name in groundTruthNames:
        for scanned_pixel_percent in scannedPixelsPercent:
            padis_queue.put((gt_name, scanned_pixel_percent))
    """

    padis_workers = [PadisWorker(padis_queue, result_queue) for _ in range(PADIS_WORKER_POOL_SIZE)]
    output_worker = OutputWorker(result_queue)

    log("===== Starting Parallel Runs =====")
    for w in padis_workers:
        w.start()
    output_worker.start()

    # Sampler tasks run in separate processes (bypasses GIL for CPU-bound work).
    with ProcessPoolExecutor(max_workers=STANDARD_WORKER_POOL_SIZE) as executor:
        futures = {executor.submit(run_sampler, *task): task for task in sampler_tasks}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                if result:
                    result_queue.put(result)
                else:
                    log(f"[WORKER WARNING] No result for {task}")
            except Exception as e:
                log(f"[WORKER ERROR] {task} | {e}\n{traceback.format_exc()}")

    # Low-dwell tasks run in separate processes using the same completion handling.
    low_dwell_tasks = []
    while True:
        try:
            low_dwell_tasks.append(low_dwell_queue.get_nowait())
        except queue.Empty:
            break

    with ProcessPoolExecutor(max_workers=STANDARD_WORKER_POOL_SIZE) as executor:
        futures = {executor.submit(run_low_dwell_time_sampler, *task): task for task in low_dwell_tasks}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                if result:
                    result_queue.put(result)
                else:
                    log(f"[LOW DWELL WARNING] No result for {task}")
            except Exception as e:
                log(f"[LOW DWELL ERROR] {task} | {e}\n{traceback.format_exc()}")

    for w in padis_workers:
        w.join()
    result_queue.put(None)  # Signal output worker to stop
    output_worker.join()
    log("===== All Runs Completed =====")
    log(f"Saved per-frame results to {CSV_PATH}")
    t_experiment_end = time.perf_counter()
    log(f"[TIMING] Experiment total: {t_experiment_end - t_experiment_start:.2f}s using {STANDARD_WORKER_POOL_SIZE} sampler workers")

if __name__ == "__main__":
    main()