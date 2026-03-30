import os
import numpy as np
import pandas as pd
import concurrent.futures
import queue
import threading
import logging
import traceback
from datetime import datetime

from src.stadsadaptivesampler.src.stads.stads import AdaptiveSampler
from src.stadsadaptivesampler.src.stads.stratified_sampler import StratifiedSampler
from src.stadsadaptivesampler.src.stads.monitor import save_absolute_error_map, save_pixel_wise_psnr_plots
from src.stadsadaptivesampler.src.stads.config import HYDRATION_ONE, LI_EXPULSION_ONE, LI_EXPULSION_TWO, SI_LITHIATION_ONE

import padis_fsr
from padis_fsr import generate_mask_for_frame, run_padis_fsr_video_with_masks

logging.basicConfig(level=logging.INFO)

# --------------------
# CONFIG
# --------------------
GROUNDTRUTH_MAP = {
    "hydration_one": HYDRATION_ONE,
    "li_expulsion_one": LI_EXPULSION_ONE,
    "li_expulsion_two": LI_EXPULSION_TWO,
    "si_lithiation_one": SI_LITHIATION_ONE
}

groundTruthNames = list(GROUNDTRUTH_MAP.keys())
sparsityPercents = [10, 15, 25, 50]
temporalSamplerOptions = [True, False]
temporalReconstructionOptions = [True, False]

numberOfFrames = 20
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
LOGFILE = "script_log.txt"
STANDARD_WORKER_POOL_SIZE = 12
PADIS_WORKER_POOL_SIZE = 4


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
# Load video
# --------------------
def load_video(gt_name, num_frames):
    video = GROUNDTRUTH_MAP[gt_name][:num_frames]
    if video.ndim == 4 and video.shape[-1] == 1:
        video = video.squeeze(-1)
    return video


# --------------------
# STADS wrapper
# --------------------
def run_sampler(gt_name, sparsity, sampler_type, withTemporalSampler, withTemporalReconstruction):
    local_results = []

    log(f"Starting: {sampler_type} | {gt_name} | S={sparsity}% | Temporal={withTemporalSampler}")
    try:
        gt_video = load_video(gt_name,numberOfFrames)
        trueNumberOfFrames = min(gt_video.shape[0],numberOfFrames)

        if sampler_type == "adaptive":
            sampler = AdaptiveSampler(
                initialSampling="stratified",
                interpolMethod="linear",
                sparsityPercent=sparsity,
                numberOfFrames=trueNumberOfFrames,
                groundTruthName=gt_name,
                withTemporal=withTemporalSampler#,
                #withTemporalReconstruction=withTemporalReconstruction
            )
        else:
            sampler = StratifiedSampler(
                interpolMethod="linear",
                sparsityPercent=sparsity,
                numberOfFrames=trueNumberOfFrames,
                groundTruthName=gt_name,
            )

        rec_video, PSNRs, SSIMs = sampler.run()

        # Save figures
        example_dir = os.path.join(output_dir, "examples", sampler_type, f"sparsity_{sparsity}", gt_name)
        os.makedirs(example_dir, exist_ok=True)

        for frame_idx in range(trueNumberOfFrames):
            log(f"Skipping saving figures for frame {frame_idx}...")
            #sampler.save_figures(frameNumber=frame_idx, save_path=example_dir)

        # Collect results (LOCAL!)
        for frame_idx in range(trueNumberOfFrames):
            local_results.append({
                "sampler": sampler_type,
                "withTemporalSampler": withTemporalSampler if sampler_type == "adaptive" else False,
                "withTemporalReconstruction": withTemporalReconstruction if sampler_type == "adaptive" else False,
                "gt_name": gt_name,
                "sparsity": sparsity,
                "frame_idx": frame_idx,
                "PSNR": PSNRs[frame_idx],
                "SSIM": SSIMs[frame_idx]
            })

        log(f"[DONE] {sampler_type} | {gt_name} | S={sparsity}%")

    except Exception as e:
        log(f"[ERROR] {sampler_type} | {gt_name} | S={sparsity}% | {e}\n{traceback.format_exc()}")

    return local_results


class SamplerWorker(threading.Thread):
    def __init__(self, task_queue, result_queue, group = None, target = None, name = None, args = ..., kwargs = None, *, daemon = None):
        self.task_queue = task_queue
        self.result_queue = result_queue
        super().__init__(group, target, name, args, kwargs, daemon=daemon)

    def run(self):
        while True:
            try:
                gt_name, sparsity, sampler_type, withTemporalSampler, withTemporalReconstruction = self.task_queue.get(timeout=5)
                result = run_sampler(gt_name, sparsity, sampler_type, withTemporalSampler, withTemporalReconstruction)
                if result:
                    self.result_queue.put(result)
                else:
                    log(f"[WORKER WARNING] No result for {gt_name} | {sparsity}% | {sampler_type}")
                self.task_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                log(f"[WORKER ERROR] {e}\n{traceback.format_exc()}")
                self.task_queue.task_done()

class PadisWorker(threading.Thread):
    def __init__(self, padis_queue, result_queue, group = None, target = None, name = None, args = ..., kwargs = None, *, daemon = None):
        self.padis_queue = padis_queue
        self.result_queue = result_queue
        super().__init__(group, target, name, args, kwargs, daemon=daemon)

    def run(self):
        while True:
            try:
                gt_name, sparsity = self.padis_queue.get(timeout=5)
                result = run_padis(gt_name, sparsity)
                if result:
                    self.result_queue.put(result)
                else:
                    log(f"[PADIS WORKER WARNING] No result for {gt_name} | {sparsity}%")
                self.padis_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                log(f"[PADIS WORKER ERROR] {e}\n{traceback.format_exc()}")
                self.padis_queue.task_done()

class OutputWorker(threading.Thread):
    def __init__(self, result_queue, group = None, target = None, name = None, args = ..., kwargs = None, *, daemon = None):
        self.result_queue = result_queue
        self.fieldnames = ["sampler", "withTemporalSampler", "withTemporalReconstruction", "gt_name", "sparsity", "frame_idx", "PSNR", "SSIM"]
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
def run_padis(gt_name, sparsity):
    local_results = []

    log(f"Starting: PADIS-FSR | {gt_name} | S={sparsity}%")
    try:
        gt_video = load_video(gt_name, numberOfFrames)
        T, H, W = gt_video.shape

        masks = [generate_mask_for_frame(gt_video[t], sparsity).astype(np.uint8) for t in range(T)]
        rec_video, psnrs, ssims = run_padis_fsr_video_with_masks(gt_video, masks)

        # Save figures
        example_dir = os.path.join(output_dir, "examples", "padis_fsr", f"sparsity_{sparsity}", gt_name)
        os.makedirs(example_dir, exist_ok=True)

        for frame_idx in range(T):
            save_absolute_error_map(
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
                "sparsity": sparsity,
                "frame_idx": frame_idx,
                "PSNR": psnrs[frame_idx],
                "SSIM": ssims[frame_idx]
            })

        log(f"[DONE] PADIS-FSR | {gt_name} | S={sparsity}%")

    except Exception as e:
        log(f"[ERROR] PADIS-FSR | {gt_name} | S={sparsity}% | {e}\n{traceback.format_exc()}")

    return local_results


# --------------------
# Main
# --------------------
def main():
    if os.path.exists(LOGFILE):
        os.remove(LOGFILE)

    task_queue = queue.Queue()
    padis_queue = queue.Queue()
    result_queue = queue.Queue()

    #experimental conditions: Main method: adaptive, with/without temporal sampler, with/without temporal reconstruction
    for gt_name in groundTruthNames:
        for sparsity in sparsityPercents:
            for withTemporalSampler in temporalSamplerOptions:
                for withTemporalReconstruction in temporalReconstructionOptions:
                    task_queue.put((gt_name, sparsity, "adaptive", withTemporalSampler, withTemporalReconstruction))

    # Stratified baseline
    for gt_name in groundTruthNames:
        for sparsity in sparsityPercents:
            task_queue.put((gt_name, sparsity, "stratified", None, None))
    
    # PADIS-FSR 
    """
    for gt_name in groundTruthNames:
        for sparsity in sparsityPercents:
            padis_queue.put((gt_name, sparsity))
    """
    

    standard_workers = [SamplerWorker(task_queue, result_queue) for _ in range(STANDARD_WORKER_POOL_SIZE)]
    padis_workers = [PadisWorker(padis_queue, result_queue) for _ in range(PADIS_WORKER_POOL_SIZE)]
    task_workers = standard_workers + padis_workers
    output_worker = OutputWorker(result_queue)

    log("===== Starting Parallel Runs =====")
    for w in task_workers:
        w.start()
    output_worker.start()

    for w in task_workers:
        w.join()
    result_queue.put(None)  # Signal output worker to stop
    output_worker.join()
    log("===== All Runs Completed =====")
    csv_path = os.path.join(output_dir, "per_frame_results.csv")
    log(f"Saved per-frame results to {csv_path}")

if __name__ == "__main__":
    main()