import os
import numpy as np
import pandas as pd
import concurrent.futures
import logging
import traceback
from datetime import datetime

from src.stadsadaptivesampler.src.stads import AdaptiveSampler
from src.stads_adaptive_sampler.src.stads.stratified_sampler import StratifiedSampler
from src.stads_adaptive_sampler.src.stads.monitor import save_absolute_error_map, save_pixel_wise_psnr_plots
from src.stads_adaptive_sampler.src.stads.config import HYDRATION_ONE, LI_EXPULSION_ONE, LI_EXPULSION_TWO, SI_LITHIATION_ONE

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
numberOfFrames = 20
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
LOGFILE = "script_log.txt"


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
def run_sampler(gt_name, sparsity, sampler_type, withTemporal=True):
    local_results = []

    log(f"Starting: {sampler_type} | {gt_name} | S={sparsity}% | Temporal={withTemporal}")
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
                withTemporal=withTemporal,
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
            sampler.save_figures(frameNumber=frame_idx, save_path=example_dir)

        # Collect results (LOCAL!)
        for frame_idx in range(trueNumberOfFrames):
            local_results.append({
                "sampler": sampler_type,
                "withTemporal": withTemporal if sampler_type == "adaptive" else None,
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
                "withTemporal": None,
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

    log("===== Starting Parallel Runs =====")

    all_results = []

    with concurrent.futures.ProcessPoolExecutor() as pool:
        futures = []

        for gt_name in groundTruthNames:
            for sparsity in sparsityPercents:
                futures.append(pool.submit(run_sampler, gt_name, sparsity, "adaptive", True))
                futures.append(pool.submit(run_sampler, gt_name, sparsity, "adaptive", False))
                futures.append(pool.submit(run_sampler, gt_name, sparsity, "stratified", None))
                futures.append(pool.submit(run_padis, gt_name, sparsity))

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    all_results.extend(result)
            except Exception as e:
                log(f"[FUTURE ERROR] {e}")

    # Save CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, "per_frame_results.csv")
    df.to_csv(csv_path, index=False)

    log(f"Saved per-frame results to {csv_path}")


if __name__ == "__main__":
    main()