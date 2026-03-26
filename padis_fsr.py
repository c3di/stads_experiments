#!/usr/bin/env python3
# padis_fsr.py
#
# Clean version with:
# - generate_mask_for_frame()
# - run_padis_fsr_frame_with_mask()
# - run_padis_fsr_video_with_masks()

import os
import tempfile
import logging
import numpy as np
from scipy.io import savemat, loadmat
import subprocess
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Mask generator (added because your script requires it)
# ------------------------------------------------------------
def generate_mask_for_frame(frame: np.ndarray, sparsity_percent: int):
    """
    Simple random sparse mask generator.
    frame: 2D uint8
    sparsity_percent: 10..100
    returns mask (0/1) uint8
    """
    H, W = frame.shape
    total = H * W
    keep = int(total * (sparsity_percent / 100.0))

    mask = np.zeros(total, dtype=np.uint8)
    mask[:keep] = 1
    np.random.shuffle(mask)
    return mask.reshape(H, W)


# ------------------------------------------------------------
# Octave interface
# ------------------------------------------------------------
def _run_octave_save_rec(mat_in: str, mat_out: str):
    cmd = (
        f"octave --quiet --eval "
        f"\"addpath('{os.getcwd()}'); "
        f"data=load('{mat_in}'); "
        f"rec=fsr_reconstruct_image(data.img, data.mask, data.quality); "
        f"save('-v7','{mat_out}','rec'); "
        f"exit;\""
    )
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _load_mat_robust(path: str) -> np.ndarray:
    try:
        data = loadmat(path)
        if "rec" in data:
            return np.array(data["rec"])
        for k, v in data.items():
            if not k.startswith("__"):
                return np.array(v)
        raise RuntimeError("No valid array in MAT file.")
    except Exception:
        import h5py
        with h5py.File(path, "r") as f:
            if "rec" in f:
                return np.array(f["rec"])
            keys = list(f.keys())
            return np.array(f[keys[0]])


# ------------------------------------------------------------
# Core reconstruction
# ------------------------------------------------------------
def _fsr_reconstruct(frame: np.ndarray, mask: np.ndarray, quality: str):
    if frame.ndim != 2:
        raise ValueError("frame must be 2D")
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")

    H, W = frame.shape
    if mask.shape != (H, W):
        raise ValueError(f"Mask shape mismatch: {mask.shape} vs {frame.shape}")

    with tempfile.TemporaryDirectory() as td:
        mat_in = os.path.join(td, "in.mat")
        mat_out = os.path.join(td, "out.mat")

        savemat(mat_in, {
            "img": frame.astype(np.float64),
            "mask": mask.astype(np.float64),
            "quality": str(quality),
        }, format="5")

        _run_octave_save_rec(mat_in, mat_out)

        rec = _load_mat_robust(mat_out)

    rec = np.array(rec, dtype=np.float32)
    if rec.max() <= 1.001:
        rec *= 255.0

    return np.clip(rec, 0, 255).astype(np.uint8)


# ------------------------------------------------------------
# Public APIs
# ------------------------------------------------------------
def run_padis_fsr_frame_with_mask(frame: np.ndarray,
                                  mask: np.ndarray,
                                  quality: str = "COMPROMISE"):
    try:
        rec = _fsr_reconstruct(frame, mask, quality)
        psnr = compute_psnr(frame, rec, data_range=255)
        ssim = compute_ssim(frame, rec, data_range=255)
        return rec, float(psnr), float(ssim)
    except Exception as e:
        logger.exception(f"Frame reconstruction failed: {e}")
        H, W = frame.shape
        return np.zeros((H, W), dtype=np.uint8), np.nan, np.nan


def run_padis_fsr_video_with_masks(video: np.ndarray,
                                   masks: list,
                                   quality: str = "COMPROMISE"):
    T, H, W = video.shape
    rec_video = np.zeros((T, H, W), dtype=np.uint8)
    psnrs = np.zeros(T, dtype=float)
    ssims = np.zeros(T, dtype=float)

    for i in range(T):
        frame = video[i]
        mask = masks[i]
        rec, p, s = run_padis_fsr_frame_with_mask(frame, mask, quality)
        rec_video[i] = rec
        psnrs[i] = p
        ssims[i] = s

    return rec_video, psnrs, ssims
