#!/usr/bin/env python3
# padis_helpers.py
# Robust, pure-Python version of PADIS helper routines (based on original).
# - Includes missing params fields (sig_min_curr, sig_max_factor_curr)
# - Handles zero-size / division-by-zero edge cases
# - Uses numpy; intentionally avoids numba to prevent TBB/fork issues

from __future__ import annotations
import numpy as np
import math
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class params:
    """
    Parameter container used by adaptive_sampling_mask_simplified5.
    Fields added to match original expectations:
      - sig_min_curr (sigma min)
      - sig_max_factor_curr (sigma max scaling)
    """
    def __init__(self,
                 max_add: float,
                 ld_dens: float,
                 des_dens: float,
                 var_border: int,
                 tau: int = 7,
                 w_max: int = 32,
                 k_A: int = 2,
                 sig_min_curr: float = 0.2,
                 sig_max_factor_curr: float = 0.85):
        # original required fields
        self.max_add = max_add
        self.ld_dens = ld_dens
        self.des_dens = des_dens
        self.var_border = var_border
        self.tau = tau
        self.w_max = w_max
        self.k_A = k_A
        # new/expected fields
        self.sig_min_curr = sig_min_curr
        self.sig_max_factor_curr = sig_max_factor_curr

# ---------------------
# Helper functions
# ---------------------
def get_var_map_weighted(mask: np.ndarray, img: np.ndarray, border: int) -> np.ndarray:
    """
    Compute a local, weighted variance map:
    For each pixel, within a (2*border+1)^2 neighborhood, compute a weighted variance
    of sampled pixels (mask==1), using Gaussian-like weighting centered at the pixel.
    This is a direct (but vector-limited) translation of the original algorithm.
    """
    sy, sx = mask.shape
    out = np.zeros_like(mask, dtype=float)

    if border <= 0:
        # simplest case: just compute local variance over entire mask neighborhood=1 => fallback
        ys, xs = np.where(mask == 1)
        if ys.size == 0:
            return out
        overall_var = np.var(img[ys, xs])
        out.fill(overall_var)
        return out

    # iterate — this is somewhat heavy but explicit and robust
    for i_y in range(sy):
        by0 = min(border, i_y)
        y0 = i_y - by0
        y1 = i_y + border + 1
        for i_x in range(sx):
            bx0 = min(border, i_x)
            x0 = i_x - bx0
            x1 = i_x + border + 1

            m_arr = mask[y0:y1, x0:x1]
            arr = img[y0:y1, x0:x1]

            Y, X = np.nonzero(m_arr == 1)
            N = Y.size
            if N > 1:
                vals = arr[Y, X].astype(np.float64)
                # compute weights: center index corresponds to 'border' in original code
                # When truncated at edges, still use border as reference (same as original)
                if border == 0:
                    weights = np.ones(N, dtype=np.float64)
                else:
                    # weights: exp(-((x-border)**2+(y-border)**2)/(border**2))
                    dx = (X.astype(np.float64) - border)
                    dy = (Y.astype(np.float64) - border)
                    denom = (border ** 2) if (border != 0) else 1.0
                    weights = np.exp(-((dx * dx + dy * dy) / denom))
                    # normalise to max 1 to mimic original behaviour
                    wmax = weights.max()
                    if wmax > 0:
                        weights = weights / wmax
                mu = np.mean(vals)
                out[i_y, i_x] = np.sum(weights * (vals - mu) ** 2) / N
    return out


def get_gauss(std: float, tau: int = 7) -> np.ndarray:
    """
    Returns a modified 2D Gaussian-like kernel with parameter 'std' and 'tau'.
    Mirrors original algorithm that grows the kernel until boundary condition.
    """
    if std <= 0:
        return np.array([[1.0]], dtype=float)
    gauss = np.zeros((1, 1), dtype=float)
    border = 0
    tmp = (1 - math.exp(-((border + 1) ** 2) / (std ** 2))) ** tau
    while tmp != 1.0:
        border += 1
        arr = np.zeros((border + 1, border + 1), dtype=float)
        arr[0:border, 0:border] = gauss
        arr[0, border] = tmp
        arr[border, 0] = tmp
        d1 = border
        for d2 in range(1, border + 1):
            val = (1 - math.exp(-((d1 ** 2 + d2 ** 2) / (std ** 2)))) ** tau
            arr[d1, d2] = val
            arr[d2, d1] = val
        gauss = arr.copy()
        tmp = (1 - math.exp(-((border + 1) ** 2) / (std ** 2))) ** tau

    # mirror and concatenate
    gauss_flip = gauss[:, ::-1]
    gauss = np.concatenate((gauss_flip[:, :-1], gauss), axis=1)
    gauss_flip = gauss[::-1, :]
    gauss = np.concatenate((gauss_flip[:-1, :], gauss), axis=0)
    return gauss


def draw_from_pdf(pdf: np.ndarray, n: int) -> np.ndarray:
    """
    Draw 'n' unique samples (y,x) from pdf. If pdf sums to zero, sample uniformly among unmasked positions.
    """
    flat_pdf = pdf.astype(float).ravel()
    s = flat_pdf.sum()
    pdf_shape = pdf.shape
    size = flat_pdf.size
    if s <= 0 or np.isnan(s):
        # fallback - draw uniformly / without replacement from all coordinates
        idx = np.random.choice(np.arange(size), size=n, replace=False)
    else:
        idx = np.random.choice(np.arange(size), size=n, replace=False, p=(flat_pdf / s))
    draw_y, draw_x = np.unravel_index(idx, pdf_shape)
    return np.vstack((draw_y, draw_x)).T


def get_stdmap_for_v5(vmap: np.ndarray, lim: float, stdmin: float, stdmax: float) -> np.ndarray:
    """
    Compute stdmap from variance map robustly.
    Handles empty-positive case gracefully.
    """
    vm = vmap.ravel()
    positive = vm > 0
    if not np.any(positive):
        # fallback: constant map with minimal std
        logger.warning("get_stdmap_for_v5: no positive variance values, returning constant stdmap")
        return np.full(vmap.shape, stdmin, dtype=float)
    vmin = np.min(vm[positive])
    vm[vm == 0] = vmin
    log_vmap = -np.log10(vm)
    bins = max(2, int(vmap.size / 100))
    v, b = np.histogram(log_vmap, bins=bins)
    cs = np.cumsum(v).astype(float)
    if cs[-1] == 0:
        logger.warning("get_stdmap_for_v5: histogram cumulative sum is zero; returning constant stdmap")
        return np.full(vmap.shape, stdmin, dtype=float)
    cs = cs / cs[-1]
    ind = int(np.argmax(cs >= lim))
    # guard index
    if ind >= len(b) - 1:
        val = b[-1]
    else:
        val = (b[ind] + b[ind + 1]) / 2.0
    mu = val
    # clip extremely large values
    log_vmap[log_vmap >= val] = mu
    if val == 0:
        logger.warning("get_stdmap_for_v5: val==0 encountered; using small epsilon to avoid division by zero")
        val = 1e-12
    lvm = log_vmap / val
    stdmap = stdmin + (stdmax - stdmin) * lvm
    stdmap = stdmap.reshape(vmap.shape)
    return stdmap


def find_gstds_simplified(density: float, tau, max_radius, k, gstd_min: float = 0.2, gstd_max_scaling: float = 0.85) -> Tuple[float, float]:
    """
    Compute bounds for Gaussian sigma based on density.
    """
    if density <= 0:
        density = 1e-12
    gstd_upper_bound = 1200
    gstd_max = gstd_max_scaling / math.sqrt(density) - gstd_min
    if gstd_max > gstd_upper_bound:
        gstd_max = gstd_upper_bound
    if gstd_max <= gstd_min:
        gstd_max = gstd_min + 0.1
    return gstd_min, gstd_max


def get_decision_map(mask: np.ndarray, stdmap: np.ndarray, tau: int) -> np.ndarray:
    """
    Compute decision (pdf) map by placing local gaussians centered at existing sampled locations.
    """
    sy, sx = mask.shape
    pdf = np.ones_like(mask, dtype=float)
    centerY, centerX = np.where(mask == 1)
    for i in range(centerY.size):
        cy = int(centerY[i]); cx = int(centerX[i])
        std = float(stdmap[cy, cx])
        gauss = get_gauss(std, tau)
        border = (gauss.shape[0] - 1) // 2
        y0 = cy - border
        x0 = cx - border
        y1 = cy + border + 1
        x1 = cx + border + 1
        # intersection with pdf bounds
        yy0 = max(y0, 0); xx0 = max(x0, 0)
        yy1 = min(y1, sy); xx1 = min(x1, sx)
        gauss_y0 = yy0 - y0; gauss_x0 = xx0 - x0
        gauss_y1 = gauss.shape[0] - (y1 - yy1)
        gauss_x1 = gauss.shape[1] - (x1 - xx1)
        try:
            pdf[yy0:yy1, xx0:xx1] *= gauss[gauss_y0:gauss_y1, gauss_x0:gauss_x1]
        except Exception as e:
            # fallback: skip this center if shapes mismatch
            logger.debug(f"get_decision_map: skipping a center due to shape mismatch: {e}")
            continue
    # ensure non-negative
    pdf = np.maximum(pdf, 0.0)
    return pdf


def adaptive_sampling_mask_simplified5(img: np.ndarray, sample_params: params, mask_initial: np.ndarray) -> np.ndarray:
    """
    High-level adaptive sampling: given image and initial mask, add samples until desired density reached.
    Returns final mask (dtype uint8, values 0/1).
    """
    H, W = img.shape[:2]
    max_add_count = int(H * W * sample_params.max_add)
    if max_add_count < 1:
        max_add_count = 1

    mask = (mask_initial > 0).astype(np.uint8)

    desired_total = int(round(sample_params.des_dens * H * W))
    num_samples = int(np.count_nonzero(mask))
    sample_counter = desired_total - num_samples
    if sample_counter <= 0:
        # already satisfied
        return mask.astype(np.uint8)

    # timing/debug
    logger.debug(f"adaptive_sampling: target {desired_total} samples, have {num_samples}, add {sample_counter}")

    while sample_counter > 0:
        # compute variance map using currently sampled pixels
        img_masked = img * mask
        vmap = get_var_map_weighted(mask, img_masked, sample_params.var_border)
        vmax = vmap.max() if vmap.size > 0 else 0.0
        if vmax == 0 or np.isnan(vmax):
            # fallback: make vmap uniform to avoid zeros
            vmap = np.ones_like(vmap, dtype=float) * 1e-12
            vmax = vmap.max()
        vmap = vmap / vmax

        # find gstds
        density_now = max(1e-12, num_samples / float(vmap.size))
        gstd_min, gstd_max = find_gstds_simplified(density_now, None, None, None,
                                                   gstd_min=sample_params.sig_min_curr,
                                                   gstd_max_scaling=sample_params.sig_max_factor_curr)
        gstd_max = gstd_min + gstd_max

        stdmap = get_stdmap_for_v5(vmap, 0.98, gstd_min, gstd_max)
        dmap = get_decision_map(mask, stdmap, sample_params.tau)

        # clip dmap to positions not already sampled
        dmap = dmap * (1 - mask)

        # determine how many to add this iteration
        add_samples = min(sample_counter, max_add_count)

        # if dmap sum is zero (rare), pick random un-sampled positions
        if np.sum(dmap) <= 0:
            # choose uniformly from zeros
            zeros = np.argwhere(mask == 0)
            if zeros.shape[0] == 0:
                break
            if add_samples >= zeros.shape[0]:
                picks = zeros
            else:
                idx = np.random.choice(np.arange(zeros.shape[0]), size=add_samples, replace=False)
                picks = zeros[idx]
            for (yy, xx) in picks:
                mask[int(yy), int(xx)] = 1
            num_samples += picks.shape[0]
            sample_counter -= picks.shape[0]
            continue

        # draw from pdf
        sample_list = draw_from_pdf(dmap, add_samples)
        for yx in sample_list:
            mask[int(yx[0]), int(yx[1])] = 1

        num_samples += sample_list.shape[0]
        sample_counter -= sample_list.shape[0]

    return mask.astype(np.uint8)
