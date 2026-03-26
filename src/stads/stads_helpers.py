import cv2
import numpy as np
from scipy.ndimage import uniform_filter

from joblib import Parallel, delayed
from numba import jit, prange

from .image_processing import compute_image_of_gradient_magnitudes
from .utility_functions import percentile_norm


@jit(nopython=True, parallel=True)
def local_variance_numba(image, windowSize):
    h, w = image.shape
    pad = windowSize // 2
    meanMap = np.zeros((h, w), dtype=np.float32)
    meanSquareMap = np.zeros((h, w), dtype=np.float32)

    for y in prange(h):
        for x in prange(w):
            mean = 0.0
            meanSq = 0.0
            count = 0

            for dy in range(-pad, pad + 1):
                for dx in range(-pad, pad + 1):
                    yy = y + dy
                    xx = x + dx

                    if yy < 0:
                        yy = -yy
                    elif yy >= h:
                        yy = 2 * h - yy - 2

                    if xx < 0:
                        xx = -xx
                    elif xx >= w:
                        xx = 2 * w - xx - 2

                    val = image[yy, xx]
                    mean += val
                    meanSq += val * val
                    count += 1

            mean /= count
            meanSq /= count
            meanMap[y, x] = mean
            meanSquareMap[y, x] = meanSq

    varianceMap = meanSquareMap - meanMap ** 2
    return meanMap, varianceMap


def compute_local_moments_of_image(image, windowSize):

    gradientMap = compute_image_of_gradient_magnitudes(image)
    meanMap, varianceMap = local_variance_numba(image.astype(np.float32), windowSize)

    varianceMap = percentile_norm(varianceMap, smooth_sigma=5)
    gradientMap = percentile_norm(gradientMap, smooth_sigma=5)

    return meanMap, gradientMap, varianceMap


def compute_local_moments_batch(frames, windowSize, n_jobs=-1):
    return Parallel(n_jobs=n_jobs)(
        delayed(compute_local_moments_of_image)(img, windowSize)
        for img in frames
    )


def compute_local_temporal_variance(previousFrame, currentFrame, windowSize):
    stack = np.stack([previousFrame, currentFrame], axis=0)
    smoothed = np.array([uniform_filter(f, size=windowSize) for f in stack])
    mean = np.mean(smoothed, axis=0)
    varianceMap = np.mean((smoothed - mean) ** 2, axis=0)
    return varianceMap


def compute_temporal_variance_batch(frame_pairs, windowSize, n_jobs=-1):
    return Parallel(n_jobs=n_jobs)(
        delayed(compute_local_temporal_variance)(pair[0], pair[1], windowSize)
        for pair in frame_pairs
    )


def compute_optical_flow(previousFrame: np.ndarray, currentFrame: np.ndarray, windowSize: int):
    pyr_scale = 0.5
    levels = 5
    iterations = 5
    poly_n = 7
    poly_sigma = 1.0
    flags = 0

    if previousFrame is None or currentFrame is None:
        raise ValueError("Could not read input images.")

    flow = cv2.calcOpticalFlowFarneback(previousFrame, currentFrame, None,
                                        pyr_scale, levels, windowSize,
                                        iterations, poly_n, poly_sigma, flags)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    if not np.any(np.isfinite(magnitude)) or np.all(magnitude == 0):
        return np.zeros_like(magnitude, dtype=np.float32)

    upper = np.nanmax(magnitude)
    if not np.isfinite(upper) or upper <= 0:
        upper = 1e-6
    magnitude = np.clip(magnitude, 0, upper)
    magnitude /= float(upper)
    magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)

    return magnitude.astype(np.float32)


def compute_optical_flow_batch(frame_pairs, windowSize, n_jobs=-1):
    return Parallel(n_jobs=n_jobs)(
        delayed(compute_optical_flow)(pair[0], pair[1], windowSize)
        for pair in frame_pairs
    )


def compute_pdf_from_gradients_image(gradientsImage):
    gradientsImage = np.asarray(gradientsImage, dtype=np.float32)
    gradientsImage = np.clip(gradientsImage, 0, None)

    total = np.sum(gradientsImage)
    if total <= 1e-12:
        return np.zeros_like(gradientsImage, dtype=np.float32)

    pdf = gradientsImage / total
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    pdf = np.maximum(pdf, 0)

    pdf /= np.sum(pdf)
    return pdf
