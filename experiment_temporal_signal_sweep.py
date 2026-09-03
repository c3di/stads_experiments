"""Sweeps AdaptiveSampler's two interchangeable temporal PDF signals --
temporalMethod="optical_flow" (stads_adaptive_sampler's
src/stads/pdfsampling/optical_flow.py) and temporalMethod="temporal_variance"
(.../pdfsampling/temporal_variance.py) -- each over its own
downscale x sigma grid.

This does not *rely on* comparing PDFs against each other to draw its main
conclusion -- that numeric equivalence lives in stads_adaptive_sampler's
tests/pdfsampling/test_pdf_from_optical_flow.py and
test_pdf_from_temporal_variance.py, checked against each method's own
pre-decimation algorithm. This experiment instead looks primarily at what
each (method, downscale, sigma) combination does downstream: the actual
sampling pattern and reconstructed frames the sampler produces, since that
is what the algorithm is actually evaluated on. Per-frame PSNR/SSIM go to
temporal_signal_sweep_results.csv; "reconstruction", "samples",
"temporal_variance", "flow" and "pdf" (both the blended pdf and, if enabled
below, its unblended pdf_spatial/pdf_temporal contributions) are written per
combination for visual inspection.

Runs the same single dataset/sparsity/alpha/adaptive-fraction configuration
experiments_main.py is currently pinned to -- this sweeps only the temporal-
signal parameters, not re-sweeping every other axis on top of them.
"""
import concurrent.futures
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import logging

from experiment_common import (
    GROUNDTRUTH_NAMES, debug_images_dict, RunConfig, run_sampler,
    BASE_CSV_FIELDNAMES, write_results, log, LINE_PROFILE_ENABLED,
)

logging.basicConfig(level=logging.INFO)

# --------------------
# CONFIG
# --------------------
# The same single task experiments_main.py currently runs -- this experiment
# sweeps only TEMPORAL_METHODS/DOWNSCALE_FACTORS/SIGMAS on top of it, not
# every other axis.
GT_NAME = GROUNDTRUTH_NAMES[0]
SCANNED_PIXEL_PERCENT = 1.0
INTERPOL_METHOD = "cubic"
HAS_TEMPORAL_SAMPLER = True
HAS_TEMPORAL_RECONSTRUCTION = True
ALPHA = 2.0
ADAPTIVE_FRACTION = 0.25
MIN_DENSITY_GAMMA = 0.25

TEMPORAL_METHODS = ["optical_flow", "temporal_variance"]

# downscale=1 is both methods' own "no decimation"; 2/4/8 group 2x2/4x4/8x8
# pixels respectively before blurring the temporal signal into a PDF.
DOWNSCALE_FACTORS = [1, 2, 4]

# sigma=0 skips the Gaussian entirely. Applied at whichever resolution the
# paired downscale produces, not auto-rescaled -- see
# stads.pdfsampling.pdf_temporal_shaping's PDF_TEMPORAL_DOWNSCALE/_SIGMA,
# the shared library defaults both methods now use.
SIGMAS = [0, 2, 4, 8]

limit_number_of_frames_to = 500
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
LOGFILE = "temporal_signal_sweep_log.txt"
CSV_PATH = os.path.join(output_dir, "temporal_signal_sweep_results.csv")
STANDARD_WORKER_POOL_SIZE = 6

# reconstruction + samples + temporal_variance + flow + pdf (+ its
# pdf_spatial/pdf_temporal unblended contributions): lets this experiment
# judge the downstream effect on the sampling pattern and reconstruction,
# while also being able to inspect the temporal signal and the pdf it drove
# directly. temporal_variance/flow/pdf_temporal stay harmless to enable
# regardless of temporalMethod: whichever one didn't run this frame simply
# never writes a page for it (see PdfTemporalContributionDebugImage).
#
# To enable a further debug output, add its kind name to this set --
# debug_images_dict() turns every other known kind explicitly off (see
# experiment_common.ALL_DEBUG_IMAGE_KINDS for the complete list: currently
# "reconstruction", "samples", "pdf", "pdf_spatial", "pdf_temporal", "flow",
# "temporal_variance", "error", "psnr", "ssim", "triangulation"). Example,
# uncommented, adds the per-frame pixel-wise PSNR heatmap on top of the ones
# already enabled below:
#
#   DEBUG_IMAGES_DICT = debug_images_dict(
#       {"reconstruction", "samples", "temporal_variance", "flow", "pdf",
#        "pdf_spatial", "pdf_temporal", "psnr"})
DEBUG_IMAGES_DICT = debug_images_dict(
    {"reconstruction", "samples"})

RUN_CONFIG = RunConfig(
    output_dir=output_dir,
    limit_number_of_frames_to=limit_number_of_frames_to,
    debug_images_dict=DEBUG_IMAGES_DICT,
    log_path=LOGFILE,
    line_profile_enabled=LINE_PROFILE_ENABLED,
)

CSV_FIELDNAMES = BASE_CSV_FIELDNAMES + ["temporalMethod", "downscale", "sigma"]


def _extra_sampler_kwargs(method):
    """temporalMethod is the only knob run_sampler doesn't already expose by
    name -- downscale/sigma are the same pdf_temporal_downscale/
    pdf_temporal_sigma parameters regardless of which method is active."""
    return {} if method == "optical_flow" else {"temporalMethod": method}


def main():
    t_start = time.perf_counter()

    if os.path.exists(LOGFILE):
        os.remove(LOGFILE)
    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)

    tasks = [(method, downscale, sigma)
            for method in TEMPORAL_METHODS
            for downscale in DOWNSCALE_FACTORS
            for sigma in SIGMAS]
    log(LOGFILE, f"===== Starting temporal-signal sweep: {len(tasks)} runs =====")

    with ProcessPoolExecutor(max_workers=STANDARD_WORKER_POOL_SIZE) as executor:
        futures = {}
        for method, downscale, sigma in tasks:
            future = executor.submit(
                run_sampler, RUN_CONFIG, GT_NAME, SCANNED_PIXEL_PERCENT, "adaptive",
                INTERPOL_METHOD, HAS_TEMPORAL_SAMPLER, HAS_TEMPORAL_RECONSTRUCTION,
                ALPHA, ADAPTIVE_FRACTION, MIN_DENSITY_GAMMA,
                pdf_temporal_downscale=downscale, pdf_temporal_sigma=sigma,
                extra_sampler_kwargs=_extra_sampler_kwargs(method),
                extra_path_parts=(f"method_{method}", f"downscale_{downscale}", f"sigma_{sigma}"),
            )
            futures[future] = (method, downscale, sigma)

        for future in concurrent.futures.as_completed(futures):
            method, downscale, sigma = futures[future]
            try:
                result = future.result()
                if result:
                    for row in result:
                        row["temporalMethod"] = method
                        row["downscale"] = downscale
                        row["sigma"] = sigma
                    write_results(result, CSV_PATH, CSV_FIELDNAMES, LOGFILE)
                else:
                    log(LOGFILE,
                        f"[WORKER WARNING] No result for method={method} "
                        f"downscale={downscale} sigma={sigma}")
            except Exception as e:
                log(LOGFILE,
                    f"[WORKER ERROR] method={method} downscale={downscale} sigma={sigma} | "
                    f"{e}\n{traceback.format_exc()}")

    log(LOGFILE, "===== Temporal-signal sweep completed =====")
    log(LOGFILE, f"Saved per-frame results to {CSV_PATH}")
    t_end = time.perf_counter()
    log(LOGFILE,
        f"[TIMING] Sweep total: {t_end - t_start:.2f}s using "
        f"{STANDARD_WORKER_POOL_SIZE} workers, {len(tasks)} runs")


if __name__ == "__main__":
    main()
