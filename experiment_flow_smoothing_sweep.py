"""Sweeps compute_pdf_from_optical_flow's downscale/sigma parameters
(stads_adaptive_sampler's src/stads/pdfsampling/optical_flow.py) end to end.

This deliberately does NOT compare PDFs against each other -- that
comparison lives in stads_adaptive_sampler's
tests/pdfsampling/test_pdf_from_optical_flow.py as a numeric-equivalence
check against the pre-decimation algorithm. This experiment instead looks at
what each (downscale, sigma) setting does downstream: the actual sampling
pattern and reconstructed frames the sampler produces, since that is what the
algorithm is actually evaluated on. Per-frame PSNR/SSIM go to
flow_sweep_results.csv; the "reconstruction" and "samples" debug images (off:
"pdf", since that's the thing this experiment is explicitly not judging by)
are written per combination for visual inspection.

Runs the same single dataset/sparsity/alpha/adaptive-fraction configuration
experiments_main.py is currently pinned to -- this sweeps only the two new
parameters, not re-sweeping every other axis on top of them.
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
# sweeps only DOWNSCALE_FACTORS/SIGMAS on top of it, not every other axis.
GT_NAME = GROUNDTRUTH_NAMES[0]
SCANNED_PIXEL_PERCENT = 0.5
INTERPOL_METHOD = "cubic"
HAS_TEMPORAL_SAMPLER = True
HAS_TEMPORAL_RECONSTRUCTION = True
ALPHA = 2.0
ADAPTIVE_FRACTION = 0.1

# downscale=1 is compute_pdf_from_optical_flow's own "no decimation"; 2/4/8
# group 2x2/4x4/8x8 pixels respectively before warping and blurring the PDF.
DOWNSCALE_FACTORS = [1, 2, 4, 8]

# sigma=0 skips the Gaussian entirely. Applied at whichever resolution the
# paired downscale produces, not auto-rescaled -- see optical_flow.py's
# PDF_SMOOTHING_SIGMA docstring for why 1.25 is the library default at
# downscale=4, not 5.
SIGMAS = [0, 1.25, 1.5, 2, 4, 8]

limit_number_of_frames_to = 500
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
LOGFILE = "flow_sweep_log.txt"
CSV_PATH = os.path.join(output_dir, "flow_sweep_results.csv")
STANDARD_WORKER_POOL_SIZE = 6

# reconstruction + samples only: this experiment judges the downstream effect
# on the sampling pattern and reconstruction, not the pdf itself -- see the
# module docstring.
DEBUG_IMAGES_DICT = debug_images_dict({"reconstruction", "samples"})

RUN_CONFIG = RunConfig(
    output_dir=output_dir,
    limit_number_of_frames_to=limit_number_of_frames_to,
    debug_images_dict=DEBUG_IMAGES_DICT,
    log_path=LOGFILE,
    line_profile_enabled=LINE_PROFILE_ENABLED,
)

CSV_FIELDNAMES = BASE_CSV_FIELDNAMES + ["flowPdfDownscale", "flowPdfSigma"]


def main():
    t_start = time.perf_counter()

    if os.path.exists(LOGFILE):
        os.remove(LOGFILE)
    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)

    tasks = [(downscale, sigma) for downscale in DOWNSCALE_FACTORS for sigma in SIGMAS]
    log(LOGFILE, f"===== Starting flow-smoothing sweep: {len(tasks)} runs =====")

    with ProcessPoolExecutor(max_workers=STANDARD_WORKER_POOL_SIZE) as executor:
        futures = {}
        for downscale, sigma in tasks:
            future = executor.submit(
                run_sampler, RUN_CONFIG, GT_NAME, SCANNED_PIXEL_PERCENT, "adaptive",
                INTERPOL_METHOD, HAS_TEMPORAL_SAMPLER, HAS_TEMPORAL_RECONSTRUCTION,
                ALPHA, ADAPTIVE_FRACTION,
                {"flowPdfDownscale": downscale, "flowPdfSigma": sigma},
                (f"downscale_{downscale}", f"sigma_{sigma}"),
            )
            futures[future] = (downscale, sigma)

        for future in concurrent.futures.as_completed(futures):
            downscale, sigma = futures[future]
            try:
                result = future.result()
                if result:
                    for row in result:
                        row["flowPdfDownscale"] = downscale
                        row["flowPdfSigma"] = sigma
                    write_results(result, CSV_PATH, CSV_FIELDNAMES, LOGFILE)
                else:
                    log(LOGFILE, f"[WORKER WARNING] No result for downscale={downscale} sigma={sigma}")
            except Exception as e:
                log(LOGFILE,
                    f"[WORKER ERROR] downscale={downscale} sigma={sigma} | "
                    f"{e}\n{traceback.format_exc()}")

    log(LOGFILE, "===== Flow-smoothing sweep completed =====")
    log(LOGFILE, f"Saved per-frame results to {CSV_PATH}")
    t_end = time.perf_counter()
    log(LOGFILE,
        f"[TIMING] Sweep total: {t_end - t_start:.2f}s using "
        f"{STANDARD_WORKER_POOL_SIZE} workers, {len(tasks)} runs")


if __name__ == "__main__":
    main()
