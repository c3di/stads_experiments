"""Shared infrastructure between experiments_main.py and
experiment_flow_smoothing_sweep.py -- ground-truth resolution, the
AdaptiveSampler/StratifiedSampler run-and-record wrapper, CSV writing, and
optional line-by-line profiling. Kept here rather than duplicated so the two
scripts' sampler-construction logic can't drift apart.
"""
import os
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from stads.stads import AdaptiveSampler
from stads.stratified_sampler import StratifiedSampler
from stads.video_downloader import DEFAULT_SAVE_DIR

from stads.debug_images.reconstruction import ReconstructionDebugImage
from stads.debug_images.samples import SamplesDebugImage
from stads.debug_images.pdf import PdfDebugImage
from stads.debug_images.flow import FlowDebugImage
from stads.debug_images.error import ErrorMapDebugImage
from stads.debug_images.psnr import PsnrMapDebugImage
from stads.debug_images.ssim import SsimMapDebugImage
from stads.debug_images.triangulation import TriangulationDebugImage

# display name -> (filename under DEFAULT_SAVE_DIR, total dwell time)
GROUNDTRUTH_MAP = {
    #"HYDRATION_ONE": ("Hydration.tif", 25000),
    "LI_EXPULSION_ONE": ("Li_Expulsion_1.tif", 20000),
    # "LI_EXPULSION_TWO": ("Li_Expulsion_2.tif", 20000),
    #"SI_LITHIATION_ONE": ("Si_Lithiation.tif", 20000),
    #"EDS_AEROSPACE_ONE": ("EDS_aerospace_one.tif", 20000),
    #"EDS_AEROSPACE_TWO":   ("EDS_aerospace_two.tif", 20000),
    #"TITANIUM_STRAIN_ONE": ("Titanium_strain.tif", 20000)
}

GROUNDTRUTH_NAMES = list(GROUNDTRUTH_MAP.keys())


def _ground_truth_path(gt_name):
    filename, _ = GROUNDTRUTH_MAP[gt_name]
    if not os.path.splitext(filename)[1]:
        filename = filename + ".tif"
    return str(DEFAULT_SAVE_DIR / filename)


# --------------------
# Logging
# --------------------
def log(logfile_path, msg: str):
    """Prints and appends `msg` (timestamped) to logfile_path.

    Takes the path explicitly rather than closing over it, so this stays a
    plain picklable function -- ProcessPoolExecutor workers need to be able
    to import and call it by reference.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"{now} | {msg}"
    print(full_msg, flush=True)
    try:
        with open(logfile_path, "a") as f:
            f.write(full_msg + "\n")
    except Exception:
        pass


# --------------------
# Debug images
# --------------------
ALL_DEBUG_IMAGE_KINDS = [
    ReconstructionDebugImage.kind, SamplesDebugImage.kind, PdfDebugImage.kind,
    FlowDebugImage.kind, ErrorMapDebugImage.kind, PsnrMapDebugImage.kind,
    SsimMapDebugImage.kind, TriangulationDebugImage.kind,
]


def debug_images_dict(enabled_kinds):
    """{kind: bool} for every top-level DebugImage kind (not the _zoom
    companions, which stay at their own class default) -- True for
    `enabled_kinds`, explicitly False for the rest, so the result is a
    complete spec rather than relying on each class's own default_enabled."""
    enabled_kinds = set(enabled_kinds)
    return {kind: (kind in enabled_kinds) for kind in ALL_DEBUG_IMAGE_KINDS}


# --------------------
# Line-by-line profiling (opt-in)
# --------------------
LINE_PROFILE_ENABLED = os.environ.get("STADS_LINE_PROFILE") == "1"


def make_line_profiler():
    """A LineProfiler instrumented for the pdf/overlay_masks phases (see
    stads.py's [PHASE-TOTAL] log) -- the functions profiled during the
    optical-flow performance investigation."""
    from line_profiler import LineProfiler
    from stads.pdfsampling import blend as pdf_blend
    from stads.pdfsampling.pdf_from_signal import compute_pdf_from_gradients_image, percentile_norm
    from stads.pdfsampling.optical_flow import compute_pdf_from_optical_flow
    from stads.debug_images.base import DebugImage
    from stads.debug_images.samples import _build_sampling_mask
    from stads.debug_images.rendering import overlay_sampling_masks, overlay_colour_map, to_uint8_rgb

    profiler = LineProfiler()
    for fn in (
        pdf_blend.spatiotemporal,
        pdf_blend.spatial_only,
        compute_pdf_from_gradients_image,
        percentile_norm,
        compute_pdf_from_optical_flow,
        DebugImage.process,
        DebugImage.write,
        PdfDebugImage._process,
        SamplesDebugImage._process,
        _build_sampling_mask,
        overlay_sampling_masks,
        overlay_colour_map,
        to_uint8_rgb,
    ):
        profiler.add_function(fn)
    return profiler


# --------------------
# Sampler run wrapper
# --------------------
@dataclass
class RunConfig:
    """Everything run_sampler needs beyond one task's own parameters -- both
    experiments_main.py and experiment_flow_smoothing_sweep.py build one of
    these once and pass it to every task."""
    output_dir: str
    limit_number_of_frames_to: Optional[int]
    debug_images_dict: Optional[dict]
    log_path: str
    line_profile_enabled: bool = False


def run_sampler(config: RunConfig, gt_name, scanned_pixel_percent, sampler_type,
                interpol_method="linear", has_temporal_sampler=True,
                has_temporal_reconstruction=True, alpha=None, adaptive_fraction=0.0,
                extra_sampler_kwargs=None, extra_path_parts=()):
    """Build, run and record one AdaptiveSampler/StratifiedSampler task.

    extra_sampler_kwargs is forwarded to AdaptiveSampler's constructor only
    (StratifiedSampler tasks ignore it) -- lets a caller like
    experiment_flow_smoothing_sweep.py vary a knob this function doesn't know
    about. extra_path_parts is appended to the run's output directory, for
    the same reason.
    """
    local_results = []
    t_overall_start = time.perf_counter()

    log(config.log_path,
        f"Starting: {sampler_type} | interpol={interpol_method} | {gt_name} | "
        f"S={scanned_pixel_percent}% | SamplerTemporal={has_temporal_sampler}| "
        f"ReconstructionTemporal={has_temporal_reconstruction} | alpha={alpha} | "
        f"adaptive={adaptive_fraction}")
    try:
        ground_truth_path = _ground_truth_path(gt_name)
        if sampler_type == "adaptive":
            sampler = AdaptiveSampler(
                initialSampling="stratified",
                boundaryPlacement="border",
                interpolMethod=interpol_method,
                sparsityPercent=scanned_pixel_percent,
                limit_number_of_frames_to=config.limit_number_of_frames_to,
                groundTruthPath=ground_truth_path,
                alpha=alpha,
                withTemporalSampling=has_temporal_sampler,
                withTemporalReconstruction=has_temporal_reconstruction,
                adaptiveRefinementFraction=adaptive_fraction,
                debugImages=config.debug_images_dict,
                **(extra_sampler_kwargs or {}),
            )
        else:
            sampler = StratifiedSampler(
                interpolMethod=interpol_method,
                sparsityPercent=scanned_pixel_percent,
                limit_number_of_frames_to=config.limit_number_of_frames_to,
                groundTruthPath=ground_truth_path,
            )

        trueNumberOfFrames = sampler.numberOfFrames

        # interpol_method is part of the path: without it a cubic run would
        # overwrite the linear run's figures for the same configuration.
        # adaptive_fraction likewise, or a refined run would overwrite the
        # unrefined one it is meant to be compared against. extra_path_parts
        # does the same job for whatever else a caller is sweeping.
        example_dir = os.path.join(
            config.output_dir, "examples", sampler_type, f"interpol_{interpol_method}",
            f"sparsity_{scanned_pixel_percent}", gt_name,
            f"sampler_{has_temporal_sampler}_reconstruction_{has_temporal_reconstruction}",
            f"alpha_{alpha}", f"adaptive_{adaptive_fraction}", *extra_path_parts)
        os.makedirs(example_dir, exist_ok=True)

        t_run_start = time.perf_counter()
        # Both sampler families write each frame's enabled debug images as
        # they're produced (see stads' debug_images/) rather than in a
        # separate pass afterward, so this includes that I/O.
        if config.line_profile_enabled:
            profiler = make_line_profiler()
            rec_video, PSNRs, SSIMs = profiler.runcall(sampler.run, save_path=example_dir)
            profile_path = os.path.join(example_dir, "line_profile.txt")
            with open(profile_path, "w") as f:
                profiler.print_stats(stream=f)
            log(config.log_path, f"[PROFILE] wrote line-profiler report to {profile_path}")
        else:
            rec_video, PSNRs, SSIMs = sampler.run(save_path=example_dir)
        t_run_end = time.perf_counter()
        log(config.log_path,
            f"[TIMING] sampler.run(): {t_run_end - t_run_start:.2f}s | {sampler_type} | "
            f"{gt_name} | S={scanned_pixel_percent}% | alpha={alpha}")

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
                "adaptiveFraction": adaptive_fraction if sampler_type == "adaptive" else None,
            })

        log(config.log_path,
            f"[DONE] {sampler_type} | interpol={interpol_method} | {gt_name} | S={scanned_pixel_percent}%")

    except Exception as e:
        log(config.log_path,
            f"[ERROR] {sampler_type} | interpol={interpol_method} | {gt_name} | "
            f"S={scanned_pixel_percent}% | {e}\n{traceback.format_exc()}")

    t_overall_end = time.perf_counter()
    log(config.log_path,
        f"[TIMING] run_sampler() total: {t_overall_end - t_overall_start:.2f}s | "
        f"{sampler_type} | interpol={interpol_method} | {gt_name} | "
        f"S={scanned_pixel_percent}% | alpha={alpha}")
    return local_results


# --------------------
# CSV output
# --------------------
#: Every key run_sampler's result dicts always carry. A caller with extra
#: swept parameters (e.g. flow_smoothing_sweep's downscale/sigma) appends its
#: own columns to a copy of this list rather than to this one.
BASE_CSV_FIELDNAMES = ["sampler", "withTemporalSampler", "withTemporalReconstruction", "gt_name",
                       "scanned_pixel_percent", "frame_idx", "PSNR", "SSIM", "alpha", "beta",
                       "adaptiveFraction"]


def write_results(results, csv_path, fieldnames, log_path):
    """Append a batch of per-frame result dicts to csv_path, called directly
    from the main thread as each ProcessPoolExecutor future completes."""
    if not results:
        return
    df = pd.DataFrame(results)
    if not os.path.exists(csv_path):
        log(log_path, f"[OUTPUT] Creating new CSV file: {csv_path}")
        df.to_csv(csv_path, index=False, columns=fieldnames)
    else:
        df.to_csv(csv_path, index=False, mode='a', header=False, columns=fieldnames)
