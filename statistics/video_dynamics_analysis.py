#!/usr/bin/env python3
"""
Video Dynamics Analysis

Analyzes ground truth videos to quantify temporal activity and spatial
concentration of changes. Provides a 2D metric (temporal_activity, spatial_concentration)
for classifying videos based on their dynamics.

Metric Interpretation:
- temporal_activity: Average real pixel change per frame (above the noise
  floor), normalized by the video's own contrast (std of pixel intensities)
  - High: frame-to-frame changes are large relative to the video's contrast
  - Low: minimal changes relative to the video's contrast
  
- spatial_concentration (0-1): How localized changes are, measured as the
  normalized entropy of the change distribution
  - 1: Changes concentrated in small regions (e.g., bubble bursting)
  - 0: Changes distributed globally (e.g., camera pan, gradual fading)
  
  Shot noise is removed with a robust per-diff-map noise estimate (MAD
  around the median, which is invariant to spatially uniform intensity
  shifts from auto contrast); only changes above the noise cut count.

Relative metrics: in addition to the absolute values, results from
analyze_all_ground_truths carry min-max normalized versions over the
analyzed set (relative_activity, relative_concentration), placing each
video within the result space of all videos (0 = lowest, 1 = highest).

Example:
    High activity + High concentration: Localized rapid events (bubble burst)
    High activity + Low concentration: Global rapid changes (turbulent flow)
    Low activity + High concentration: Localized slow changes (slow-moving object)
    Low activity + Low concentration: Global slow changes (fading, drift)

Usage:
    python video_dynamics_analysis.py
    
    # Or programmatically:
    from video_dynamics_analysis import analyze_video_dynamics, analyze_all_ground_truths
    activity, concentration = analyze_video_dynamics(video_frames)
    results = analyze_all_ground_truths()
"""

import os
import sys
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add parent directory to path for importing experiment_common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from experiment_common import GROUNDTRUTH_MAP, _ground_truth_path
    HAS_EXPERIMENT_COMMON = True
except ImportError:
    GROUNDTRUTH_MAP: Dict[str, Tuple[str, int]] = {}
    _ground_truth_path = None
    HAS_EXPERIMENT_COMMON = False
    print("Warning: experiment_common not available, using fallback GT paths")

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    tifffile = None
    HAS_TIFFFILE = False
    print("Warning: tifffile not available, some functions disabled")


# MAD of |N(0, sigma)| around its median, in units of sigma. The
# frame-to-frame diff of unchanged pixels is half-normal, so the diff
# map's MAD (divided by this constant) estimates the noise std.
_HALF_NORMAL_MAD = 0.3989


def load_video_frames(video_path: str, frame_limit: Optional[int] = None) -> np.ndarray:
    """
    Load video frames from a TIFF file.
    
    Args:
        video_path: Path to the TIFF file
        frame_limit: Maximum number of frames to load (None for all)
        
    Returns:
        numpy array with shape (T, H, W) containing grayscale frames,
        preserving the file's dtype
    """
    if tifffile is None:
        raise ImportError("tifffile is required to load videos")
    
    with tifffile.TiffFile(video_path) as tif:
        frames = tif.asarray()
    
    # Handle different TIFF formats. Channels are detected from the last
    # axis (RGB/RGBA have 3 or 4 samples); the frame count never enters
    # the decision, so long grayscale stacks are not mistaken for
    # single color frames.
    if frames.ndim == 2:
        # Single grayscale frame
        frames = frames[np.newaxis, ...]
    elif frames.ndim == 3:
        if frames.shape[-1] in (3, 4):
            # Single RGB(A) frame -> grayscale
            frames = frames.mean(axis=-1, dtype=np.float64)[np.newaxis, ...]
        # else: (T, H, W) grayscale stack, use as is
    elif frames.ndim == 4:
        # (T, H, W, C)
        if frames.shape[-1] in (3, 4):
            # RGB(A) video -> grayscale
            frames = frames.mean(axis=-1, dtype=np.float64)
        else:
            frames = frames[..., 0]
    else:
        raise ValueError(f"Unsupported TIFF array shape {frames.shape}")
    
    # Apply frame limit
    if frame_limit is not None and frames.shape[0] > frame_limit:
        frames = frames[:frame_limit]
    
    return frames


def analyze_video_dynamics(
    video_frames: np.ndarray,
    noise_factor: float = 3.0
) -> Tuple[float, float]:
    """
    Analyze video dynamics to extract temporal activity and spatial concentration.
    
    Args:
        video_frames: Grayscale video as numpy array with shape (T, H, W)
        noise_factor: Multiples of the estimated frame-to-frame noise std
                      that a pixel change must exceed to count as change
                      (default: 3.0, ~the 99.7% mass of the noise). Lower
                      keeps weaker changes, higher strips heavier noise.
                      None disables noise correction entirely: every
                      frame-to-frame difference then contributes to both
                      metrics unchanged.
        
    Returns:
        Tuple of (temporal_activity, spatial_concentration)
        - temporal_activity: average per-frame change above the noise cut,
          divided by the video's own contrast (std of pixel intensities).
          ~1.0 means real frame-to-frame changes are comparable in size to
          the video's overall contrast.
        - spatial_concentration: 1 - H/log(N) of the residual change
          distribution (normalized entropy of the per-pixel change shares).
          0 = perfectly global change, ~1 = change concentrated on few
          pixels. Diff maps whose residual is indistinguishable from the
          noise tail score 0.0.
        
    Noise model: the frame-to-frame diff of unchanged pixels is half-normal
    (|n_t - n_{t-1}| for i.i.d. shot noise). Its std is estimated per diff
    map from the MAD around the median. The MAD is invariant to a spatially
    uniform offset, so residual auto-contrast intensity shifts do not
    inflate the noise estimate, and it is robust to localized change, which
    touches a minority of pixels. A spatially uniform shift that survives
    the intensity normalization is absorbed by the cut together with the
    noise; a genuine uniform change is indistinguishable from such a shift
    in a diff map, so both are ignored. For integer-valued frames whose
    shot noise was smoothed below the quantization step by temporal
    interpolation, the MAD estimate collapses to 0; a 1-step diff is then
    treated as quantization dither rather than change. Diff maps where
    real change covers close to or more than half the pixels invalidate
    the MAD estimate.
    """
    if video_frames.ndim != 3:
        raise ValueError(f"Expected 3D array (T, H, W), got shape {video_frames.shape}")
    
    T = video_frames.shape[0]
    
    if T < 2:
        return 0.0, 0.0
    
    # Video contrast: intensity spread of the video itself
    contrast = float(np.std(video_frames, dtype=np.float64))
    
    mean_diff_sum = 0.0
    concentrations = []
    
    # Process frame-to-frame differences one at a time so long videos
    # do not require all diff maps in memory at once
    integer_data = np.issubdtype(video_frames.dtype, np.integer)
    for t in range(1, T):
        diff = np.abs(video_frames[t].astype(np.float64) - video_frames[t-1].astype(np.float64))
        
        if noise_factor is None:
            # No noise correction: every difference counts as change
            cut = 0.0
            residual = diff
        else:
            # Noise estimate for this diff map. Subtracting k * sigma
            # removes the shot-noise floor; pixels must change by more
            # than the noise to contribute to either metric.
            med = float(np.median(diff))
            mad = float(np.median(np.abs(diff - med)))
            if mad > 0.0:
                cut = noise_factor * mad / _HALF_NORMAL_MAD
            elif integer_data:
                # Degenerate noise estimate: temporal interpolation can
                # smooth the shot noise below the quantization step,
                # leaving a diff map of exact zeros plus 1-step rounding
                # dither. The MAD then collapses to 0 and nothing would
                # be cut. For integer data a 1-step diff is the minimum
                # representable change and is treated as dither, not
                # change.
                cut = 1.0
            else:
                cut = 0.0
            
            residual = diff - cut
            residual[residual < 0.0] = 0.0
        
        mean_diff_sum += float(np.mean(residual))
        
        # Spatial concentration: normalized entropy of the residual change
        # distribution. p_i is pixel i's share of the real change; entropy
        # measures the effective number of pixels the change lives on.
        flat = residual.ravel()
        total = float(np.sum(flat))
        
        # Residuals at the sparse noise tail beyond the cut carry almost
        # no mass relative to the noise scale; treat such maps as
        # unchanged. With no measurable noise (sigma 0), only require
        # that some residual exists.
        if total <= 0.0 or (cut > 0.0 and float(np.mean(flat)) < 0.01 * cut):
            concentrations.append(0.0)
            continue
        
        p = flat / total
        nz = p[p > 0.0]
        entropy = -float(np.sum(nz * np.log(nz)))
        # 1 = perfectly localized (change on a single pixel), 0 = perfectly global
        concentrations.append(1.0 - entropy / np.log(flat.size))
    
    temporal_activity = mean_diff_sum / (T - 1)
    if contrast > 0.0:
        temporal_activity /= contrast
    else:
        temporal_activity = 0.0
    
    spatial_concentration = float(np.mean(concentrations))
    
    return temporal_activity, spatial_concentration


def analyze_all_ground_truths(
    noise_factor: Optional[float] = 3.0,
    frame_limit: Optional[int] = None
) -> List[Dict]:
    """
    Analyze all ground truth videos from GROUNDTRUTH_MAP.
    
    Args:
        noise_factor: Multiples of estimated noise std for the change cut
                      (default: 3.0); None disables noise correction
        frame_limit: Maximum frames to analyze per video (default: None = all)
        
    Returns:
        List of dicts, each containing:
        - gt_name: Ground truth name
        - temporal_activity: float
        - spatial_concentration: float
        - relative_activity: temporal_activity min-max normalized over all results
        - relative_concentration: spatial_concentration min-max normalized over all results
        - num_frames: Number of frames analyzed
        - total_frames: Total frames in video
    """
    if not HAS_EXPERIMENT_COMMON or _ground_truth_path is None:
        raise ImportError("experiment_common is required for GT analysis")
    
    results = []
    
    for gt_name, (filename, _) in GROUNDTRUTH_MAP.items():
        gt_path = _ground_truth_path(gt_name)
        
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth file not found: {gt_path}")
            continue
        
        try:
            video = load_video_frames(gt_path, frame_limit=frame_limit)
            total_frames = video.shape[0]
            
            if total_frames < 2:
                print(f"Warning: {gt_name} has only {total_frames} frame(s), skipping")
                continue
            
            activity, concentration = analyze_video_dynamics(video, noise_factor)
            
            results.append({
                'gt_name': gt_name,
                'temporal_activity': activity,
                'spatial_concentration': concentration,
                'num_frames': video.shape[0],
                'total_frames': total_frames,
                'height': video.shape[1],
                'width': video.shape[2]
            })
            
            print(f"{gt_name}: activity={activity:.4f}, concentration={concentration:.4f}, "
                  f"frames={video.shape[0]}, size={video.shape[1]}x{video.shape[2]}")
            
        except Exception as e:
            print(f"Error analyzing {gt_name}: {e}")
    
    compute_relative_metrics(results)
    return results


def compute_relative_metrics(results: List[Dict]) -> List[Dict]:
    """
    Add min-max normalized metrics to each result, placing every video
    within the result space spanned by all videos: 0 = lowest, 1 = highest
    in the set. Sets without spread map to 0.0.
    
    Adds 'relative_activity' and 'relative_concentration' to each dict
    and returns the same list. Idempotent.
    """
    for key, rel_key in (('temporal_activity', 'relative_activity'),
                         ('spatial_concentration', 'relative_concentration')):
        vals = [r[key] for r in results]
        lo, hi = min(vals), max(vals)
        for r in results:
            r[rel_key] = (r[key] - lo) / (hi - lo) if hi > lo else 0.0
    return results


def _bbox_overlap_area(b1, b2) -> float:
    """Overlapping area of two matplotlib Bbox objects (0 if disjoint)."""
    dx = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
    dy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return dx * dy


def _bbox_overhang(bbox, frame) -> float:
    """Total overhang (px) of bbox outside frame, summed over all sides."""
    return (max(0.0, frame.x0 - bbox.x0) + max(0.0, bbox.x1 - frame.x1)
            + max(0.0, frame.y0 - bbox.y0) + max(0.0, bbox.y1 - frame.y1))


# Candidate label offsets (points) relative to the marker, from close to far
_LABEL_OFFSETS = [
    (0, 16), (16, 0), (0, -16), (-16, 0),
    (14, 14), (-14, 14), (14, -14), (-14, -14),
    (0, 30), (30, 0), (0, -30), (-30, 0),
    (26, 26), (-26, 26), (26, -26), (-26, -26),
    (0, 44), (44, 0), (0, -44), (-44, 0),
    (38, 38), (-38, 38), (38, -38), (-38, -38),
    (0, 60), (60, 0), (0, -60), (-60, 0),
    (50, 50), (-50, 50), (50, -50), (-50, -50),
]


def _split_label(name: str, max_len: int = 16) -> str:
    """Split a long name into two lines at the space closest to the middle."""
    if len(name) <= max_len or ' ' not in name:
        return name
    mid = len(name) // 2
    spaces = [i for i, c in enumerate(name) if c == ' ']
    cut = min(spaces, key=lambda i: abs(i - mid))
    return name[:cut] + '\n' + name[cut + 1:]


def _place_scatter_labels(ax, fig, xs, ys, labels, marker_sizes, obstacles):
    """
    Place labels around scatter points, greedily choosing offsets that keep
    them clear of markers, previously placed labels, and the given obstacle
    texts. A thin leader line connects each label to its point.

    Candidate positions are measured with a throwaway annotation without
    leader line (Annotation.get_window_extent would otherwise union the
    text with the leader line), and the real label is only created at the
    winning offset.
    """
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0

    frame = ax.get_window_extent(renderer)
    obstacle_boxes = [t.get_window_extent(renderer) for t in obstacles]

    marker_boxes = []
    for x, y, s in zip(xs, ys, marker_sizes):
        cx, cy = ax.transData.transform((x, y))
        r = np.sqrt(s) * px_per_pt / 2.0 + 1.0
        marker_boxes.append(Bbox.from_bounds(cx - r, cy - r, 2 * r, 2 * r))

    def conflict(bbox, other, weight):
        area = _bbox_overlap_area(bbox, other)
        if area > 0.0:
            return 1e6 * weight + 10.0 * weight * area
        return 0.0

    label_bbox = dict(facecolor='white', alpha=0.85,
                      edgecolor='gray', boxstyle='round,pad=0.25')

    placed = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        best_score = None
        best_offset = None
        best_bbox = None
        for dx, dy in _LABEL_OFFSETS:
            probe = ax.annotate(
                labels[i],
                xy=(x, y), xycoords='data',
                xytext=(dx, dy), textcoords='offset points',
                fontsize=8.5, ha='center', va='center', zorder=4,
                bbox=label_bbox,
            )
            bbox = probe.get_window_extent(renderer)
            probe.remove()
            # Mildly prefer labels close to their point
            score = 0.01 * (dx * dx + dy * dy)
            # Discourage leaving the axes, but overlap is far worse
            score += 1e3 * _bbox_overhang(bbox, frame)
            score += conflict(bbox, marker_boxes[i], 0.5)
            for j, mb in enumerate(marker_boxes):
                if j != i:
                    score += conflict(bbox, mb, 1.0)
            for ob in obstacle_boxes:
                score += conflict(bbox, ob, 1.0)
            for pb in placed:
                score += conflict(bbox, pb, 1.0)
            if best_score is None or score < best_score:
                best_score = score
                best_offset = (dx, dy)
                best_bbox = bbox
        ax.annotate(
            labels[i],
            xy=(x, y), xycoords='data',
            xytext=best_offset, textcoords='offset points',
            fontsize=8.5, ha='center', va='center', zorder=4,
            bbox=label_bbox,
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.7,
                            shrinkA=1, shrinkB=4),
        )
        placed.append(best_bbox)


def plot_dynamics_scatter(
    results: List[Dict],
    output_path: Optional[str] = None,
    show: bool = True,
    relative: bool = False,
    labels: Optional[Dict[str, str]] = None
):
    """
    Plot ground truths on a 2D scatter of temporal activity vs spatial concentration.
    
    Args:
        results: List of result dicts from analyze_all_ground_truths
        output_path: Path to save the plot (optional)
        show: Whether to display the plot
        relative: Plot the min-max normalized metrics (position within the
                 result space of all videos) instead of the absolute ones
        labels: Optional mapping from ground truth names to display names
                 used verbatim in the plot; names without an entry fall
                 back to the in-code name with underscores replaced
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, cannot plot")
        return
    
    if not results:
        print("No results to plot")
        return
    
    # Extract data
    names = [r['gt_name'] for r in results]
    frame_counts = [r['num_frames'] for r in results]
    if relative:
        if 'relative_activity' not in results[0]:
            compute_relative_metrics(results)
        activities = [r['relative_activity'] for r in results]
        concentrations = [r['relative_concentration'] for r in results]
    else:
        activities = [r['temporal_activity'] for r in results]
        concentrations = [r['spatial_concentration'] for r in results]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot scatter (fully opaque markers, sized by frame count)
    sizes = [fc * 2 for fc in frame_counts]
    ax.scatter(
        activities, concentrations,
        s=sizes,
        alpha=1.0,
        edgecolors='black',
        linewidths=0.8,
        zorder=3
    )
    
    # Labels and title
    if relative:
        ax.set_xlabel('Relative Temporal Activity (0=lowest, 1=highest in GT set)', fontsize=12)
        ax.set_ylabel('Relative Spatial Concentration (0=lowest, 1=highest in GT set)', fontsize=12)
        ax.set_title('Ground Truth Video Dynamics Analysis (relative)', fontsize=14)
    else:
        ax.set_xlabel('Temporal Activity (change relative to video contrast)', fontsize=12)
        ax.set_ylabel('Spatial Concentration (0=Global, 1=Localized)', fontsize=12)
        ax.set_title('Ground Truth Video Dynamics Analysis', fontsize=14)
    ax.grid(True, alpha=0.3)
    if relative:
        ax.set_xlim((-0.05, 1.05))
    else:
        ax.set_xlim((0.0, max(activities) * 1.1))
    ax.set_ylim((-0.05, 1.05))
    
    # Quadrant lines and corner descriptions, only in relative space where
    # the center of the result space separates high/low activity and
    # localized/global change
    quadrant_texts = []
    if relative:
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        for x_frac, y_frac, ha, va, text in [
            (0.52, 0.97, 'left', 'top', 'Localized\nHigh Activity'),
            (0.48, 0.97, 'right', 'top', 'Localized\nLow Activity'),
            (0.52, 0.03, 'left', 'bottom', 'Global\nHigh Activity'),
            (0.48, 0.03, 'right', 'bottom', 'Global\nLow Activity'),
        ]:
            quadrant_texts.append(ax.text(
                x_frac, y_frac, text, transform=ax.transAxes,
                ha=ha, va=va, fontsize=10, fontstyle='italic', zorder=2
            ))
    
    if labels:
        point_labels = [labels.get(name, _split_label(name.replace('_', ' ')))
                        for name in names]
    else:
        point_labels = [_split_label(name.replace('_', ' ')) for name in names]
    
    plt.tight_layout()
    
    # Place point labels after the layout is final so candidate positions
    # are evaluated against the true figure geometry
    _place_scatter_labels(
        ax, fig,
        activities, concentrations,
        point_labels,
        sizes,
        quadrant_texts,
    )
    
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        # Clear a pre-existing target: stale Windows file states
        # (delete-pending, held handles) make rewrites fail with
        # cryptic Errno 22 instead of overwriting cleanly
        try:
            os.remove(output_path)
        except OSError:
            pass
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to: {output_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)
    return fig, ax


def print_results_table(results: List[Dict]):
    """Print results as a formatted table, including relative metrics."""
    if not results:
        print("No results to display")
        return
    
    if 'relative_activity' not in results[0]:
        compute_relative_metrics(results)
    
    # Sort by temporal activity
    results = sorted(results, key=lambda x: x['temporal_activity'], reverse=True)
    
    header = (f"{'Ground Truth':<30} {'Activity':>9} {'Conc':>8} "
              f"{'RelAct':>8} {'RelConc':>8} {'Frames':>7}")
    rule = "=" * len(header)
    print("\n" + rule)
    print(header)
    print(rule)
    for r in results:
        print(f"{r['gt_name']:<30} {r['temporal_activity']:>9.4f} {r['spatial_concentration']:>8.4f} "
              f"{r['relative_activity']:>8.4f} {r['relative_concentration']:>8.4f} {r['num_frames']:>7}")
    print(rule)


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze video dynamics for ground truth videos'
    )
    parser.add_argument(
        '--noise-factor', type=float, default=3.0,
        help='Multiples of estimated noise std for the change cut (default: 3.0)'
    )
    parser.add_argument(
        '--frame-limit', type=int, default=None,
        help='Maximum frames to analyze per video (default: all)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output path for scatter plot (default: show only)'
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Skip plotting'
    )
    parser.add_argument(
        '--csv', type=str, default=None,
        help='Save results to CSV file'
    )
    parser.add_argument(
        '--relative', action='store_true',
        help='Plot relative metrics (min-max over the GT set) instead of absolute'
    )
    parser.add_argument(
        '--no-noise-correction', action='store_true',
        help='Disable noise correction entirely (every difference counts as change)'
    )
    
    args = parser.parse_args()
    
    # Analyze all ground truths
    results = analyze_all_ground_truths(
        noise_factor=(None if args.no_noise_correction else args.noise_factor),
        frame_limit=args.frame_limit
    )
    
    if not results:
        print("No valid ground truths to analyze")
        return
    
    # Print table
    print_results_table(results)
    
    # Save to CSV if requested
    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['gt_name', 'temporal_activity',
                                                     'spatial_concentration',
                                                     'relative_activity', 'relative_concentration',
                                                     'num_frames', 'total_frames', 'height', 'width'])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"Results saved to: {args.csv}")
    
    # Plot if requested
    if not args.no_plot:
        try:
            plot_dynamics_scatter(results, output_path=args.output,
                                  show=(args.output is None), relative=args.relative)
        except ImportError:
            print("matplotlib not available, skipping plot")


if __name__ == '__main__':
    main()
