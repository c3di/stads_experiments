#!/usr/bin/env python3
"""
Frame Strips for Paper Script

Generates frame strip visualizations for paper figures comparing adaptive, 
stratified, and low-dwell reconstructions.

Layout: 3-row composite image
Row 1: Adaptive reconstruction frames (equidistantly sampled)
Row 2: Stratified reconstruction frames (same frame indices)
Row 3: Low-dwell frames (same frame indices)

Each row contains num_frames images arranged horizontally with spacing.
Frames can be cropped to a specified window region.

Usage (programmatic):
    from frame_strips_for_paper import create_frame_strips
    
    output_path = create_frame_strips(
        gt_name="LI_EXPULSION_ONE_10FPS",
        scanned_pixel_percent=0.1,
        num_frames=3,
        target_metric="psnr",
        window=None,  # or (center_x, center_y, width, height)
        analysis_file_path="statistics/parameter_tune_analysis.txt",
        base_dir="plots",
        output_dir="statistics/frame_strips"
    )
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from tifffile import tifffile

# Add parent directory to path for importing experiment_common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment_common import GROUNDTRUTH_MAP, _ground_truth_path


def resize_frame(frame: np.ndarray, target_H: int, target_W: int) -> np.ndarray:
    """Resize a frame to target dimensions using LANCZOS interpolation."""
    is_single_channel_3d = frame.ndim == 3 and frame.shape[2] == 1
    
    if frame.ndim == 2:
        img = Image.fromarray(frame, mode='L')
    elif frame.ndim == 3:
        if frame.shape[2] == 3:
            img = Image.fromarray(frame, mode='RGB')
        elif frame.shape[2] == 4:
            img = Image.fromarray(frame, mode='RGBA')
        else:
            img = Image.fromarray(frame[..., 0], mode='L')
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")
    
    img = img.resize((target_W, target_H), Image.LANCZOS)
    result = np.array(img, dtype=frame.dtype)
    
    if is_single_channel_3d:
        result = result[..., np.newaxis]
    
    return result


def build_stratified_path(gt_name: str, scanned_pixel_percent: float,
                          interpol_method: str = "cubic", base_dir: str = "plots") -> str:
    """Build the path to stratified reconstruction results."""
    path_parts = [
        base_dir, "examples", "stratified",
        f"interpol_{interpol_method}",
        f"sparsity_{scanned_pixel_percent}", gt_name,
        "sampler_False_reconstruction_False",
        "temporalMethod_optical_flow",
        "temporalResidualCutoff_12.0",
        "temporalResidualConfidenceScale_100.0",
        "sampleSequence_stratified",
        "alpha_None", "adaptive_0.0"
    ]
    return os.path.join(*path_parts)


def build_adaptive_path_from_config(config: Dict, base_dir: str = "plots") -> str:
    """Build the path to adaptive reconstruction results from a config dict."""
    adaptive_dir = os.path.join(
        base_dir, "examples", config["sampler_type"],
        f"interpol_{config['interpol_method']}",
        f"sparsity_{config['scanned_pixel_percent']}", config["gt_name"],
        f"sampler_{config['has_temporal_sampler']}_reconstruction_{config['has_temporal_reconstruction']}",
        f"temporalMethod_{config['temporal_method']}",
        f"temporalResidualCutoff_{config['temporal_residual_cutoff']}",
        f"temporalResidualConfidenceScale_{config['temporal_residual_confidence_scale']}",
        f"sampleSequence_{config['sample_sequence']}",
        f"alpha_{config['alpha']}", f"adaptive_{config['adaptive_fraction']}"
    )
    return adaptive_dir


def load_frames_from_path(exp_dir: str, result_type: str = "reconstruction") -> np.ndarray:
    """
    Load frames from a result directory.
    
    Args:
        exp_dir: Path to the experiment directory
        result_type: Type of result file (e.g., "reconstruction")
    
    Returns:
        Array of shape (T, H, W) as uint8
    """
    result_file = os.path.join(exp_dir, f"{result_type}.tiff")
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Result file not found: {result_file}")
    
    with tifffile.TiffFile(result_file) as tif:
        num_pages = len(tif.pages)
        frames = tif.asarray(key=None)
        
        if frames.ndim == 2:
            # Could be single frame or multi-page TIFF that asarray flattened
            # If there are multiple pages, we need to load each page individually
            if num_pages > 1:
                frames_list = [page.asarray() for page in tif.pages]
                frames = np.stack(frames_list, axis=0)
            else:
                # Single frame - add time dimension
                frames = frames[np.newaxis, ...]
        elif frames.ndim == 3:
            # Could be (T, H, W) or (H, W, C)
            if frames.shape[0] < 500:
                # This is (T, H, W)
                pass
            else:
                # This is (H, W, C) - single frame with channels
                frames = frames[np.newaxis, ...]
        elif frames.ndim == 4:
            # This is (T, H, W, C) - take first channel
            if frames.shape[-1] == 1:
                frames = frames[..., 0]
            else:
                frames = frames[..., 0]
    
    # Convert to uint8
    if frames.dtype != np.uint8:
        if np.issubdtype(frames.dtype, np.floating):
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)
    
    return frames


def load_all_low_dwell_frames(gt_name: str, sparsity: float, base_dir: str = "plots") -> np.ndarray:
    """
    Load all low-dwell frames for a given GT and sparsity.
    
    Args:
        gt_name: Ground truth name
        sparsity: Scanned pixel percentage
        base_dir: Base directory for results
    
    Returns:
        Array of shape (T, H, W) as uint8
    """
    low_dwell_dir = os.path.join(base_dir, "examples", "low_dwell", f"sparsity_{sparsity}", gt_name)
    if not os.path.exists(low_dwell_dir):
        raise FileNotFoundError(f"Low-dwell directory not found: {low_dwell_dir}")
    
    frame_files = sorted([f for f in os.listdir(low_dwell_dir)
                         if f.startswith("frame_") and f.endswith("_low_dwell.tiff")])
    if not frame_files:
        raise FileNotFoundError(f"No low-dwell frame files found in: {low_dwell_dir}")
    
    frames = []
    for frame_file in frame_files:
        frame = tifffile.imread(os.path.join(low_dwell_dir, frame_file))
        
        # Handle different formats
        if frame.ndim == 3:
            if frame.shape[0] >= 100:
                # This is (H, W, C) - single frame
                frame = frame[..., 0]
            else:
                # This is (T, H, W) - but we're loading individual files
                frame = frame[0] if frame.shape[0] == 1 else frame[..., 0]
        
        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        frames.append(frame)
    
    return np.stack(frames, axis=0)


def parse_best_config_by_metric(
    analysis_file_path: str,
    gt_name: str,
    scanned_pixel_percent: float,
    target_metric: str
) -> Dict:
    """
    Parse the analysis file to extract the best configuration for a specific metric.
    
    Args:
        analysis_file_path: Path to parameter_tune_analysis.txt
        gt_name: Ground truth name to find
        scanned_pixel_percent: Sparsity value to find
        target_metric: One of "psnr", "ssim", or "combined"
    
    Returns:
        Dictionary containing the configuration parameters
    
    Raises:
        ValueError: If the metric is not found
    """
    if target_metric not in ["psnr", "ssim", "combined"]:
        raise ValueError(f"target_metric must be 'psnr', 'ssim', or 'combined', got {target_metric}")
    
    metric_labels = {
        "psnr": "Best PSNR",
        "ssim": "Best SSIM",
        "combined": "Best Combined"
    }
    
    target_label = metric_labels[target_metric]
    
    with open(analysis_file_path, 'r') as f:
        lines = f.readlines()
    
    current_gt = None
    current_sparsity = None
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Check for GT section header (e.g., "--- HYDRATION_ONE ---")
        # GT headers start at beginning of line, sparsity headers have leading spaces
        if stripped.startswith('---') and stripped.endswith('---') and not line.startswith('  '):
            current_gt = stripped[3:-3].strip()
            i += 1
            continue
        
        # Check for sparsity section header (e.g., "  --- scanned_pixel_percent = 0.1 ---")
        if stripped.startswith('--- scanned_pixel_percent'):
            # Extract sparsity value
            inner = stripped[3:-3].strip()  # Remove outer ---
            parts = inner.split('=')
            if len(parts) >= 2:
                current_sparsity = float(parts[1].strip())
            i += 1
            continue
        
        # Check for target metric line
        if target_label in stripped:
            # Found the metric section, get the config from the next line
            j = i + 1
            config_line = None
            while j < len(lines):
                next_line = lines[j].strip()
                if next_line and next_line.startswith('withTemporalSampler'):
                    config_line = next_line
                    break
                j += 1
            
            if config_line and current_gt == gt_name and current_sparsity == scanned_pixel_percent:
                # Parse the configuration
                config = {}
                config['gt_name'] = current_gt
                config['scanned_pixel_percent'] = current_sparsity
                
                # Extract key-value pairs from the config line
                parts = [p.strip() for p in config_line.split(',')]
                for part in parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Handle boolean values
                        if value == 'True':
                            value = True
                        elif value == 'False':
                            value = False
                        # Handle numeric values
                        elif value.replace('.', '').replace('-', '').isdigit():
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        
                        # Map parameter names to our config keys
                        if key == 'gt_name':
                            # Already set
                            pass
                        elif key == 'scanned_pixel_percent':
                            # Already set
                            pass
                        elif key == 'adaptiveFraction':
                            config['adaptive_fraction'] = value
                        elif key == 'sampleSequence':
                            config['sample_sequence'] = value
                        elif key == 'temporalResidualCutoff':
                            config['temporal_residual_cutoff'] = value
                        elif key == 'temporalResidualConfidenceScale':
                            config['temporal_residual_confidence_scale'] = value
                        elif key == 'temporalMethod':
                            config['temporal_method'] = value
                        elif key == 'alpha':
                            config['alpha'] = value
                        elif key == 'minDensityGamma':
                            config['min_density_gamma'] = value
                        elif key == 'beta':
                            config['beta'] = value
                        elif key == 'withTemporalSampler':
                            config['has_temporal_sampler'] = value
                        elif key == 'withTemporalReconstruction':
                            config['has_temporal_reconstruction'] = value
                
                # Add fixed parameters for adaptive sampler
                config['sampler_type'] = 'adaptive'
                config['interpol_method'] = 'cubic'
                
                return config
            
            i = j + 1
            continue
        
        i += 1
    
    raise ValueError(f"Could not find {target_label} for {gt_name} at sparsity {scanned_pixel_percent}")


def extract_equidistant_frame_indices(total_frames: int, num_frames: int) -> List[int]:
    """
    Extract equidistantly spaced frame indices.
    
    Args:
        total_frames: Total number of frames in the video
        num_frames: Number of frames to extract (minimum 2)
    
    Returns:
        List of frame indices, always including first and last frames
    
    Examples:
        total_frames=10, num_frames=2 -> [0, 9]
        total_frames=10, num_frames=3 -> [0, 4, 9]
        total_frames=10, num_frames=5 -> [0, 2, 4, 7, 9]
    """
    if num_frames < 2:
        raise ValueError(f"num_frames must be at least 2, got {num_frames}")
    
    # Use the minimum of requested and available frames
    actual_num_frames = min(num_frames, total_frames)
    
    if actual_num_frames == 1:
        return [0]
    
    # Generate equidistant indices
    indices = []
    for i in range(actual_num_frames):
        idx = int(i * (total_frames - 1) / (actual_num_frames - 1))
        indices.append(idx)
    
    return indices


def apply_window_crop(
    frame: np.ndarray,
    window: Tuple[Union[int, float], Union[int, float], int, int],
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None
) -> np.ndarray:
    """
    Crop a frame to a window region.
    
    Args:
        frame: Input frame of shape (H, W) or (H, W, C)
        window: Tuple of (center_x, center_y, width, height)
            - center_x, center_y: Can be int (pixels) or float (0-1 percentage)
            - width, height: Always in pixels
        frame_width: Optional reference width (if not provided, uses frame.shape[1])
        frame_height: Optional reference height (if not provided, uses frame.shape[0])
    
    Returns:
        Cropped frame
    """
    H, W = frame.shape[:2]
    
    if frame_width is None:
        frame_width = W
    if frame_height is None:
        frame_height = H
    
    center_x, center_y, win_width, win_height = window
    
    # Convert center to pixels if percentage (0-1)
    if isinstance(center_x, float) and 0 <= center_x <= 1:
        center_x = int(center_x * frame_width)
    if isinstance(center_y, float) and 0 <= center_y <= 1:
        center_y = int(center_y * frame_height)
    
    center_x = int(center_x)
    center_y = int(center_y)
    
    # Calculate crop boundaries
    x_start = max(0, center_x - win_width // 2)
    x_end = min(W, x_start + win_width)
    y_start = max(0, center_y - win_height // 2)
    y_end = min(H, y_start + win_height)
    
    # Adjust if window exceeds boundaries
    if x_end - x_start < win_width:
        # Window goes beyond right edge, shift left
        x_start = max(0, x_end - win_width)
        x_end = x_start + win_width
        if x_end > W:
            x_end = W
            x_start = max(0, x_end - win_width)
    
    if y_end - y_start < win_height:
        # Window goes beyond bottom edge, shift up
        y_start = max(0, y_end - win_height)
        y_end = y_start + win_height
        if y_end > H:
            y_end = H
            y_start = max(0, y_end - win_height)
    
    # Crop the frame
    if frame.ndim == 2:
        return frame[y_start:y_end, x_start:x_end]
    else:
        return frame[y_start:y_end, x_start:x_end, ...]


def create_composite_strip(
    frames: List[np.ndarray],
    spacing: int = 5,
    target_height: Optional[int] = None
) -> np.ndarray:
    """
    Create a horizontal strip of frames with spacing.
    
    Args:
        frames: List of frames (each can be (H, W) or (H, W, C))
        spacing: Number of pixels between frames
        target_height: If provided, resize all frames to this height
    
    Returns:
        Composite strip as (H, total_width) or (H, total_width, C)
    """
    if not frames:
        return None
    
    # Determine if frames are grayscale or color
    is_color = frames[0].ndim == 3 and frames[0].shape[2] > 1
    
    # Get original dimensions
    ref_H = frames[0].shape[0]
    ref_W = frames[0].shape[1]
    
    # If target_height is specified, resize all frames
    if target_height is not None and target_height != ref_H:
        resized_frames = []
        for frame in frames:
            resized = resize_frame(frame, target_height, int(target_height * frame.shape[1] / frame.shape[0]))
            resized_frames.append(resized)
        frames = resized_frames
        ref_H = target_height
    
    # Calculate total width
    if is_color:
        frame_heights = [f.shape[0] for f in frames]
        frame_widths = [f.shape[1] for f in frames]
    else:
        frame_heights = [f.shape[0] for f in frames]
        frame_widths = [f.shape[1] for f in frames]
    
    total_width = sum(frame_widths) + spacing * (len(frames) - 1)
    
    # Create output array
    if is_color:
        output = np.zeros((ref_H, total_width, frames[0].shape[2]), dtype=frames[0].dtype)
    else:
        output = np.zeros((ref_H, total_width), dtype=frames[0].dtype)
    
    # Fill output with frames
    x_offset = 0
    for frame in frames:
        w = frame.shape[1]
        h = frame.shape[0]
        
        if is_color:
            output[:h, x_offset:x_offset+w, :] = frame[:h, :w, :]
        else:
            output[:h, x_offset:x_offset+w] = frame[:h, :w]
        
        x_offset += w + spacing
    
    return output


def stack_strips_vertically(
    strips: List[np.ndarray],
    vertical_spacing: int = 10,
    spacing_color: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Stack multiple horizontal strips vertically with spacing.
    
    Args:
        strips: List of horizontal strip images
        vertical_spacing: Number of pixels between strips
        spacing_color: Color for spacing (if None, uses black for RGB or 0 for grayscale)
    
    Returns:
        Composite image with all strips stacked vertically
    """
    if not strips:
        return None
    
    # Determine if we need RGB output
    if len(strips[0].shape) == 3 and strips[0].shape[2] == 3:
        is_rgb = True
    else:
        is_rgb = False
    
    # Calculate total height
    strip_heights = [s.shape[0] for s in strips]
    total_height = sum(strip_heights) + vertical_spacing * (len(strips) - 1)
    
    # Find maximum width
    max_width = max([s.shape[1] for s in strips])
    
    # Create output
    if is_rgb:
        if spacing_color is None:
            spacing_color = (0, 0, 0)
        output = np.zeros((total_height, max_width, 3), dtype=np.uint8)
        
        # Fill spacing with color
        for i in range(len(strips) - 1):
            y_start = sum(strip_heights[:i+1]) + i * vertical_spacing
            y_end = y_start + vertical_spacing
            output[y_start:y_end, :, :] = spacing_color
        
        # Fill strips
        y_offset = 0
        for i, strip in enumerate(strips):
            h, w = strip.shape[0], strip.shape[1]
            output[y_offset:y_offset+h, :w, :] = strip
            y_offset += h + vertical_spacing
    else:
        output = np.zeros((total_height, max_width), dtype=np.uint8)
        
        # Fill strips
        y_offset = 0
        for i, strip in enumerate(strips):
            h, w = strip.shape[0], strip.shape[1]
            output[y_offset:y_offset+h, :w] = strip
            y_offset += h + vertical_spacing
    
    return output


def ensure_same_frame_count(
    adaptive_frames: np.ndarray,
    stratified_frames: np.ndarray,
    low_dwell_frames: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensure all three frame sets have the same number of frames by trimming.
    
    Args:
        adaptive_frames: Adaptive reconstruction frames (T_a, H, W)
        stratified_frames: Stratified reconstruction frames (T_s, H, W)
        low_dwell_frames: Low-dwell frames (T_ld, H, W)
    
    Returns:
        Tuple of (adaptive, stratified, low_dwell) all with min(T_a, T_s, T_ld) frames
    """
    T_adaptive = adaptive_frames.shape[0]
    T_stratified = stratified_frames.shape[0]
    T_low_dwell = low_dwell_frames.shape[0]
    
    min_frames = min(T_adaptive, T_stratified, T_low_dwell)
    
    if T_adaptive > min_frames:
        adaptive_frames = adaptive_frames[:min_frames]
    if T_stratified > min_frames:
        stratified_frames = stratified_frames[:min_frames]
    if T_low_dwell > min_frames:
        low_dwell_frames = low_dwell_frames[:min_frames]
    
    return adaptive_frames, stratified_frames, low_dwell_frames


def ensure_same_dimensions(
    adaptive_frames: np.ndarray,
    stratified_frames: np.ndarray,
    low_dwell_frames: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensure all frames have the same spatial dimensions by resizing.
    Uses adaptive frames as reference.
    
    Args:
        adaptive_frames: Reference frames (T, H, W)
        stratified_frames: Frames to match (T, H_s, W_s)
        low_dwell_frames: Frames to match (T, H_ld, W_ld)
    
    Returns:
        Tuple of all frames resized to adaptive dimensions
    """
    ref_H, ref_W = adaptive_frames.shape[1], adaptive_frames.shape[2]
    
    # Resize stratified frames if needed
    if stratified_frames.shape[1:] != (ref_H, ref_W):
        print(f"Resizing stratified frames from {stratified_frames.shape[1:]} to {ref_H}x{ref_W}")
        resized_strat = []
        for t in range(stratified_frames.shape[0]):
            resized = resize_frame(stratified_frames[t], ref_H, ref_W)
            resized_strat.append(resized)
        stratified_frames = np.stack(resized_strat, axis=0)
    
    # Resize low_dwell frames if needed
    if low_dwell_frames.shape[1:] != (ref_H, ref_W):
        print(f"Resizing low-dwell frames from {low_dwell_frames.shape[1:]} to {ref_H}x{ref_W}")
        resized_ld = []
        for t in range(low_dwell_frames.shape[0]):
            resized = resize_frame(low_dwell_frames[t], ref_H, ref_W)
            resized_ld.append(resized)
        low_dwell_frames = np.stack(resized_ld, axis=0)
    
    return adaptive_frames, stratified_frames, low_dwell_frames


def build_output_filename(
    gt_name: str,
    sparsity: float,
    target_metric: str,
    num_frames: int,
    window: Optional[Tuple] = None
) -> str:
    """Build a descriptive output filename."""
    parts = [
        f"frame_strip_{gt_name}",
        f"s{sparsity}",
        f"metric_{target_metric}",
        f"frames_{num_frames}"
    ]
    
    if window is not None:
        center_x, center_y, win_w, win_h = window
        # Indicate window is present
        parts.append("windowed")
        # Add window dimensions
        parts.append(f"w{win_w}x{win_h}")
    
    return "_".join(parts) + ".tiff"


def create_frame_strips(
    gt_name: str,
    scanned_pixel_percent: float,
    num_frames: int,
    target_metric: str,
    window: Optional[Tuple[Union[int, float], Union[int, float], int, int]] = None,
    analysis_file_path: str = "parameter_tune_analysis.txt",
    base_dir: str = "plots",
    output_dir: Optional[str] = None,
    result_type: str = "reconstruction",
    horizontal_spacing: int = 5,
    vertical_spacing: int = 10
) -> str:
    """
    Create frame strips visualization comparing adaptive, stratified, and low-dwell results.
    
    Args:
        gt_name: Ground truth name (e.g., "LI_EXPULSION_ONE_10FPS")
        scanned_pixel_percent: Sparsity value (e.g., 0.1, 0.5, 1.0)
        num_frames: Number of frames to extract (minimum 2)
        target_metric: One of "psnr", "ssim", or "combined"
        window: Optional window tuple (center_x, center_y, width, height)
            - center_x, center_y: Can be int (pixels) or float (0-1 percentage)
            - width, height: In pixels
        analysis_file_path: Path to parameter_tune_analysis.txt
        base_dir: Base directory for experiment results
        output_dir: Directory to save output (default: same as script dir)
        result_type: Type of result to load (default: "reconstruction")
        horizontal_spacing: Pixels between frames in a row
        vertical_spacing: Pixels between rows
    
    Returns:
        Path to the generated TIFF file
    
    Workflow:
        1. Parse best config for given GT, sparsity, and metric
        2. Build paths to adaptive, stratified, and low-dwell results
        3. Load all frames
        4. Ensure same frame count and dimensions
        5. Extract equidistant frames
        6. Apply window cropping if specified
        7. Create 3-row composite image
        8. Save as TIFF
    """
    print(f"\nProcessing frame strips for {gt_name} at sparsity {scanned_pixel_percent}...")
    print(f"  Target metric: {target_metric}, Num frames: {num_frames}")
    
    # 1. Parse best config
    print(f"  Parsing best config from {analysis_file_path}...")
    config = parse_best_config_by_metric(
        analysis_file_path, gt_name, scanned_pixel_percent, target_metric
    )
    print(f"  Found config: alpha={config.get('alpha')}, adaptive_fraction={config.get('adaptive_fraction')}")
    
    # 2. Build paths
    adaptive_dir = build_adaptive_path_from_config(config, base_dir)
    stratified_dir = build_stratified_path(
        gt_name, scanned_pixel_percent, config.get('interpol_method', 'cubic'), base_dir
    )
    
    print(f"  Adaptive path: {adaptive_dir}")
    print(f"  Stratified path: {stratified_dir}")
    
    # 3. Load frames
    print(f"  Loading frames...")
    adaptive_frames = load_frames_from_path(adaptive_dir, result_type)
    stratified_frames = load_frames_from_path(stratified_dir, result_type)
    low_dwell_frames = load_all_low_dwell_frames(gt_name, scanned_pixel_percent, base_dir)
    
    print(f"  Loaded: adaptive={adaptive_frames.shape}, stratified={stratified_frames.shape}, low_dwell={low_dwell_frames.shape}")
    
    # 4. Ensure same frame count and dimensions
    adaptive_frames, stratified_frames, low_dwell_frames = ensure_same_frame_count(
        adaptive_frames, stratified_frames, low_dwell_frames
    )
    adaptive_frames, stratified_frames, low_dwell_frames = ensure_same_dimensions(
        adaptive_frames, stratified_frames, low_dwell_frames
    )
    
    T = adaptive_frames.shape[0]
    print(f"  After alignment: all have {T} frames of size {adaptive_frames.shape[1:]} ")
    
    # 5. Extract equidistant frames
    frame_indices = extract_equidistant_frame_indices(T, num_frames)
    print(f"  Extracting frames at indices: {frame_indices}")
    
    adaptive_selected = adaptive_frames[frame_indices]
    stratified_selected = stratified_frames[frame_indices]
    low_dwell_selected = low_dwell_frames[frame_indices]
    
    # 6. Apply window cropping if specified
    if window is not None:
        print(f"  Applying window crop: center={window[0], window[1]}, size={window[2]}x{window[3]}")
        ref_H, ref_W = adaptive_frames.shape[1], adaptive_frames.shape[2]
        
        adaptive_selected = [
            apply_window_crop(f, window, ref_W, ref_H) 
            for f in adaptive_selected
        ]
        stratified_selected = [
            apply_window_crop(f, window, ref_W, ref_H) 
            for f in stratified_selected
        ]
        low_dwell_selected = [
            apply_window_crop(f, window, ref_W, ref_H) 
            for f in low_dwell_selected
        ]
    
    # Convert to lists for create_composite_strip
    adaptive_list = [adaptive_selected[i] for i in range(len(adaptive_selected))]
    stratified_list = [stratified_selected[i] for i in range(len(stratified_selected))]
    low_dwell_list = [low_dwell_selected[i] for i in range(len(low_dwell_selected))]
    
    # 7. Create strips
    print(f"  Creating composite strips...")
    adaptive_strip = create_composite_strip(adaptive_list, spacing=horizontal_spacing)
    stratified_strip = create_composite_strip(stratified_list, spacing=horizontal_spacing)
    low_dwell_strip = create_composite_strip(low_dwell_list, spacing=horizontal_spacing)
    
    print(f"  Strip sizes: adaptive={adaptive_strip.shape}, stratified={stratified_strip.shape}, low_dwell={low_dwell_strip.shape}")
    
    # 8. Stack strips vertically
    final_image = stack_strips_vertically(
        [adaptive_strip, stratified_strip, low_dwell_strip],
        vertical_spacing=vertical_spacing
    )
    
    print(f"  Final image size: {final_image.shape}")
    
    # 9. Save output
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    os.makedirs(output_dir, exist_ok=True)
    
    filename = build_output_filename(gt_name, scanned_pixel_percent, target_metric, num_frames, window)
    # Sanitize filename (replace problematic characters)
    filename = filename.replace(" ", "_").replace("(", "_").replace(")", "_")
    output_path = os.path.join(output_dir, filename)
    
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception as e:
            print(f"  Warning: Could not remove existing file: {e}")
    
    if final_image.ndim == 3:
        tifffile.imwrite(output_path, final_image, photometric='rgb')
    else:
        tifffile.imwrite(output_path, final_image)
    
    print(f"  Saved to: {output_path}")
    return output_path


# Example usage and CLI
if __name__ == '__main__':
    # Example: Create frame strips for LI_EXPULSION_ONE_10FPS at sparsity 0.1
    # Using best PSNR config, 3 frames, no window
    print("=" * 80)
    print("Frame Strips for Paper - Example Usage")
    print("=" * 80)
    
    # Example 1: Basic usage
    output_path = create_frame_strips(
        gt_name="LI_EXPULSION_ONE_10FPS",
        scanned_pixel_percent=0.1,
        num_frames=3,
        target_metric="psnr",
        window=None,
        analysis_file_path="statistics/parameter_tune_analysis.txt",
        base_dir="plots",
        output_dir="statistics/frame_strips"
    )
    print(f"\nGenerated: {output_path}")
    
    # Example 2: With window cropping (center at 50%, 50%, window size 200x200)
    # output_path = create_frame_strips(
    #     gt_name="LI_EXPULSION_ONE_10FPS",
    #     scanned_pixel_percent=0.1,
    #     num_frames=5,
    #     target_metric="combined",
    #     window=(0.5, 0.5, 200, 200),  # Center at 50%, 50%, 200x200 window
    #     analysis_file_path="statistics/parameter_tune_analysis.txt",
    #     base_dir="plots",
    #     output_dir="statistics/frame_strips"
    # )
    # print(f"\nGenerated: {output_path}")
