#!/usr/bin/env python3
"""
Experiment Result Comparison Script

Generates composite multi-frame TIF stacks comparing experiment results frame-by-frame.
Each page in the output TIF is a grid layout with experiment results and optional ground truths.

Layout Rules:
- With GTs (n_gt > 0): Experiments max 3 columns, GTs in single column on right
- Without GTs (n_gt == 0): Experiments max 4 columns

Usage:
    python experiment_comparison.py \
        --configs config1.json config2.json ... \
        --result-type reconstruction \
        --gt-name HYDRATION_ONE \
        --noise-equivalent 0.5 \
        --include-full-gt \
        --output comparison.tiff

Or programmatically:
    from experiment_comparison import compare_experiments
    compare_experiments(configs, result_type="reconstruction", ...)
"""

import argparse
import os
import sys
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image
from tifffile import tifffile

# Add parent directory to path for importing experiment_common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment_common import GROUNDTRUTH_MAP, _ground_truth_path


# Debug image kinds available in the system
ALL_DEBUG_IMAGE_KINDS = [
    "reconstruction", "samples", "pdf", "pdf_spatial", "pdf_temporal",
    "flow", "temporal_variance", "error", "psnr", "ssim", "triangulation"
]

# Path-defining parameters (all ExperimentRun params except min_density_gamma)
REQUIRED_CONFIG_KEYS = {
    "sampler_type", "interpol_method", "scanned_pixel_percent", "gt_name",
    "has_temporal_sampler", "has_temporal_reconstruction",
    "alpha", "adaptive_fraction",
    "temporal_method", "temporal_residual_cutoff", "temporal_residual_confidence_scale",
    "sample_sequence"
}

# Border colors for ground truth frames
GT_BORDER_COLORS = {
    "full": (0, 255, 0),      # Green for full-scan GT
    "noise_equivalent": (255, 165, 0)  # Orange for noise-equivalent GT
}


def resize_frame(frame: np.ndarray, target_H: int, target_W: int) -> np.ndarray:
    """
    Resize a frame to target dimensions using PIL's Image.resize with LANCZOS filter.
    
    Args:
        frame: Input frame array (H, W) or (H, W, C)
        target_H: Target height
        target_W: Target width
        
    Returns:
        Resized frame array with same dtype and number of channels
    """
    # Get current dimensions and track if we need to restore single channel
    is_single_channel_3d = frame.ndim == 3 and frame.shape[2] == 1
    
    if frame.ndim == 2:
        # Grayscale
        img = Image.fromarray(frame, mode='L')
    elif frame.ndim == 3:
        if frame.shape[2] == 3:
            img = Image.fromarray(frame, mode='RGB')
        elif frame.shape[2] == 4:
            img = Image.fromarray(frame, mode='RGBA')
        else:
            # Single channel but 3D - take first channel
            img = Image.fromarray(frame[..., 0], mode='L')
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")
    
    # Resize using LANCZOS (high-quality downsampling)
    img = img.resize((target_W, target_H), Image.LANCZOS)
    
    # Convert back to numpy
    result = np.array(img, dtype=frame.dtype)
    
    # If input was 3D with single channel, restore that structure
    if is_single_channel_3d:
        result = result[..., np.newaxis]
    
    return result


def build_experiment_path(config: Dict, base_dir: str = "plots") -> str:
    """
    Build the filesystem path for an experiment's result directory.
    
    The path structure is:
    {base_dir}/examples/{sampler_type}/interpol_{interpol_method}/
    sparsity_{scanned_pixel_percent}/{gt_name}/
    sampler_{has_temporal_sampler}_reconstruction_{has_temporal_reconstruction}/
    temporalMethod_{temporal_method}/temporalResidualCutoff_{temporal_residual_cutoff}/
    temporalResidualConfidenceScale_{temporal_residual_confidence_scale}/
    sampleSequence_{sample_sequence}/alpha_{alpha}/adaptive_{adaptive_fraction}/
    
    Args:
        config: Dict with experiment configuration parameters
        base_dir: Base directory (default: "plots")
        
    Returns:
        Full path string to the experiment directory
    """
    # Validate required keys
    missing = REQUIRED_CONFIG_KEYS - set(config.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    
    path_parts = [
        base_dir,
        "examples",
        config["sampler_type"],
        f"interpol_{config['interpol_method']}",
        f"sparsity_{config['scanned_pixel_percent']}",
        config["gt_name"],
        f"sampler_{config['has_temporal_sampler']}_"
        f"reconstruction_{config['has_temporal_reconstruction']}",
        f"temporalMethod_{config['temporal_method']}",
        f"temporalResidualCutoff_{config['temporal_residual_cutoff']}",
        f"temporalResidualConfidenceScale_{config['temporal_residual_confidence_scale']}",
        f"sampleSequence_{config['sample_sequence']}",
        f"alpha_{config['alpha']}",
        f"adaptive_{config['adaptive_fraction']}"
    ]
    
    return os.path.join(*path_parts)


def load_all_frames_from_experiment(
        config: Dict, result_type: str, base_dir: str = "plots"
) -> np.ndarray:
    """
    Load all frames from an experiment's multi-frame TIFF result.
    
    Args:
        config: Experiment configuration dict
        result_type: Debug image kind (e.g., "reconstruction", "samples")
        base_dir: Base directory for experiment results
        
    Returns:
        3D numpy array with shape (T, H, W) containing all frames as uint8
    """
    # Build path to experiment directory
    exp_dir = build_experiment_path(config, base_dir)
    
    # Path to the result file
    result_file = os.path.join(exp_dir, f"{result_type}.tiff")
    
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Result file not found: {result_file}")
    
    # Load multi-frame TIFF
    # First, try to inspect the TIFF file to get accurate page count
    with tifffile.TiffFile(result_file) as tif:
        num_pages = len(tif.pages)
        print(f"    TIFF file has {num_pages} pages")
        
        # Load all pages - use pages directly to ensure we get all of them
        frames = tif.asarray(key=None)  # key=None should load all pages
        # If that doesn't work, try loading pages individually
        if frames.ndim == 2:
            print(f"    asarray() only returned 2D, loading pages individually")
            frames_list = []
            for page in tif.pages:
                page_data = page.asarray()
                frames_list.append(page_data)
            frames = np.stack(frames_list, axis=0)
    
    # Debug: print original shape
    print(f"    Raw loaded shape: {frames.shape}, dtype: {frames.dtype}, ndim: {frames.ndim}")
    
    # Handle different TIFF formats
    if frames.ndim == 2:
        # Single frame - reshape to (1, H, W)
        print(f"    Converting 2D to 3D (1, H, W)")
        frames = frames[np.newaxis, ...]
    elif frames.ndim == 3:
        # Could be (H, W, C) or (T, H, W)
        # Check if first dimension is small (likely frames) or large (likely height)
        if frames.shape[0] < 100:  # Assume this is T
            print(f"    Assuming shape is (T, H, W) with T={frames.shape[0]}")
            pass  # Already (T, H, W)
        else:
            # This is (H, W, C) - single frame with channels
            print(f"    Assuming shape is (H, W, C) with H={frames.shape[0]}")
            print("    converting to (1, H, W)")
            frames = frames[np.newaxis, ...]
    elif frames.ndim == 4:
        # (T, H, W, C) - convert to (T, H, W) by taking first channel
        print(f"    Converting 4D (T, H, W, C) to 3D (T, H, W) by taking first channel")
        if frames.shape[-1] == 1:
            frames = frames[..., 0]
        else:
            # Take first channel
            frames = frames[..., 0]
    
    # Convert to uint8 if needed
    if frames.dtype != np.uint8:
        # Scale to 0-255 if float
        if np.issubdtype(frames.dtype, np.floating):
            print(f"    Converting from float to uint8")
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)
    
    print(f"    Final shape: {frames.shape}")
    return frames


def load_all_low_dwell_frames(gt_name: str, sparsity: float, base_dir: str = "plots") -> np.ndarray:
    """
    Load all frames from low-dwell results.
    
    Low-dwell files are stored as individual files: frame_{XXX}_low_dwell.tiff
    
    Args:
        gt_name: Ground truth name
        sparsity: Scanned pixel percentage (e.g., 0.1, 0.5)
        base_dir: Base directory for results
        
    Returns:
        3D numpy array with shape (T, H, W) containing all frames as uint8
    """
    # Build path to low-dwell directory
    low_dwell_dir = os.path.join(base_dir, "examples", "low_dwell", f"sparsity_{sparsity}", gt_name)
    
    if not os.path.exists(low_dwell_dir):
        raise FileNotFoundError(f"Low-dwell directory not found: {low_dwell_dir}")
    
    # Find all frame files
    frame_files = sorted([
        f for f in os.listdir(low_dwell_dir) 
        if f.startswith("frame_") and f.endswith("_low_dwell.tiff")
    ])
    
    if not frame_files:
        raise FileNotFoundError(f"No low-dwell frame files found in: {low_dwell_dir}")
    
    # Load all frames
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(low_dwell_dir, frame_file)
        frame = tifffile.imread(frame_path)
        
        # Convert to 2D uint8
        if frame.ndim == 3:
            if frame.shape[-1] == 1:
                frame = frame[..., 0]
            else:
                frame = frame[..., 0]  # Take first channel
        
        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        frames.append(frame)
    
    return np.stack(frames, axis=0)


def load_all_ground_truth_frames(gt_name: str) -> np.ndarray:
    """
    Load all frames from full-scan ground truth.
    
    Args:
        gt_name: Ground truth name
        
    Returns:
        3D numpy array with shape (T, H, W) containing all frames as uint8
    """
    # Get ground truth path
    gt_path = _ground_truth_path(gt_name)
    
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    
    # Load multi-frame TIFF
    frames = tifffile.imread(gt_path)
    
    # Handle different formats
    if frames.ndim == 2:
        frames = frames[np.newaxis, ...]
    elif frames.ndim == 3:
        if frames.shape[0] < 100:
            pass  # (T, H, W)
        else:
            frames = frames[np.newaxis, ...]  # (H, W, C) -> (1, H, W, C)
    elif frames.ndim == 4:
        if frames.shape[-1] == 1:
            frames = frames[..., 0]
        else:
            frames = frames[..., 0]
    
    # Convert to uint8
    if frames.dtype != np.uint8:
        if np.issubdtype(frames.dtype, np.floating):
            # Normalize to 0-255
            frame_min = frames.min()
            frame_max = frames.max()
            if frame_max > frame_min:
                frames = ((frames - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
            else:
                frames = np.zeros_like(frames, dtype=np.uint8)
        else:
            frames = frames.astype(np.uint8)
    
    return frames


def add_colored_border(
        image: np.ndarray, color: Tuple[int, int, int], width: int = 3
) -> np.ndarray:
    """
    Add a colored border to an image.
    
    Args:
        image: 2D grayscale image array (H, W) as uint8
        color: RGB tuple for border color
        width: Border width in pixels
        
    Returns:
        3D RGB image array (H + 2*width, W + 2*width, 3) with border
    """
    if image.ndim == 2:
        # Convert grayscale to RGB
        image_rgb = np.stack([image, image, image], axis=-1)
    else:
        image_rgb = image.copy()
    
    H, W = image.shape[:2]
    
    # Create new image with border
    new_H = H + 2 * width
    new_W = W + 2 * width
    bordered = np.zeros((new_H, new_W, 3), dtype=np.uint8)
    
    # Fill border with color
    # Top border
    bordered[:width, :, :] = color
    # Bottom border
    bordered[-width:, :, :] = color
    # Left border
    bordered[:, :width, :] = color
    # Right border
    bordered[:, -width:, :] = color
    
    # Fill center with image
    bordered[width:width+H, width:width+W, :] = image_rgb
    
    return bordered


def create_composite_page(exp_frames_2d: List[np.ndarray], 
                          gt_frames_2d: List[np.ndarray], 
                          gt_colors: List[Tuple[int, int, int]]) -> np.ndarray:
    """
    Create a single composite grid page from 2D frame slices.
    
    Layout:
    - With GTs (n_gt > 0): Experiments max 3 columns, GTs in single column on right
    - Without GTs (n_gt == 0): Experiments max 4 columns
    
    Args:
        exp_frames_2d: List of 2D experiment frame arrays (H, W)
        gt_frames_2d: List of 2D ground truth frame arrays (H, W)
        gt_colors: List of RGB color tuples for each GT frame
        
    Returns:
        3D RGB composite image array (composite_H, composite_W, 3)
    """
    n_exp = len(exp_frames_2d)
    n_gt = len(gt_frames_2d)
    
    if n_exp == 0:
        raise ValueError("At least one experiment frame required")
    
    # Determine max columns for experiments
    if n_gt > 0:
        max_exp_cols = 3
    else:
        max_exp_cols = 4
    
    # Calculate grid dimensions for experiments
    exp_cols = min(max_exp_cols, n_exp)
    exp_rows = (n_exp + exp_cols - 1) // exp_cols
    
    # Total columns = exp_cols + 1 (GTs go in a single column on the right)
    total_cols = exp_cols + 1
    
    # Total rows = max(exp_rows, n_gt) since GTs stack vertically in their column
    total_rows = max(exp_rows, n_gt)
    
    # Get dimensions from first frame
    H, W = exp_frames_2d[0].shape
    
    # Create output image (RGB)
    # Each cell has the original frame size
    output_H = total_rows * H
    output_W = total_cols * W
    output = np.zeros((output_H, output_W, 3), dtype=np.uint8)
    
    # Fill experiment frames
    for i, frame in enumerate(exp_frames_2d):
        row = i // exp_cols
        col = i % exp_cols
        
        # Convert to RGB if grayscale
        if frame.ndim == 2:
            frame_rgb = np.stack([frame, frame, frame], axis=-1)
        else:
            frame_rgb = frame.copy()
        
        y_start = row * H
        y_end = y_start + H
        x_start = col * W
        x_end = x_start + W
        
        output[y_start:y_end, x_start:x_end, :] = frame_rgb
    
    # Fill GT frames (with borders) - all in the rightmost column
    gt_col = exp_cols  # All GTs go in the rightmost column (index = exp_cols)
    
    for i, (gt_frame, color) in enumerate(zip(gt_frames_2d, gt_colors)):
        row = i  # GTs stack vertically in their column
        
        # Convert to RGB if grayscale
        if gt_frame.ndim == 2:
            gt_frame_rgb = np.stack([gt_frame, gt_frame, gt_frame], axis=-1)
        else:
            gt_frame_rgb = gt_frame.copy()
        
        # Add border
        bordered = add_colored_border(gt_frame_rgb, color, width=3)
        
        # Calculate position (border adds to dimensions)
        bordered_H, bordered_W = bordered.shape[:2]
        
        y_start = row * H
        y_end = y_start + bordered_H
        x_start = gt_col * W
        x_end = x_start + bordered_W
        
        # Ensure we don't exceed output bounds
        if y_end > output_H or x_end > output_W:
            # Resize output if needed (shouldn't happen with correct layout)
            new_H = max(output_H, y_end)
            new_W = max(output_W, x_end)
            new_output = np.zeros((new_H, new_W, 3), dtype=np.uint8)
            new_output[:output_H, :output_W, :] = output
            output = new_output
            output_H, output_W = new_H, new_W
        
        output[y_start:y_end, x_start:x_end, :] = bordered
    
    return output


def validate_config(config: Dict) -> None:
    """
    Validate an experiment configuration dict.
    
    Args:
        config: Experiment configuration dict
        
    Raises:
        ValueError: If required keys are missing or values are invalid
    """
    missing = REQUIRED_CONFIG_KEYS - set(config.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    
    # Check that boolean values are actual booleans
    bool_keys = {"has_temporal_sampler", "has_temporal_reconstruction"}
    for key in bool_keys:
        if key in config and not isinstance(config[key], bool):
            raise ValueError(f"{key} must be a boolean, got {type(config[key])}")


def validate_result_type(result_type: str) -> None:
    """
    Validate the result type.
    
    Args:
        result_type: Debug image kind
        
    Raises:
        ValueError: If result_type is not valid
    """
    if result_type not in ALL_DEBUG_IMAGE_KINDS:
        raise ValueError(
            f"Invalid result_type '{result_type}'. "
            f"Must be one of: {ALL_DEBUG_IMAGE_KINDS}"
        )


def compare_experiments(
    experiment_configs: List[Dict],
    result_type: str = "reconstruction",
    gt_name: Optional[str] = None,
    noise_equivalent_sparsity: Optional[float] = None,
    include_full_gt: bool = False,
    output_path: Optional[str] = None,
    base_dir: str = "plots"
) -> str:
    """
    Compare experiments and generate composite multi-frame TIF stack.
    
    Each page in the output TIF is a composite grid for one frame, with:
    - Experiment results on the left (max 3 columns if GTs present, else max 4)
    - Ground truths on the right in single column with colored borders
    
    Args:
        experiment_configs: List of experiment configuration dicts
        result_type: Debug image kind to compare (default: "reconstruction")
        gt_name: Ground truth name (optional, inferred from configs if all share same)
        noise_equivalent_sparsity: Pixel percentage for low-dwell GT (optional)
        include_full_gt: Whether to include full-scan GT (default: False)
        output_path: Output TIF path (optional, auto-generated if None)
        base_dir: Base directory for experiment results (default: "plots")
        
    Returns:
        Path to saved composite TIF
        
    Raises:
        ValueError: If configs have different gt_names, missing params, etc.
    """
    # Validate inputs
    if not experiment_configs:
        raise ValueError("At least one experiment config required")
    
    # Validate result type
    validate_result_type(result_type)
    
    # Validate all configs
    for i, config in enumerate(experiment_configs):
        validate_config(config)
    
    # Check all configs have same gt_name
    gt_names = {config["gt_name"] for config in experiment_configs}
    if len(gt_names) > 1:
        raise ValueError(f"All configs must have same gt_name, got: {gt_names}")
    
    # Use provided gt_name or infer from configs
    if gt_name is None:
        gt_name = next(iter(gt_names))
    elif gt_name not in gt_names:
        raise ValueError(f"gt_name '{gt_name}' doesn't match configs: {gt_names}")
    
    # Load all frames from experiments
    exp_frame_arrays = []  # List of (T, H, W) arrays
    ref_H, ref_W = None, None
    
    for config in experiment_configs:
        frames = load_all_frames_from_experiment(config, result_type, base_dir)
        exp_frame_arrays.append(frames)
        
        # Debug output
        print(f"  Loaded {frames.shape[0]} frames from {result_type}.tiff")
        print(f"    (H={frames.shape[1]}, W={frames.shape[2]})")
        
        # Check dimensions match reference
        if ref_H is None:
            ref_H, ref_W = frames.shape[1], frames.shape[2]
        elif frames.shape[1] != ref_H or frames.shape[2] != ref_W:
            raise ValueError(
                f"Frame dimensions mismatch: {frames.shape} vs reference ({ref_H}, {ref_W})"
            )
    
    # Determine total frames T
    T = exp_frame_arrays[0].shape[0]
    print(f"  Total frames to process: {T}")
    
    # Load GT frames
    gt_frame_arrays = []  # List of (T, H, W) arrays
    gt_color_list = []    # List of border colors
    
    # Noise-equivalent GT
    if noise_equivalent_sparsity is not None:
        gt_frames = load_all_low_dwell_frames(gt_name, noise_equivalent_sparsity, base_dir)
        
        # Check and fix dimensions if needed
        if gt_frames.shape[1] != ref_H or gt_frames.shape[2] != ref_W:
            print(f"  Resizing low-dwell GT from {gt_frames.shape[1:]} to {ref_H}x{ref_W}")
            resized_frames = []
            for t in range(gt_frames.shape[0]):
                resized = resize_frame(gt_frames[t], ref_H, ref_W)
                resized_frames.append(resized)
            gt_frames = np.stack(resized_frames, axis=0)
        
        # Check frame count
        if gt_frames.shape[0] != T:
            raise ValueError(
                f"Low-dwell GT has {gt_frames.shape[0]} frames, "
                f"expected {T}"
            )
        
        gt_frame_arrays.append(gt_frames)
        gt_color_list.append(GT_BORDER_COLORS["noise_equivalent"])
    
    # Full-scan GT
    if include_full_gt:
        gt_frames = load_all_ground_truth_frames(gt_name)
        
        # Check and fix dimensions if needed
        if gt_frames.shape[1] != ref_H or gt_frames.shape[2] != ref_W:
            print(f"  Resizing full-scan GT from {gt_frames.shape[1:]} to {ref_H}x{ref_W}")
            resized_frames = []
            for t in range(gt_frames.shape[0]):
                resized = resize_frame(gt_frames[t], ref_H, ref_W)
                resized_frames.append(resized)
            gt_frames = np.stack(resized_frames, axis=0)
        
        # Check frame count
        if gt_frames.shape[0] != T:
            # If GT has more frames than experiments, take only the first T frames
            if gt_frames.shape[0] > T:
                print(f"  Truncating full GT from {gt_frames.shape[0]} frames to {T}")
                gt_frames = gt_frames[:T]
            else:
                raise ValueError(
                    f"Full GT has {gt_frames.shape[0]} frames, "
                    f"expected {T}"
                )
        
        gt_frame_arrays.append(gt_frames)
        gt_color_list.append(GT_BORDER_COLORS["full"])
    
    # Generate output path if not provided
    if output_path is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(
            output_dir, 
            f"comparison_{result_type}_{gt_name}.tiff"
        )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Remove existing file if it exists to avoid locked file issues on Windows
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            print(f"  Removed existing file: {output_path}")
        except Exception as e:
            print(f"  Warning: Could not remove existing file {output_path}: {e}")
    
    # Create composite pages for each frame
    composite_pages = []
    
    for t in range(T):
        # Extract 2D slices for this frame
        exp_frames_2d = [arr[t] for arr in exp_frame_arrays]
        gt_frames_2d = [arr[t] for arr in gt_frame_arrays]
        
        # Create composite page
        page = create_composite_page(exp_frames_2d, gt_frames_2d, gt_color_list)
        composite_pages.append(page)
        print(f"  Created composite page {t}: shape {page.shape}")
    
    print(f"  Total composite pages: {len(composite_pages)}")
    
    # Save as multi-frame TIF
    # Write each page individually to ensure multi-page structure
    
    # Check if we have multiple pages
    if len(composite_pages) <= 1:
        # Single frame - save directly
        print(f"  Writing single page to {output_path}")
        tifffile.imwrite(output_path, composite_pages[0], photometric='rgb')
    else:
        # Multiple frames - use TiffWriter to ensure proper multi-page output
        # Use bigtiff=False to match the reference script's behavior
        print(f"  Writing {len(composite_pages)} pages to {output_path} using TiffWriter")
        with tifffile.TiffWriter(output_path, bigtiff=False) as tif:
            for i, page in enumerate(composite_pages):
                tif.write(page, photometric='rgb', description=f'frame {i}')
    
    return output_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Compare experiment results and generate composite TIF stack'
    )
    
    parser.add_argument(
        '--configs', type=str, nargs='+', required=True,
        help='JSON files containing experiment configurations'
    )
    parser.add_argument(
        '--result-type', type=str, default='reconstruction',
        help='Debug image kind to compare (default: reconstruction)'
    )
    parser.add_argument(
        '--gt-name', type=str, default=None,
        help='Ground truth name (optional, inferred from configs)'
    )
    parser.add_argument(
        '--noise-equivalent', type=float, default=None,
        help='Pixel percentage for noise-equivalent ground truth'
    )
    parser.add_argument(
        '--include-full-gt', action='store_true', default=False,
        help='Include full-scan ground truth'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output TIF path (optional, auto-generated)'
    )
    parser.add_argument(
        '--base-dir', type=str, default='plots',
        help='Base directory for experiment results (default: plots)'
    )
    
    args = parser.parse_args()
    
    # Load configs from JSON files
    import json
    experiment_configs = []
    for config_file in args.configs:
        with open(config_file, 'r') as f:
            config = json.load(f)
            experiment_configs.append(config)
    
    # Run comparison
    output_path = compare_experiments(
        experiment_configs=experiment_configs,
        result_type=args.result_type,
        gt_name=args.gt_name,
        noise_equivalent_sparsity=args.noise_equivalent,
        include_full_gt=args.include_full_gt,
        output_path=args.output,
        base_dir=args.base_dir
    )
    
    print(f"Composite TIF saved to: {output_path}")


if __name__ == '__main__':
    main()
