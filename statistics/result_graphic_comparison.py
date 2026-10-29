#!/usr/bin/env python3
"""
Result Graphic Comparison Script

Generates a composite multi-frame TIF stack comparing a single adaptive experiment
against its stratified counterpart with both ground truths.

Layout: Fixed 4-column, 1-row grid per frame
[Exp (adaptive)] [StratComp (stratified)] [LowDwellGT] [OriginalGT]

Usage (programmatic only):
    from result_graphic_comparison import create_result_graphic_comparison
    
    config = {
        "sampler_type": "adaptive",
        "interpol_method": "cubic",
        "scanned_pixel_percent": 0.5,
        "gt_name": "LI_EXPULSION_TWO_ORIGINAL",
        "has_temporal_sampler": True,
        "has_temporal_reconstruction": True,
        "alpha": 5.0,
        "adaptive_fraction": 0.5,
        "minDensityGamma": 0.1,
        "temporal_method": "temporal_variance",
        "temporal_residual_cutoff": 12.0,
        "temporal_residual_confidence_scale": 250.0,
        "sample_sequence": "halton"
    }
    
    output_path = create_result_graphic_comparison(
        config=config,
        result_type="reconstruction",
        base_dir="plots",
        output_path="output_comparison.tiff"
    )
"""

import os
import sys
from typing import Dict, Optional

import numpy as np
from PIL import Image
from tifffile import tifffile

# Add parent directory to path for importing experiment_common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment_common import GROUNDTRUTH_MAP, _ground_truth_path

# Border colors for ground truth frames
GT_BORDER_COLORS = {
    "full": (0, 255, 0),      # Green for full-scan GT
    "noise_equivalent": (255, 165, 0)  # Orange for noise-equivalent GT
}


def resize_frame(frame, target_H, target_W):
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


def build_stratified_path(gt_name, scanned_pixel_percent, 
                           interpol_method="cubic", base_dir="plots"):
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


def load_frames_from_path(exp_dir, result_type):
    result_file = os.path.join(exp_dir, f"{result_type}.tiff")
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Result file not found: {result_file}")
    with tifffile.TiffFile(result_file) as tif:
        num_pages = len(tif.pages)
        frames = tif.asarray(key=None)
        if frames.ndim == 2:
            frames_list = [page.asarray() for page in tif.pages]
            frames = np.stack(frames_list, axis=0)
    
    # Handle different TIFF formats - goal is to get (T, H, W) as uint8
    if frames.ndim == 2:
        # Single frame - add time dimension
        frames = frames[np.newaxis, ...]
    elif frames.ndim == 3:
        # Could be (T, H, W) or (H, W, C)
        # If first dimension is small (< 500), assume it's T
        # If first dimension is large (>= 500), assume it's H and this is (H, W, C)
        if frames.shape[0] < 500:
            # This is (T, H, W)
            pass
        else:
            # This is likely (H, W, C) - single frame with channels
            frames = frames[np.newaxis, ...]
    elif frames.ndim == 4:
        # Could be (T, H, W, C) - take first channel
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


def load_all_low_dwell_frames(gt_name, sparsity, base_dir="plots"):
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
                # This is (T, H, W) - but we're loading individual files, so this shouldn't happen
                frame = frame[0] if frame.shape[0] == 1 else frame[..., 0]
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if np.issubdtype(frame.dtype, np.floating) else frame.astype(np.uint8)
        frames.append(frame)
    return np.stack(frames, axis=0)


def load_all_ground_truth_frames(gt_name):
    gt_path = _ground_truth_path(gt_name)
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    frames = tifffile.imread(gt_path)
    
    # Handle different TIFF formats - goal is to get (T, H, W) as uint8
    if frames.ndim == 2:
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
            fmin, fmax = frames.min(), frames.max()
            frames = ((frames - fmin) / (fmax - fmin) * 255).astype(np.uint8) if fmax > fmin else np.zeros_like(frames, dtype=np.uint8)
        else:
            frames = frames.astype(np.uint8)
    return frames


def add_colored_border(image, color, width=3):
    if image.ndim == 2:
        image_rgb = np.stack([image, image, image], axis=-1)
    else:
        image_rgb = image.copy()
    H, W = image.shape[:2]
    new_H, new_W = H + 2 * width, W + 2 * width
    bordered = np.zeros((new_H, new_W, 3), dtype=np.uint8)
    bordered[:width, :, :] = color
    bordered[-width:, :, :] = color
    bordered[:, :width, :] = color
    bordered[:, -width:, :] = color
    bordered[width:width+H, width:width+W, :] = image_rgb
    return bordered


def create_composite_page_fixed_4col(exp_frame, strat_frame, low_dwell_frame, full_gt_frame):
    H, W = exp_frame.shape
    def to_rgb(frame):
        return np.stack([frame, frame, frame], axis=-1) if frame.ndim == 2 else frame.copy()
    exp_rgb = to_rgb(exp_frame)
    strat_rgb = to_rgb(strat_frame)
    low_dwell_bordered = add_colored_border(to_rgb(low_dwell_frame), GT_BORDER_COLORS["noise_equivalent"])
    full_gt_bordered = add_colored_border(to_rgb(full_gt_frame), GT_BORDER_COLORS["full"])
    bordered_H, bordered_W = low_dwell_bordered.shape[:2]
    output_W = 2 * W + 2 * bordered_W
    output_H = max(H, bordered_H)
    output = np.zeros((output_H, output_W, 3), dtype=np.uint8)
    output[0:H, 0:W, :] = exp_rgb
    output[0:H, W:2*W, :] = strat_rgb
    output[0:bordered_H, 2*W:2*W + bordered_W, :] = low_dwell_bordered
    output[0:bordered_H, 2*W + bordered_W:2*W + 2*bordered_W, :] = full_gt_bordered
    return output


def create_result_graphic_comparison(config, result_type="reconstruction", 
                                      base_dir="plots", output_path=None):
    required_keys = {"sampler_type", "interpol_method", "scanned_pixel_percent", "gt_name",
                     "has_temporal_sampler", "has_temporal_reconstruction",
                     "alpha", "adaptive_fraction", "temporal_method", 
                     "temporal_residual_cutoff", "temporal_residual_confidence_scale", "sample_sequence"}
    missing = required_keys - set(config.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    gt_name = config["gt_name"]
    sparsity = config["scanned_pixel_percent"]
    interpol_method = config["interpol_method"]
    adaptive_dir = os.path.join(base_dir, "examples", config["sampler_type"],
                                f"interpol_{interpol_method}", f"sparsity_{sparsity}", gt_name,
                                f"sampler_{config['has_temporal_sampler']}_reconstruction_{config['has_temporal_reconstruction']}",
                                f"temporalMethod_{config['temporal_method']}",
                                f"temporalResidualCutoff_{config['temporal_residual_cutoff']}",
                                f"temporalResidualConfidenceScale_{config['temporal_residual_confidence_scale']}",
                                f"sampleSequence_{config['sample_sequence']}",
                                f"alpha_{config['alpha']}", f"adaptive_{config['adaptive_fraction']}")
    stratified_dir = build_stratified_path(gt_name, sparsity, interpol_method, base_dir)
    print(f"Loading adaptive from: {adaptive_dir}")
    print(f"Loading stratified from: {stratified_dir}")
    adaptive_frames = load_frames_from_path(adaptive_dir, result_type)
    stratified_frames = load_frames_from_path(stratified_dir, result_type)
    if adaptive_frames.shape[0] != stratified_frames.shape[0]:
        raise ValueError(f"Frame count mismatch: adaptive={adaptive_frames.shape[0]}, stratified={stratified_frames.shape[0]}")
    T = adaptive_frames.shape[0]
    ref_H, ref_W = adaptive_frames.shape[1], adaptive_frames.shape[2]
    if stratified_frames.shape[1:] != (ref_H, ref_W):
        print(f"Resizing stratified frames from {stratified_frames.shape[1:]} to {ref_H}x{ref_W}")
        resized_strat = []
        for t in range(stratified_frames.shape[0]):
            resized_strat.append(resize_frame(stratified_frames[t], ref_H, ref_W))
        stratified_frames = np.stack(resized_strat, axis=0)
    low_dwell_frames = load_all_low_dwell_frames(gt_name, sparsity, base_dir)
    full_gt_frames = load_all_ground_truth_frames(gt_name)
    for gt_name_suffix, gt_frames in [("low-dwell", low_dwell_frames), ("full", full_gt_frames)]:
        if gt_frames.shape[1:] != (ref_H, ref_W):
            print(f"Resizing {gt_name_suffix} GT from {gt_frames.shape[1:]} to {ref_H}x{ref_W}")
            resized = [resize_frame(gt_frames[t], ref_H, ref_W) for t in range(gt_frames.shape[0])]
            gt_frames = np.stack(resized, axis=0)
        if gt_frames.shape[0] > T:
            gt_frames = gt_frames[:T]
        elif gt_frames.shape[0] < T:
            raise ValueError(f"{gt_name_suffix.capitalize()} GT has {gt_frames.shape[0]} frames, expected {T}")
        if gt_name_suffix == "low-dwell":
            low_dwell_frames = gt_frames
        else:
            full_gt_frames = gt_frames
    if output_path is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(output_dir, f"result_comparison_{result_type}_{gt_name}_sparsity_{sparsity}.tiff")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception as e:
            print(f"Warning: Could not remove existing file: {e}")
    composite_pages = []
    for t in range(T):
        page = create_composite_page_fixed_4col(
            adaptive_frames[t], stratified_frames[t], low_dwell_frames[t], full_gt_frames[t]
        )
        composite_pages.append(page)
        print(f"Created composite page {t}: shape {page.shape}")
    print(f"Total composite pages: {len(composite_pages)}")
    if len(composite_pages) <= 1:
        tifffile.imwrite(output_path, composite_pages[0], photometric='rgb')
    else:
        with tifffile.TiffWriter(output_path, bigtiff=False) as tif:
            for i, page in enumerate(composite_pages):
                tif.write(page, photometric='rgb', description=f'frame {i}')
    return output_path


def parse_best_combined_configs(analysis_file_path="parameter_tune_analysis.txt"):
    """
    Parse the parameter_tune_analysis.txt file to extract best combined configurations.
    
    Returns a list of dicts, each containing the parameters for the best combined
    configuration for each gt_name x sparsity tuple.
    
    Each dict contains:
    - gt_name
    - scanned_pixel_percent
    - alpha
    - adaptive_fraction (adaptiveFraction in file)
    - sample_sequence (sampleSequence in file)
    - temporal_residual_cutoff (temporalResidualCutoff in file)
    - temporal_residual_confidence_scale (temporalResidualConfidenceScale in file)
    - temporal_method (temporalMethod in file)
    - score
    """
    configs = []
    
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
            # Format: '--- scanned_pixel_percent = 0.1 ---'
            inner = stripped[3:-3].strip()  # Remove outer ---
            parts = inner.split('=')
            if len(parts) >= 2:
                current_sparsity = float(parts[1].strip())
            i += 1
            continue
        
        # Check for Best Combined line
        if 'Best Combined' in stripped:
            score = None
            
            # Extract score from current line
            score_start = stripped.find('score=')
            if score_start != -1:
                score_end = stripped.find(')', score_start)
                if score_end != -1:
                    score_str = stripped[score_start + 6:score_end]
                    try:
                        score = int(score_str)
                    except ValueError:
                        pass
            
            # Find the config line (next non-empty line)
            j = i + 1
            best_combined_line = None
            while j < len(lines):
                next_line = lines[j].strip()
                if next_line and next_line.startswith('withTemporalSampler'):
                    best_combined_line = next_line
                    break
                j += 1
            
            if best_combined_line and current_gt and current_sparsity is not None:
                # Parse the configuration
                config = {}
                config['gt_name'] = current_gt
                config['scanned_pixel_percent'] = current_sparsity
                config['combined_score'] = score
                
                # Extract key-value pairs from the config line
                parts = [p.strip() for p in best_combined_line.split(',')]
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
                        if key == 'adaptiveFraction':
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
                
                # Add fixed parameters for adaptive sampler
                config['sampler_type'] = 'adaptive'
                config['interpol_method'] = 'cubic'
                config['has_temporal_sampler'] = True
                config['has_temporal_reconstruction'] = True
                
                configs.append(config)
            
            i = j + 1
            continue
        
        i += 1
    
    return configs


def create_best_config_comparisons(
    analysis_file_path="parameter_tune_analysis.txt",
    result_type="reconstruction",
    base_dir="plots",
    output_dir=None
):
    """
    Create result comparison TIFFs for each gt_name x sparsity tuple using the best
    combined configuration from the parameter tuning analysis.
    
    The output filename includes: GT, sparsity, alpha, adaptiveFraction, sample_sequence,
    temporalResidualCutoff, and temporalResidualConfidenceScale for identification.
    
    Args:
        analysis_file_path: Path to the parameter_tune_analysis.txt file
        result_type: Debug image kind to compare (default: "reconstruction")
        base_dir: Base directory for experiment results (default: "plots")
        output_dir: Directory to save output TIFFs (default: same as script dir)
        
    Returns:
        List of paths to the generated comparison TIFFs
    """
    # Parse the best configurations
    best_configs = parse_best_combined_configs(analysis_file_path)
    
    print(f"Found {len(best_configs)} best configurations to process")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = []
    
    for config in best_configs:
        gt_name = config['gt_name']
        sparsity = config['scanned_pixel_percent']
        alpha = config['alpha']
        adaptive_fraction = config['adaptive_fraction']
        sample_sequence = config['sample_sequence']
        temporal_cutoff = config['temporal_residual_cutoff']
        temporal_scale = config['temporal_residual_confidence_scale']
        
        # Build output filename with all identifying parameters
        # Format: result_comparison_best_{gt_name}_s{sparsity}_a{alpha}_af{adaptive_fraction}_seq{sample_sequence}_method{temporal_method}_cutoff{temporal_cutoff}_scale{temporal_scale}.tiff
        temporal_method = config.get('temporal_method', 'unknown')
        filename = (
            f"result_comparison_best_{gt_name}_s{sparsity}_"
            f"a{alpha}_af{adaptive_fraction}_seq{sample_sequence}_"
            f"method{temporal_method}_cutoff{temporal_cutoff}_scale{temporal_scale}.tiff"
        )
        
        # Sanitize filename (replace spaces and special chars)
        filename = filename.replace(" ", "_")
        output_path = os.path.join(output_dir, filename)
        
        print(f"\nProcessing {gt_name} at sparsity {sparsity}...")
        print(f"  Config: alpha={alpha}, adaptive_fraction={adaptive_fraction}, "
              f"sample_sequence={sample_sequence}, temporal_cutoff={temporal_cutoff}, "
              f"temporal_scale={temporal_scale}")
        
        try:
            result_path = create_result_graphic_comparison(
                config=config,
                result_type=result_type,
                base_dir=base_dir,
                output_path=output_path
            )
            generated_files.append(result_path)
            print(f"  Saved to: {result_path}")
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  Skipping this configuration")
    
    print(f"\nGenerated {len(generated_files)} comparison TIFFs out of {len(best_configs)} configs")
    return generated_files


# Example usage
if __name__ == '__main__':
    # Example 1: Single config (original example)
    # This matches a configuration from compare_li_expulsion_two_sparsity
    """
    config = {
        "sampler_type": "adaptive",
        "interpol_method": "cubic",
        "scanned_pixel_percent": 0.5,
        "gt_name": "LI_EXPULSION_TWO_ORIGINAL",
        "has_temporal_sampler": True,
        "has_temporal_reconstruction": True,
        "minDensityGamma": 0.1,
        "alpha": 5.0,
        "adaptive_fraction": 0.5,
        "temporal_method": "temporal_variance",
        "temporal_residual_cutoff": 12.0,
        "temporal_residual_confidence_scale": 250.0,
        "sample_sequence": "halton"
    }
    output_path = create_result_graphic_comparison(
        config=config, result_type="reconstruction",
        base_dir="plots", output_path="result_comparison_example.tiff"
    )
    print(f"Composite TIF saved to: {output_path}")
"""
    # Example 2: Process all best configs from parameter tuning analysis
    # This generates comparison TIFFs for each gt_name x sparsity tuple
    # using the best combined configuration
    generated = create_best_config_comparisons(
        analysis_file_path="statistics/parameter_tune_analysis.txt",
        result_type="reconstruction",
        base_dir="plots",
        output_dir="best_config_comparisons"
    )
    print(f"Generated {len(generated)} comparison TIFFs")

