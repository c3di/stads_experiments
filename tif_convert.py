#!/usr/bin/env python3
"""
TIF Stack to MP4 Video Converter

Converts .tif stacks to .mp4 videos for embedding in presentations.

Usage:
    python tif_convert.py file1.tif file2.tif ...
    python tif_convert.py file1.tif -n "suffix"
    python tif_convert.py file1.tif file2.tif -n "processed"

Examples:
    python tif_convert.py sample1.tif sample2.tif
        -> Creates: sample1.mp4, sample2.mp4
    
    python tif_convert.py sample3.tif -n "original"
        -> Creates: sample3_original.mp4

Dependencies:
    - tifffile
    - imageio
    - imageio-ffmpeg (for MP4 support)
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import tifffile
    import imageio
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Please install with: pip install tifffile imageio imageio-ffmpeg numpy")
    sys.exit(1)


def normalize_channel(channel, dtype, full_range):
    """
    Normalize a single channel to 0-1 float range.
    
    Args:
        channel: The channel data (2D numpy array)
        dtype: Original dtype of the data
        full_range: Tuple of (min, max) of the original data
    
    Returns:
        Normalized channel as float32 in [0, 1]
    """
    channel = channel.astype(np.float32)
    
    if dtype == np.float32 or dtype == np.float64:
        # Float data: normalize based on actual range
        ch_min = float(np.min(channel))
        ch_max = float(np.max(channel))
        if ch_max > ch_min:
            return (channel - ch_min) / (ch_max - ch_min)
        else:
            return np.full_like(channel, 0.5, dtype=np.float32)
    elif dtype == np.uint16:
        # 16-bit unsigned: scale from 0-65535
        return channel / 65535.0
    elif dtype == np.uint8:
        # 8-bit unsigned: scale from 0-255
        return channel / 255.0
    elif dtype == np.int32 or dtype == np.int16:
        # Signed integers: normalize based on actual range
        ch_min = float(np.min(channel))
        ch_max = float(np.max(channel))
        if ch_max > ch_min:
            return (channel - ch_min) / (ch_max - ch_min)
        else:
            return np.full_like(channel, 0.5, dtype=np.float32)
    else:
        # Fallback: use the provided full range
        ch_min, ch_max = full_range
        if ch_max > ch_min:
            return (channel - ch_min) / (ch_max - ch_min)
        else:
            return np.full_like(channel, 0.5, dtype=np.float32)


def convert_tif_to_mp4(tif_path, output_path, fps=10, macro_block_size=None, 
                      invert=False, gamma=1.0, diagnostics=False):
    """
    Convert a .tif stack to an .mp4 video.
    
    Args:
        tif_path: Path to input .tif file
        output_path: Path to output .mp4 file
        fps: Frames per second for the output video
        macro_block_size: Macro block size for FFMPEG (default: None = auto)
        invert: Whether to invert the color map
        gamma: Gamma correction value
        diagnostics: Print detailed diagnostics
    """
    print(f"Reading {tif_path}...")
    
    # Read the TIF stack
    try:
        # Try reading with tifffile
        with tifffile.TiffFile(tif_path) as tif:
            frames = []
            for page in tif.pages:
                frame = page.asarray()
                frames.append(frame)
        
        if diagnostics:
            print(f"  Total frames: {len(frames)}")
            for i, frame in enumerate(frames[:3]):  # Show first 3 frames
                print(f"  Frame {i}: dtype={frame.dtype}, shape={frame.shape}, "
                      f"min={np.min(frame):.4f}, max={np.max(frame):.4f}, "
                      f"mean={np.mean(frame):.4f}")
            
        # Ensure all frames have the same shape
        if len(frames) == 0:
            raise ValueError("No frames found in TIF file")
            
        # Get the shape from the first frame
        first_frame = frames[0]
        print(f"  First frame dtype: {first_frame.dtype}, shape: {first_frame.shape}")
        
        # Check if RGB or grayscale
        is_grayscale = len(first_frame.shape) == 2
        is_rgb = len(first_frame.shape) == 3 and first_frame.shape[2] in (3, 4)
        is_single_channel = len(first_frame.shape) == 3 and first_frame.shape[2] == 1
        
        if diagnostics:
            if is_grayscale:
                print(f"  Detected: Grayscale (2D)")
            elif is_single_channel:
                print(f"  Detected: Single channel 3D (H,W,1)")
            elif is_rgb:
                print(f"  Detected: RGB/RGBA (H,W,{first_frame.shape[2]})")
            else:
                print(f"  Detected: Unknown format")
        
        # Normalize and convert to uint8
        normalized_frames = []
        for i, frame in enumerate(frames):
            original_dtype = frame.dtype
            original_range = (float(np.min(frame)), float(np.max(frame)))
            
            # For multi-channel images, process each channel separately
            if len(frame.shape) == 3 and frame.shape[2] > 1:
                # RGB or RGBA - process each channel
                processed_channels = []
                for channel in range(frame.shape[2]):
                    ch = frame[..., channel]
                    ch = normalize_channel(ch, original_dtype, original_range)
                    processed_channels.append(ch)
                frame = np.stack(processed_channels, axis=-1)
            else:
                # Grayscale or single channel
                frame = normalize_channel(frame, original_dtype, original_range)
            
            # Apply gamma correction
            if gamma != 1.0:
                frame = np.clip(frame, 1e-6, 1.0)
                frame = np.power(frame, 1.0 / gamma)
            
            # Apply inversion if requested
            if invert:
                frame = 1.0 - frame
            
            # Convert to uint8 (0-255)
            frame = (frame * 255.0).astype(np.uint8)
            
            # Check if frame has very low values
            if np.max(frame) < 10:
                print(f"  Warning: Frame {i} has very low values after conversion. "
                      f"Original dtype: {original_dtype}, range: {original_range}")
            
            normalized_frames.append(frame)
        
        frames = normalized_frames
        
        # Ensure frames are 3-channel RGB for MP4
        first_out_frame = frames[0]
        if len(first_out_frame.shape) == 2:
            # Grayscale (H,W) -> RGB (H,W,3)
            frames = [np.stack([f]*3, axis=-1) for f in frames]
            if diagnostics:
                print(f"  Converted grayscale to RGB")
        elif len(first_out_frame.shape) == 3 and first_out_frame.shape[2] == 1:
            # Single channel (H,W,1) -> RGB (H,W,3)
            frames = [np.concatenate([f]*3, axis=-1) for f in frames]
            if diagnostics:
                print(f"  Converted single channel to RGB")
        elif len(first_out_frame.shape) == 3 and first_out_frame.shape[2] == 4:
            # RGBA -> RGB (drop alpha)
            frames = [f[..., :3] for f in frames]
            if diagnostics:
                print(f"  Converted RGBA to RGB")
        
        height, width = frames[0].shape[:2]
        print(f"  Found {len(frames)} frames of size {width}x{height}")
        
        # Write as MP4
        print(f"Writing {output_path}...")
        
        # Build writer kwargs
        writer_kwargs = {'fps': fps, 'quality': 10}
        if macro_block_size is not None:
            writer_kwargs['macro_block_size'] = macro_block_size
        
        # Try to use libx264 (H.264) codec for better PowerPoint compatibility
        # imageio-ffmpeg uses libx264 by default if available, but we can try to force it
        try:
            with imageio.get_writer(output_path, format='ffmpeg', codec='libx264', **writer_kwargs) as writer:
                for frame in frames:
                    writer.append_data(frame)
        except Exception as e:
            # Fall back to default codec
            print(f"  libx264 not available, falling back to default: {e}")
            with imageio.get_writer(output_path, **writer_kwargs) as writer:
                for frame in frames:
                    writer.append_data(frame)
        
        print(f"  Created: {output_path}")
        
    except Exception as e:
        print(f"Error processing {tif_path}: {e}")
        raise


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Convert TIF stacks to MP4 videos for presentations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tif_convert.py sample1.tif sample2.tif
      -> Creates: sample1.mp4, sample2.mp4
  
  python tif_convert.py sample3.tif -n "original"
      -> Creates: sample3_original.mp4
  
  python tif_convert.py *.tif -n "processed"
      -> Creates: file1_processed.mp4, file2_processed.mp4, ...
"""
    )
    
    parser.add_argument(
        'tif_files',
        nargs='+',
        help='Input .tif files to convert'
    )
    
    parser.add_argument(
        '-n', '--name-suffix',
        type=str,
        default=None,
        help='Optional suffix to append to output filename before extension'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Frames per second for output video (default: 10)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Optional output directory (default: same as input)'
    )
    
    parser.add_argument(
        '--macro-block-size',
        type=int,
        default=None,
        help='Macro block size for FFMPEG (default: auto, typically 16). '
             'Set to 1 to disable resizing warnings (risking incompatibility)'
    )
    
    parser.add_argument(
        '--invert',
        action='store_true',
        default=False,
        help='Invert the color map (useful if images appear as negatives)'
    )
    
    parser.add_argument(
        '--gamma',
        type=float,
        default=1.0,
        help='Gamma correction value (default: 1.0 = no correction)'
    )
    
    parser.add_argument(
        '--diagnostics',
        action='store_true',
        default=False,
        help='Print detailed diagnostics about frame data'
    )
    
    args = parser.parse_args()
    
    # Process each TIF file
    for tif_path in args.tif_files:
        # Validate input file
        if not os.path.exists(tif_path):
            print(f"Error: File not found: {tif_path}")
            continue
            
        # Check if it's a TIF/TIFF file
        is_tif = tif_path.lower().endswith('.tif') or tif_path.lower().endswith('.tiff')
        if not is_tif:
            print(f"Error: Not a TIF/TIFF file: {tif_path}")
            continue
        
        # Construct output filename
        base_name = Path(tif_path).stem
        parent_dir = Path(tif_path).parent
        
        # Apply name suffix if provided
        if args.name_suffix:
            output_stem = f"{base_name}_{args.name_suffix}"
        else:
            output_stem = base_name
        
        # Apply output directory if provided
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = parent_dir
        
        output_path = output_dir / f"{output_stem}.mp4"
        
        # Convert the TIF to MP4
        try:
            convert_tif_to_mp4(tif_path, str(output_path), 
                            fps=args.fps, 
                            macro_block_size=args.macro_block_size,
                            invert=args.invert,
                            gamma=args.gamma,
                            diagnostics=args.diagnostics)
        except Exception as e:
            print(f"Failed to convert {tif_path}: {e}")
            continue
    
    print("\nAll conversions complete!")


if __name__ == '__main__':
    main()
