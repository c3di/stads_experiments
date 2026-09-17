#!/usr/bin/env python3
"""
TIF Stack to GIF Converter

Converts .tif stacks to .gif animations for embedding in presentations.
GIF format is universally compatible with PowerPoint and other applications.

Usage:
    python tif_to_gif.py file1.tif file2.tif ...
    python tif_to_gif.py file1.tif -n "suffix"
    python tif_to_gif.py file1.tif file2.tif -n "processed"

Examples:
    python tif_to_gif.py sample1.tif sample2.tif
        -> Creates: sample1.gif, sample2.gif
    
    python tif_to_gif.py sample3.tif -n "original"
        -> Creates: sample3_original.gif

Dependencies:
    - tifffile
    - Pillow (PIL)
    - numpy
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from PIL import Image
    import tifffile
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Please install with: pip install tifffile Pillow numpy")
    sys.exit(1)


def convert_tif_to_gif(tif_path, output_path, fps=10, duration=None, 
                       invert=False, gamma=1.0, diagnostics=False):
    """
    Convert a .tif stack to a .gif animation.
    
    Args:
        tif_path: Path to input .tif file
        output_path: Path to output .gif file
        fps: Frames per second for the output animation
        duration: Duration per frame in milliseconds (overrides fps if provided)
        invert: Whether to invert the color map
        gamma: Gamma correction value
        diagnostics: Print detailed diagnostics
    """
    print(f"Reading {tif_path}...")
    
    # Read the TIF stack
    with tifffile.TiffFile(tif_path) as tif:
        frames = [page.asarray() for page in tif.pages]
    
    if len(frames) == 0:
        raise ValueError("No frames found in TIF file")
    
    first_frame = frames[0]
    height, width = first_frame.shape[:2]
    is_grayscale = len(first_frame.shape) == 2
    
    if diagnostics:
        print(f"  Total frames: {len(frames)}")
        print(f"  Frame size: {width}x{height}")
        print(f"  Type: {'Grayscale' if is_grayscale else 'RGB'}")
        for i, frame in enumerate(frames[:3]):
            print(f"  Frame {i}: dtype={frame.dtype}, min={np.min(frame):.2f}, "
                  f"max={np.max(frame):.2f}, mean={np.mean(frame):.2f}")
    
    print(f"  Found {len(frames)} frames of size {width}x{height}")
    
    # Process frames
    pil_frames = []
    for frame in frames:
        frame = frame.astype(np.float32)
        
        # Normalize based on dtype
        if frame.dtype == np.float32:
            frame_min = float(np.min(frame))
            frame_max = float(np.max(frame))
            if frame_max > frame_min:
                frame = (frame - frame_min) / (frame_max - frame_min)
            else:
                frame = np.full_like(frame, 0.5)
        elif frame.dtype == np.uint16:
            frame = frame / 65535.0
        elif frame.dtype == np.uint8:
            frame = frame / 255.0
        elif frame.dtype in (np.int16, np.int32):
            frame_min = float(np.min(frame))
            frame_max = float(np.max(frame))
            if frame_max > frame_min:
                frame = (frame - frame_min) / (frame_max - frame_min)
            else:
                frame = np.full_like(frame, 0.5)
        
        # Apply gamma correction
        if gamma != 1.0:
            frame = np.clip(frame, 1e-6, 1.0)
            frame = np.power(frame, 1.0 / gamma)
        
        # Apply inversion if requested
        if invert:
            frame = 1.0 - frame
        
        # Convert to uint8 (0-255)
        frame = (frame * 255.0).astype(np.uint8)
        
        # Convert to RGB if grayscale
        if is_grayscale:
            # Stack grayscale to RGB
            rgb_frame = np.stack([frame]*3, axis=-1)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            rgb_frame = np.concatenate([frame]*3, axis=-1)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            # RGBA - drop alpha
            rgb_frame = frame[..., :3]
        else:
            rgb_frame = frame
        
        pil_frames.append(Image.fromarray(rgb_frame))
    
    # Calculate duration per frame
    if duration is None:
        # GIF duration is in milliseconds, 100 = 10 fps
        duration_per_frame = int(1000 / fps) if fps > 0 else 100
    else:
        duration_per_frame = duration
    
    print(f"  Writing {output_path} (frame duration: {duration_per_frame}ms)...")
    
    # Save as GIF
    # GIF has a limit of 256 colors, so we need to optimize
    # For scientific images, we can try to preserve quality
    if len(pil_frames) == 1:
        # Single frame - just save as static GIF
        pil_frames[0].save(output_path)
    else:
        # Multiple frames - save as animated GIF
        # Use the first frame as the base and append others
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_per_frame,
            loop=0,  # Loop forever
            optimize=True,
            # Use quantization to reduce colors for better GIF quality
            # GIF only supports 256 colors total
            quantizer=None  # Let PIL handle it
        )
    
    print(f"  Created: {output_path}")
    
    # Print file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {size_mb:.2f} MB")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Convert TIF stacks to GIF animations for presentations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tif_to_gif.py sample1.tif sample2.tif
      -> Creates: sample1.gif, sample2.gif
  
  python tif_to_gif.py sample3.tif -n "original"
      -> Creates: sample3_original.gif
  
  python tif_to_gif.py *.tif --fps 5
      -> Creates GIFs with 5 frames per second
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
        help='Frames per second for output animation (default: 10)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Duration per frame in milliseconds (overrides fps)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Optional output directory (default: same as input)'
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
        
        output_path = output_dir / f"{output_stem}.gif"
        
        # Convert the TIF to GIF
        try:
            convert_tif_to_gif(tif_path, str(output_path),
                               fps=args.fps,
                               duration=args.duration,
                               invert=args.invert,
                               gamma=args.gamma,
                               diagnostics=args.diagnostics)
        except Exception as e:
            print(f"Failed to convert {tif_path}: {e}")
            continue
    
    print("\nAll conversions complete!")


if __name__ == '__main__':
    main()
