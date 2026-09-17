#!/usr/bin/env python3
"""
TIF Stack to MP4 Video Converter using OpenCV

Converts .tif stacks to .mp4 videos for embedding in presentations.
Uses OpenCV which often handles color conversion better than imageio.

Usage:
    python tif_to_video_cv.py file1.tif file2.tif ...
    python tif_to_video_cv.py file1.tif -n "suffix"
    python tif_to_video_cv.py file1.tif file2.tif -n "processed"

Examples:
    python tif_to_video_cv.py sample1.tif sample2.tif
        -> Creates: sample1.mp4, sample2.mp4
    
    python tif_to_video_cv.py sample3.tif -n "original"
        -> Creates: sample3_original.mp4

Dependencies:
    - tifffile
    - opencv-python
    - numpy
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import cv2
    import tifffile
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Please install with: pip install tifffile opencv-python numpy")
    sys.exit(1)


def convert_tif_to_mp4_cv(tif_path, output_path, fps=10, 
                         invert=False, gamma=1.0, diagnostics=False):
    """
    Convert a .tif stack to an .mp4 video using OpenCV.
    
    Args:
        tif_path: Path to input .tif file
        output_path: Path to output .mp4 file
        fps: Frames per second for the output video
        invert: Whether to invert the color map
        gamma: Gamma correction value
        diagnostics: Print detailed diagnostics
    """
    print(f"Reading {tif_path}...")
    
    # Read the TIF stack
    try:
        with tifffile.TiffFile(tif_path) as tif:
            frames = []
            for page in tif.pages:
                frame = page.asarray()
                frames.append(frame)
        
        if diagnostics:
            print(f"  Total frames: {len(frames)}")
            for i, frame in enumerate(frames[:3]):  # Show first 3 frames
                print(f"  Frame {i}: dtype={frame.dtype}, shape={frame.shape}, "
                      f"min={np.min(frame):.2f}, max={np.max(frame):.2f}, "
                      f"mean={np.mean(frame):.2f}")
        
        # Ensure all frames have the same shape
        if len(frames) == 0:
            raise ValueError("No frames found in TIF file")
        
        # Get the shape from the first frame
        first_frame = frames[0]
        height, width = first_frame.shape[:2]
        
        # Process each frame
        processed_frames = []
        for i, frame in enumerate(frames):
            original_dtype = frame.dtype
            
            # Handle normalization based on original dtype
            if original_dtype == np.uint8:
                # Already in 0-255 range, convert to float32 but don't normalize
                frame = frame.astype(np.float32) / 255.0
            elif original_dtype == np.uint16:
                # Scale from 0-65535 to 0-1
                frame = frame.astype(np.float32) / 65535.0
            elif original_dtype == np.float32 or original_dtype == np.float64:
                # Float data: normalize to 0-1 based on actual range
                frame = frame.astype(np.float32)
                frame_min = float(np.min(frame))
                frame_max = float(np.max(frame))
                if frame_max > frame_min:
                    frame = (frame - frame_min) / (frame_max - frame_min)
                else:
                    frame = np.full_like(frame, 0.5, dtype=np.float32)
            else:
                # Other integer types: normalize based on actual range
                frame = frame.astype(np.float32)
                frame_min = float(np.min(frame))
                frame_max = float(np.max(frame))
                if frame_max > frame_min:
                    frame = (frame - frame_min) / (frame_max - frame_min)
                else:
                    frame = np.full_like(frame, 0.5, dtype=np.float32)
            
            # Apply gamma correction
            if gamma != 1.0:
                frame = np.clip(frame, 1e-6, 1.0)
                frame = np.power(frame, 1.0 / gamma)
            
            # Apply inversion if requested
            if invert:
                frame = 1.0 - frame
            
            # Convert to uint8 (0-255)
            frame = (frame * 255.0).astype(np.uint8)
            
            # Ensure frame is 3-channel BGR for OpenCV
            if len(frame.shape) == 2:
                # Grayscale -> BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                # RGB -> BGR (OpenCV uses BGR order)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 1:
                # Single channel -> BGR
                frame = cv2.cvtColor(frame.squeeze(), cv2.COLOR_GRAY2BGR)
            
            processed_frames.append(frame)
        
        print(f"  Found {len(processed_frames)} frames of size {width}x{height}")
        
        # Write as MP4 using OpenCV
        print(f"Writing {output_path}...")
        
        # Get the first frame to determine video size
        first_processed = processed_frames[0]
        height_out, width_out = first_processed.shape[:2]
        
        # Define the codec and create VideoWriter object
        # Try H.264 first (avc1), fall back to mp4v
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width_out, height_out), isColor=True)
            if not out.isOpened():
                # Fall back to mp4v if avc1 doesn't work
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width_out, height_out), isColor=True)
        except:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width_out, height_out), isColor=True)
        
        for frame in processed_frames:
            out.write(frame)
        
        out.release()
        print(f"  Created: {output_path}")
        
    except Exception as e:
        print(f"Error processing {tif_path}: {e}")
        raise


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Convert TIF stacks to MP4 videos using OpenCV.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tif_to_video_cv.py sample1.tif sample2.tif
      -> Creates: sample1.mp4, sample2.mp4
  
  python tif_to_video_cv.py sample3.tif -n "original"
      -> Creates: sample3_original.mp4
  
  python tif_to_video_cv.py *.tif -n "processed"
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
        '--invert',
        action='store_true',
        default=False,
        help='Invert the color map'
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
            convert_tif_to_mp4_cv(tif_path, str(output_path), 
                                fps=args.fps,
                                invert=args.invert,
                                gamma=args.gamma,
                                diagnostics=args.diagnostics)
        except Exception as e:
            print(f"Failed to convert {tif_path}: {e}")
            continue
    
    print("\nAll conversions complete!")


if __name__ == '__main__':
    main()
