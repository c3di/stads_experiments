#!/usr/bin/env python3
"""
TIF Stack to MP4 Video Converter using ffmpeg directly.
This ensures maximum compatibility with PowerPoint by using very specific encoding parameters.

Usage:
    python tif_to_video_ffmpeg.py file1.tif file2.tif ...
    python tif_to_video_ffmpeg.py file1.tif -n "suffix"
"""

import argparse
import os
import sys
import subprocess
import tempfile
from pathlib import Path

try:
    import tifffile
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Please install with: pip install tifffile numpy")
    sys.exit(1)


def find_ffmpeg():
    """Find ffmpeg executable."""
    # Try system ffmpeg first (usually has better codec support)
    candidates = [
        '/c/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg',
        'C:/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe',
        'C:/ffmpeg/bin/ffmpeg.exe',
        'C:/Program Files/ffmpeg/bin/ffmpeg.exe',
        'ffmpeg',
    ]
    
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    
    # Fall back to imageio-ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except:
        pass
    
    return 'ffmpeg'  # Hope it's in PATH


def convert_tif_to_mp4_ffmpeg(tif_path, output_path, fps=10, diagnostics=False):
    """
    Convert TIF stack to MP4 using ffmpeg directly with PowerPoint-compatible settings.
    
    PowerPoint-compatible settings:
    - H.264 High Profile, Level 4.0
    - YUV 4:2:0 pixel format
    - No audio track
    - MP4 container
    """
    print(f"Reading {tif_path}...")
    
    # Read all frames into memory
    with tifffile.TiffFile(tif_path) as tif:
        frames = [page.asarray() for page in tif.pages]
    
    if len(frames) == 0:
        raise ValueError("No frames found in TIF file")
    
    first_frame = frames[0]
    height, width = first_frame.shape[:2]
    is_grayscale = len(first_frame.shape) == 2
    
    print(f"  Found {len(frames)} frames of size {width}x{height}")
    print(f"  Type: {'Grayscale' if is_grayscale else 'RGB'}")
    
    # Pad dimensions to be divisible by 16 (required for H.264)
    pad_w = (16 - width % 16) % 16
    pad_h = (16 - height % 16) % 16
    new_width = width + pad_w
    new_height = height + pad_h
    
    if pad_w > 0 or pad_h > 0:
        print(f"  Padding from {width}x{height} to {new_width}x{new_height}")
    
    # Create a temporary directory for raw frames
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_files = []
        
        # Write each frame as PNG to temp directory
        for i, frame in enumerate(frames):
            # Process frame
            if is_grayscale:
                # Convert grayscale to RGB
                frame = np.stack([frame]*3, axis=-1)
            elif len(frame.shape) == 3 and frame.shape[2] == 1:
                frame = np.concatenate([frame]*3, axis=-1)
            
            # Ensure uint8
            if frame.dtype != np.uint8:
                if frame.dtype == np.float32 or frame.dtype == np.float64:
                    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
                elif frame.dtype == np.uint16:
                    frame = (frame / 256).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # Pad frame
            if pad_w > 0 or pad_h > 0:
                frame = np.pad(frame, 
                              ((0, pad_h), (0, pad_w), (0, 0)), 
                              mode='constant', 
                              constant_values=0)
            
            frame_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
            # Use PIL to save as PNG (simpler than imageio)
            try:
                from PIL import Image
                img = Image.fromarray(frame)
                img.save(frame_path)
            except:
                # Fallback to imageio
                import imageio
                imageio.imwrite(frame_path, frame)
            
            frame_files.append(frame_path)
        
        # Build ffmpeg command with PowerPoint-compatible settings
        ffmpeg_path = find_ffmpeg()
        
        # ffmpeg command:
        # -y: overwrite output
        # -framerate: input frame rate
        # -i frame_%04d.png: input pattern
        # -c:v libx264: H.264 encoder
        # -profile:v high: High profile
        # -level 4.0: Level 4.0 (widely compatible)
        # -pix_fmt yuv420p: YUV 4:2:0 pixel format (required for max compatibility)
        # -preset fast: good balance of speed and compression
        # -crf 18: good quality
        # -an: no audio
        # -movflags +faststart: enable streaming
        
        cmd = [
            ffmpeg_path,
            '-y',
            '-framerate', str(fps),
            '-i', os.path.join(tmpdir, 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-profile:v', 'high',
            '-level', '4.0',
            '-pix_fmt', 'yuv420p',
            '-preset', 'fast',
            '-crf', '18',
            '-an',
            '-movflags', '+faststart',
            output_path
        ]
        
        print(f"Writing {output_path}...")
        print(f"  FFmpeg: {ffmpeg_path}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  FFmpeg stderr: {result.stderr}")
            # Try without level specification
            print("  Retrying without level specification...")
            cmd2 = [
                ffmpeg_path,
                '-y',
                '-framerate', str(fps),
                '-i', os.path.join(tmpdir, 'frame_%04d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'fast',
                '-crf', '18',
                '-an',
                '-movflags', '+faststart',
                output_path
            ]
            result = subprocess.run(cmd2, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  FFmpeg stderr (2nd attempt): {result.stderr}")
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        print(f"  Created: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert TIF stacks to MP4 videos using ffmpeg directly.',
        epilog="""
Examples:
  python tif_to_video_ffmpeg.py sample1.tif sample2.tif
  python tif_to_video_ffmpeg.py sample3.tif -n "original"
"""
    )
    
    parser.add_argument('tif_files', nargs='+', help='Input .tif files')
    parser.add_argument('-n', '--name-suffix', type=str, default=None, help='Suffix for output filename')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second (default: 10)')
    parser.add_argument('--diagnostics', action='store_true', default=False, help='Print diagnostics')
    
    args = parser.parse_args()
    
    for tif_path in args.tif_files:
        if not os.path.exists(tif_path):
            print(f"Error: File not found: {tif_path}")
            continue
        
        is_tif = tif_path.lower().endswith(('.tif', '.tiff'))
        if not is_tif:
            print(f"Error: Not a TIF/TIFF file: {tif_path}")
            continue
        
        base_name = Path(tif_path).stem
        parent_dir = Path(tif_path).parent
        
        output_stem = f"{base_name}_{args.name_suffix}" if args.name_suffix else base_name
        output_path = parent_dir / f"{output_stem}.mp4"
        
        try:
            convert_tif_to_mp4_ffmpeg(tif_path, str(output_path), 
                                      fps=args.fps,
                                      diagnostics=args.diagnostics)
        except Exception as e:
            print(f"Failed to convert {tif_path}: {e}")
            continue
    
    print("\nAll conversions complete!")


if __name__ == '__main__':
    main()
