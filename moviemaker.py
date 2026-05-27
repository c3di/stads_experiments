#!/usr/bin/env python3
"""
moviemaker.py – Turn a folder of .tif / .png frames into a GIF.

Usage
-----
    python moviemaker.py <folder> [--fps FPS] [--output OUTPUT]

The files are sorted by the last number found in their filename (ascending).
All frames are normalised together so relative intensities are preserved.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def extract_number(stem: str) -> int:
    """Return the last integer found in *stem*, or 0 if there is none."""
    nums = re.findall(r'\d+', stem)
    return int(nums[-1]) if nums else 0


def to_uint8(arr: np.ndarray, global_min: float, global_max: float) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = (arr - global_min) / (global_max - global_min + 1e-8) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def load_frame(path: Path) -> np.ndarray:
    if path.suffix.lower() in ('.tif', '.tiff'):
        return tifffile.imread(str(path))
    return np.asarray(Image.open(path))


def to_pil_rgb(arr: np.ndarray) -> Image.Image:
    """Convert any 2-D or 3-D uint8 array to a PIL RGB image."""
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return Image.fromarray(arr, mode='RGB')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Create a GIF movie from a folder of .tif or .png frames.'
    )
    parser.add_argument('folder', help='Path to the folder containing the frame files.')
    parser.add_argument('--fps', type=float, default=5.0,
                        help='Frames per second (default: 5).')
    parser.add_argument('--output', default=None,
                        help='Output GIF path (default: <folder>/movie.gif).')
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f'Not a directory: {folder}')

    tif_files = list(folder.glob('*.tif')) + list(folder.glob('*.tiff'))
    png_files = list(folder.glob('*.png'))
    files = tif_files or png_files

    if not files:
        raise SystemExit(f'No .tif/.tiff or .png files found in {folder}')

    files.sort(key=lambda p: extract_number(p.stem))
    print(f'Found {len(files)} frames:')
    for f in files:
        print(f'  {f.name}')

    raw_frames = [load_frame(f) for f in files]

    # Global min/max normalisation so relative intensities are preserved
    # across all frames rather than each frame being stretched independently.
    g_min = float(min(f.min() for f in raw_frames))
    g_max = float(max(f.max() for f in raw_frames))
    pil_frames = [to_pil_rgb(to_uint8(f, g_min, g_max)) for f in raw_frames]

    output = Path(args.output) if args.output else folder / 'movie.gif'
    duration_ms = int(round(1000.0 / args.fps))

    pil_frames[0].save(
        output,
        format='GIF',
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,               # 0 = loop forever (required for motion in PowerPoint)
        duration=duration_ms,
        optimize=False,
    )

    print(f'\nSaved → {output}  ({len(pil_frames)} frames @ {args.fps} fps, {duration_ms} ms/frame)')


if __name__ == '__main__':
    main()
