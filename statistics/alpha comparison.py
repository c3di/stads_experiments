import re
import numpy as np
import tifffile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── INPUTS ─────────────────────────────────────────────────────────────────────
BASE_DIR         = Path("D:\\stads_experiments\\plots\\examples\\adaptive")
SPARSITY_LEVELS  = [0.5, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0]  # list of sparsity levels to process
GROUND_TRUTH_NAME = "hydration_one"  # name of the ground truth dataset (subfolder in each sparsity folder)
ALPHA_LEVELS     = [0.25, 0.5, 2.0] 
MODE_FOLDER      = "sampler_True_reconstruction_True"
SHOW_LEGEND      = True          # set False to omit legend bars
SHOW_COMPARISON  = True         # add a 'no temp' column from reconstruction_False mode
COMPARISON_MODE_FOLDER = "sampler_True_reconstruction_False"
INCLUDE_ERRORS   = True         # add a third row with *_error_map.tiff images
# ───────────────────────────────────────────────────────────────────────────────


def _load_as_rgb_uint8(path: Path, fallback_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Return any image file as (H, W, 3) uint8 RGB array.
    If the file is missing, return a black image of fallback_shape (H, W) or (1, 1) as last resort."""
    if not path.exists():
        h, w = fallback_shape if fallback_shape else (1, 1)
        return np.zeros((h, w, 3), dtype=np.uint8)
    ext = path.suffix.lower()
    if ext in (".tiff", ".tif"):
        arr = tifffile.imread(str(path)).astype(np.float32)
        lo, hi = arr.min(), arr.max()
        arr = ((arr - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros_like(arr, dtype=np.uint8)
        if arr.ndim == 2:                          # greyscale → RGB
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:  # RGBA → RGB
            arr = arr[:, :, :3]
    else:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return arr


def _sorted_frames(alpha_dir: Path) -> list[int]:
    """Return frame numbers present in an alpha folder, sorted ascending."""
    pat = re.compile(r"^frame_(\d+)_reconstructed\.tiff$")
    frames = [int(m.group(1)) for f in alpha_dir.iterdir() if (m := pat.match(f.name))]
    return sorted(frames)


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a PIL font, falling back to the built-in bitmap font if no TTF is found."""
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _add_legend(grid: np.ndarray, col_labels: list[str], row_labels: list[str], title: str = "", comp_col_idx: int | None = None) -> np.ndarray:
    """Add a column-header bar and a row-label strip (rotated text).

    Parameters
    ----------
    grid:          (2*H, N*W, 3) uint8 grid image.
    col_labels:    list of N label strings, one per column (e.g. "α = 1.0" or "no temp").
    row_labels:    list of 2 row label strings.
    title:         optional headline shown above the column labels.
    comp_col_idx:  index of the comparison column; gets a yellow outline and label colour.

    Returns
    -------
    (2*H + header_h, N*W + row_w, 3) uint8 image.
    """
    total_h, total_w = grid.shape[:2]
    n_cols = len(col_labels)
    cell_w = total_w // n_cols
    cell_h = total_h // len(row_labels)

    font_size       = max(12, cell_h // 12)
    title_font_size = max(14, int(font_size * 1.25))
    font       = _get_font(font_size)
    title_font = _get_font(title_font_size)

    alpha_line_h = max(28, font_size + 8)
    title_line_h = (max(28, title_font_size + 8)) if title else 0
    header_h = alpha_line_h + title_line_h
    row_w    = max(32, cell_h // 6)

    new_h = total_h + header_h
    new_w = total_w + row_w

    canvas = Image.new("RGB", (new_w, new_h), color=(40, 40, 40))
    canvas.paste(Image.fromarray(grid), (row_w, header_h))
    draw = ImageDraw.Draw(canvas)

    # ── headline title ────────────────────────────────────────────────────────
    if title:
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x_centre = row_w + total_w // 2
        draw.text((x_centre - tw // 2, (title_line_h - th) // 2), title, fill=(255, 220, 80), font=title_font)

    # ── column headers ────────────────────────────────────────────────────────
    for i, label in enumerate(col_labels):
        x_centre = row_w + i * cell_w + cell_w // 2
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        y_alpha = title_line_h + (alpha_line_h - th) // 2
        fill = (255, 220, 0) if i == comp_col_idx else (220, 220, 220)
        draw.text((x_centre - tw // 2, y_alpha), label, fill=fill, font=font)

    # ── row labels (rotated 90°) ──────────────────────────────────────────────
    for r, label in enumerate(row_labels):
        y_centre = header_h + r * cell_h + cell_h // 2
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tmp = Image.new("RGB", (tw + 4, th + 4), color=(40, 40, 40))
        ImageDraw.Draw(tmp).text((2, 2), label, fill=(220, 220, 220), font=font)
        tmp = tmp.rotate(90, expand=True)
        rw, rh = tmp.size
        paste_x = (row_w - rw) // 2
        paste_y = y_centre - rh // 2
        canvas.paste(tmp, (paste_x, paste_y))

    # ── yellow outline around comparison column ───────────────────────────────
    if comp_col_idx is not None:
        ox0 = row_w + comp_col_idx * cell_w
        oy0 = header_h
        ox1 = row_w + (comp_col_idx + 1) * cell_w - 1
        oy1 = new_h - 1
        for t in range(3):   # 3-pixel thick outline
            draw.rectangle([ox0 + t, oy0 + t, ox1 - t, oy1 - t], outline=(255, 220, 0))

    return np.asarray(canvas, dtype=np.uint8)


def _find_any_alpha_dir(mode_dir: Path, preferred_alphas: list[float]) -> Path | None:
    """Return first available alpha subfolder, trying preferred_alphas first then scanning."""
    for a in preferred_alphas:
        d = mode_dir / f"alpha_{a}"
        if d.exists():
            return d
    for d in sorted(mode_dir.iterdir()):
        if d.is_dir() and d.name.startswith("alpha_"):
            return d
    return None


def _build_slice(frame: int, mode_dir: Path, alphas: list[float], comp_alpha_dir: Path | None = None, include_errors: bool = False) -> np.ndarray:
    """Return a 2- or 3-row × N-col grid for one frame as (R*H, N*W, 3) uint8."""
    # Determine a reference (H, W) from the first image that actually exists.
    ref_shape: tuple[int, int] | None = None
    for alpha in alphas:
        d = mode_dir / f"alpha_{alpha}"
        for name in (f"frame_{frame}_reconstructed.tiff", f"frame_{frame}_all_points.png"):
            p = d / name
            if p.exists():
                img = _load_as_rgb_uint8(p)
                ref_shape = img.shape[:2]
                break
        if ref_shape:
            break
    if ref_shape is None:
        ref_shape = (1, 1)

    row_rec, row_pts, row_err = [], [], []
    for alpha in alphas:
        d = mode_dir / f"alpha_{alpha}"
        h, w = ref_shape
        rec = _load_as_rgb_uint8(d / f"frame_{frame}_reconstructed.tiff", fallback_shape=ref_shape)
        if rec.shape[:2] != (h, w):
            rec = np.asarray(Image.fromarray(rec).resize((w, h)), dtype=np.uint8)
        pts = _load_as_rgb_uint8(d / f"frame_{frame}_all_points.png", fallback_shape=ref_shape)
        if pts.shape[:2] != (h, w):
            pts = np.asarray(Image.fromarray(pts).resize((w, h)), dtype=np.uint8)
        row_rec.append(rec)
        row_pts.append(pts)
        if include_errors:
            err = _load_as_rgb_uint8(d / f"frame_{frame}_error_map.tiff", fallback_shape=ref_shape)
            if err.shape[:2] != (h, w):
                err = np.asarray(Image.fromarray(err).resize((w, h)), dtype=np.uint8)
            row_err.append(err)

    if comp_alpha_dir is not None:
        h, w = ref_shape
        rec_c = _load_as_rgb_uint8(comp_alpha_dir / f"frame_{frame}_reconstructed.tiff", fallback_shape=ref_shape)
        if rec_c.shape[:2] != (h, w):
            rec_c = np.asarray(Image.fromarray(rec_c).resize((w, h)), dtype=np.uint8)
        pts_c = _load_as_rgb_uint8(comp_alpha_dir / f"frame_{frame}_all_points.png", fallback_shape=ref_shape)
        if pts_c.shape[:2] != (h, w):
            pts_c = np.asarray(Image.fromarray(pts_c).resize((w, h)), dtype=np.uint8)
        row_rec.append(rec_c)
        row_pts.append(pts_c)
        if include_errors:
            err_c = _load_as_rgb_uint8(comp_alpha_dir / f"frame_{frame}_error_map.tiff", fallback_shape=ref_shape)
            if err_c.shape[:2] != (h, w):
                err_c = np.asarray(Image.fromarray(err_c).resize((w, h)), dtype=np.uint8)
            row_err.append(err_c)

    rows = [np.concatenate(row_rec, axis=1), np.concatenate(row_pts, axis=1)]
    if include_errors:
        rows.append(np.concatenate(row_err, axis=1))
    return np.concatenate(rows, axis=0)


def create_alpha_comparison_stack(sparsity: float, gt_name: str, alphas: list[float]) -> None:
    mode_dir = BASE_DIR / f"sparsity_{sparsity}" / gt_name / MODE_FOLDER

    if not mode_dir.exists():
        print(f"[SKIP] not found: {mode_dir}")
        return

    missing = [a for a in alphas if not (mode_dir / f"alpha_{a}").exists()]
    if missing:
        print(f"[SKIP] sparsity={sparsity}: missing alpha folders for {missing}")
        return

    frames = _sorted_frames(mode_dir / f"alpha_{alphas[0]}")
    if not frames:
        print(f"[SKIP] sparsity={sparsity}: no frames found")
        return

    # Resolve comparison column
    comp_alpha_dir = None
    comp_col_idx = None
    if SHOW_COMPARISON:
        comp_mode_dir = BASE_DIR / f"sparsity_{sparsity}" / gt_name / COMPARISON_MODE_FOLDER
        if comp_mode_dir.exists():
            comp_alpha_dir = _find_any_alpha_dir(comp_mode_dir, alphas)
        if comp_alpha_dir is None:
            print(f"[WARN] sparsity={sparsity}: comparison dir not found, comparison column will be black")
        comp_col_idx = len(alphas)   # appended as last column

    col_labels = [f"α = {a}" for a in alphas]
    if SHOW_COMPARISON:
        col_labels.append("no temp")

    row_labels = ["reconstruction", "point selection"]
    if INCLUDE_ERRORS:
        row_labels.append("error map")

    out_path = mode_dir.parent / "alpha_comparison_stack.tiff"
    with tifffile.TiffWriter(str(out_path), bigtiff=False) as tif:
        for frame in frames:
            slc = _build_slice(frame, mode_dir, alphas, comp_alpha_dir=comp_alpha_dir, include_errors=INCLUDE_ERRORS)
            if SHOW_LEGEND:
                title = f"sparsity {sparsity}% – {gt_name}"
                slc = _add_legend(slc, col_labels, row_labels, title=title, comp_col_idx=comp_col_idx)
            tif.write(
                slc,
                photometric="rgb",
                description=f"alpha comparison slice frame {frame}",
                metadata=None,
            )

    print(f"[OK]   sparsity={sparsity}: {len(frames)} slices → {out_path}")


if __name__ == "__main__":
    for sparsity in SPARSITY_LEVELS:
        create_alpha_comparison_stack(sparsity, GROUND_TRUTH_NAME, ALPHA_LEVELS)
