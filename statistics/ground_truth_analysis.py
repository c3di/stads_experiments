from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.measure import shannon_entropy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


MOVIES_INPUT_PATH = Path.home() / ".stads_data"
OUTPUT_DIR = Path("movie_analysis")


def _compute_data_range(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
	min_val = float(min(np.min(frame_a), np.min(frame_b)))
	max_val = float(max(np.max(frame_a), np.max(frame_b)))
	data_range = max_val - min_val
	return data_range if data_range > 0 else 1.0


def analyze_tif_movie(movie_path: Path) -> tuple[pd.DataFrame, dict[str, float | str | int]]:
	frames = iio.imread(movie_path)
	frames = np.asarray(frames)

	if frames.ndim < 3:
		raise ValueError(f"Expected a stack for movie '{movie_path}', but got array shape {frames.shape}")

	frame_count = int(frames.shape[0])
	frame_rows: list[dict[str, float | str | int]] = []

	entropy_values = [float(shannon_entropy(frame)) for frame in frames]

	for frame_idx in range(frame_count):
		row = {
			"movie_name": movie_path.stem,
			"movie_path": str(movie_path),
			"frame_idx": frame_idx,
			"entropy": entropy_values[frame_idx],
			"psnr_to_prev": np.nan,
			"ssim_to_prev": np.nan,
		}

		if frame_idx > 0:
			prev_frame = np.asarray(frames[frame_idx - 1], dtype=np.float64)
			curr_frame = np.asarray(frames[frame_idx], dtype=np.float64)
			data_range = _compute_data_range(prev_frame, curr_frame)

			row["psnr_to_prev"] = float(
				peak_signal_noise_ratio(prev_frame, curr_frame, data_range=data_range)
			)
			row["ssim_to_prev"] = float(
				structural_similarity(prev_frame, curr_frame, data_range=data_range)
			)

		frame_rows.append(row)

	per_frame_df = pd.DataFrame(frame_rows)

	summary = {
		"movie_name": movie_path.stem,
		"movie_path": str(movie_path),
		"n_frames": frame_count,
		"avg_shannon_entropy": float(np.mean(entropy_values)),
		"avg_psnr_to_prev": float(per_frame_df["psnr_to_prev"].mean(skipna=True)),
		"avg_ssim_to_prev": float(per_frame_df["ssim_to_prev"].mean(skipna=True)),
	}

	return per_frame_df, summary


def generate_framewise_change_plots(per_frame_results_df: pd.DataFrame, output_dir: str | Path) -> None:
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	plot_df = per_frame_results_df.copy()
	plot_df = plot_df.dropna(subset=["frame_idx", "psnr_to_prev", "ssim_to_prev", "movie_name"])
	plot_df["frame_idx"] = pd.to_numeric(plot_df["frame_idx"], errors="coerce")
	plot_df = plot_df.dropna(subset=["frame_idx"])
	plot_df["frame_idx"] = plot_df["frame_idx"].astype(int)

	sns.set_theme(style="whitegrid", context="talk")

	metric_config = [
		("psnr_to_prev", "PSNR to Previous Frame"),
		("ssim_to_prev", "SSIM to Previous Frame"),
	]

	for metric_col, metric_label in metric_config:
		plt.figure(figsize=(12, 6))
		sns.lineplot(
			data=plot_df,
			x="frame_idx",
			y=metric_col,
			hue="movie_name",
			errorbar=None,
		)
		plt.title(f"{metric_label} by Frame Index")
		plt.xlabel("frame_idx")
		plt.ylabel(metric_label)
		plt.legend(title="movie_name", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
		plt.tight_layout()
		plt.savefig(output_dir / f"framewise_{metric_col}_lineplot.png", dpi=300)
		plt.close()


def analyze_movies(input_path: str | Path, output_dir: str | Path = "plots/statistics") -> tuple[pd.DataFrame, pd.DataFrame]:
	input_path = Path(input_path)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	if input_path.is_dir():
		movie_paths = sorted(list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff")))
	else:
		if input_path.suffix.lower() not in {".tif", ".tiff"}:
			raise ValueError(f"Expected a .tif/.tiff file or directory, got: {input_path}")
		movie_paths = [input_path]

	if not movie_paths:
		raise FileNotFoundError(f"No .tif or .tiff movies found at: {input_path}")

	all_per_frame: list[pd.DataFrame] = []
	all_summary: list[dict[str, float | str | int]] = []

	for movie_path in movie_paths:
		per_frame_df, summary = analyze_tif_movie(movie_path)
		all_per_frame.append(per_frame_df)
		all_summary.append(summary)

	per_frame_results_df = pd.concat(all_per_frame, ignore_index=True)
	summary_results_df = pd.DataFrame(all_summary)

	per_frame_csv = output_dir / "movie_framewise_change_metrics.csv"
	summary_csv = output_dir / "movie_summary_metrics.csv"

	per_frame_results_df.to_csv(per_frame_csv, index=False)
	summary_results_df.to_csv(summary_csv, index=False)
	generate_framewise_change_plots(per_frame_results_df, output_dir)

	print(f"Saved frame-wise results: {per_frame_csv}")
	print(f"Saved movie summary results: {summary_csv}")
	print(f"Saved frame-wise plots to: {output_dir}")

	return per_frame_results_df, summary_results_df


def main() -> None:
	analyze_movies(MOVIES_INPUT_PATH, OUTPUT_DIR)


if __name__ == "__main__":
	main()

