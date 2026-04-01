from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _format_group_label(
	sampler: str,
	with_temporal_sampler: str,
	with_temporal_reconstruction: str,
) -> str:
	sampler_value = str(sampler).strip()
	ts_value = str(with_temporal_sampler).strip()
	tr_value = str(with_temporal_reconstruction).strip()

	sampler_lower = sampler_value.lower()
	ts_enabled = ts_value.lower() == "true"
	tr_enabled = tr_value.lower() == "true"

	if sampler_lower == "adaptive" and ts_enabled and tr_enabled:
		return "Full"
	if sampler_lower == "stratified" and not ts_enabled and not tr_enabled:
		return "Baseline"
	
	if sampler_lower == "adaptive" and not ts_enabled and not tr_enabled:
		return "No Temporal Sampling/Reconstruction"
	
	if sampler_lower == "adaptive" and ts_enabled and not tr_enabled:
		return "Temporal Sampling Only"
	
	if sampler_lower == "adaptive" and not ts_enabled and tr_enabled:
		return "Temporal Reconstruction Only"


	return f"{sampler_value} | TS={ts_value} | TR={tr_value}"


def generate_framewise_line_plots(
	csv_path: str | Path = "plots/per_frame_results.csv",
	output_dir: str | Path = "plots/statistics",
) -> None:
	csv_path = Path(csv_path)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	df = pd.read_csv(csv_path)

	df = df.copy()
	df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="coerce")
	df["scanned_pixel_percent"] = pd.to_numeric(df["scanned_pixel_percent"], errors="coerce")
	df["withTemporalSampler"] = df["withTemporalSampler"].fillna("False").astype(str).str.strip()
	df.loc[df["withTemporalSampler"].eq(""), "withTemporalSampler"] = "False"
	df = df.dropna(subset=["frame_idx", "scanned_pixel_percent", "PSNR", "SSIM"])
	df["withTemporalReconstruction"] = df["withTemporalReconstruction"].fillna("False").astype(str).str.strip()
	df.loc[df["withTemporalReconstruction"].eq(""), "withTemporalReconstruction"] = "False"
	df["sampler"] = df["sampler"].astype(str)

	df["frame_idx"] = df["frame_idx"].astype(int)

	sns.set_theme(style="whitegrid", context="talk")

	df["group_label"] = (
		df.apply(
			lambda row: _format_group_label(
				row["sampler"],
				row["withTemporalSampler"],
				row["withTemporalReconstruction"],
			),
			axis=1,
		)
	)

	for scanned_pixel_percent, gt_df in df.groupby("scanned_pixel_percent", sort=True):
		gt_df = gt_df.sort_values(["gt_name", "group_label", "frame_idx"])

		for metric in ["PSNR", "SSIM"]:
			plt.figure(figsize=(12, 6))
			sns.lineplot(
				data=gt_df,
				x="frame_idx",
				y=metric,
				hue="group_label",
				style="gt_name",
				markers=True,
				dashes=False,
				errorbar=None,
			)

			plt.title(f"{metric} vs frame_idx | {scanned_pixel_percent}% scanned pixels")
			plt.xlabel("frame_idx")
			plt.ylabel(metric)
			if metric == "SSIM":
				plt.ylim(0, 1.0)
			else:
				plt.ylim(bottom=0)
			plt.legend(
				title="scanned_pixel_percent / group_label",
				bbox_to_anchor=(1.02, 1),
				loc="upper left",
				borderaxespad=0,
			)
			plt.tight_layout()

			output_path = output_dir / f"{scanned_pixel_percent}_{metric.lower()}_lineplot.png"
			plt.savefig(output_path, dpi=300)
			plt.close()


def generate_averaged_metric_vs_scanned_pixel_plots(
	csv_path: str | Path = "plots/per_frame_results.csv",
	output_dir: str | Path = "plots/statistics",
) -> None:
	csv_path = Path(csv_path)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	df = pd.read_csv(csv_path)

	df = df.copy()
	df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="coerce")
	df["scanned_pixel_percent"] = pd.to_numeric(df["scanned_pixel_percent"], errors="coerce")
	df["PSNR"] = pd.to_numeric(df["PSNR"], errors="coerce")
	df["SSIM"] = pd.to_numeric(df["SSIM"], errors="coerce")
	df["withTemporalSampler"] = df["withTemporalSampler"].fillna("False").astype(str).str.strip()
	df.loc[df["withTemporalSampler"].eq(""), "withTemporalSampler"] = "False"
	df["withTemporalReconstruction"] = df["withTemporalReconstruction"].fillna("False").astype(str).str.strip()
	df.loc[df["withTemporalReconstruction"].eq(""), "withTemporalReconstruction"] = "False"
	df = df.dropna(
		subset=[
			"gt_name",
			"sampler",
			"scanned_pixel_percent",
			"frame_idx",
			"PSNR",
			"SSIM",
		]
	)

	df["sampler"] = df["sampler"].astype(str)

	aggregated_df = (
		df.groupby(["gt_name", "sampler", "withTemporalSampler", "withTemporalReconstruction", "scanned_pixel_percent"], as_index=False)[["PSNR", "SSIM"]]
		.mean()
		.sort_values(["gt_name", "sampler", "withTemporalSampler", "withTemporalReconstruction", "scanned_pixel_percent"])
	)

	aggregated_df["group_label"] = (
		aggregated_df.apply(
			lambda row: _format_group_label(
				row["sampler"],
				row["withTemporalSampler"],
				row["withTemporalReconstruction"],
			),
			axis=1,
		)
	)

	sns.set_theme(style="whitegrid", context="talk")

	for gt_name, gt_df in aggregated_df.groupby("gt_name", sort=True):
		for metric in ["PSNR", "SSIM"]:
			plt.figure(figsize=(12, 6))
			sns.lineplot(
				data=gt_df,
				x="scanned_pixel_percent",
				y=metric,
				hue="group_label",
				marker="X",
				markersize=5,
				markeredgewidth=0,
				dashes=False,
				errorbar=None,
			)

			plt.title(f"Average {metric} vs scanned_pixel_percent | {gt_name}")
			plt.xlabel("scanned_pixel_percent")
			plt.ylabel(f"Average {metric}")
			if metric == "SSIM":
				plt.ylim(0, 1.0)
			else:
				plt.ylim(bottom=0)
			plt.legend(
				title="sampler | TS | TR",
				bbox_to_anchor=(1.02, 1),
				loc="upper left",
				borderaxespad=0,
			)
			plt.tight_layout()

			output_path = output_dir / f"{gt_name}_{metric.lower()}_avg_vs_scanned_pixel_percent.png"
			plt.savefig(output_path, dpi=300)
			plt.close()


if __name__ == "__main__":
	generate_averaged_metric_vs_scanned_pixel_plots()
	generate_framewise_line_plots()
