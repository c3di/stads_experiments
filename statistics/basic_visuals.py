from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

	df["frame_idx"] = df["frame_idx"].astype(int)

	sns.set_theme(style="whitegrid", context="talk")

	df["group_label"] = (
	df["sampler"]
	+ " | TS=" + df["withTemporalSampler"]
	+ " | TR=" + df["withTemporalReconstruction"]
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
		aggregated_df["sampler"]
		+ " | TS=" + aggregated_df["withTemporalSampler"]
		+ " | TR=" + aggregated_df["withTemporalReconstruction"]
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
				markers=True,
				dashes=False,
				errorbar=None,
			)

			plt.title(f"Average {metric} vs scanned_pixel_percent | {gt_name}")
			plt.xlabel("scanned_pixel_percent")
			plt.ylabel(f"Average {metric}")
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
