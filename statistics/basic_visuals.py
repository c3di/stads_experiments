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

	if sampler_lower == "low_dwell":
		return "baseline"

	if sampler_lower == "adaptive" and ts_enabled and tr_enabled:
		return "spatiotemporal"
	if sampler_lower == "stratified" and not ts_enabled and not tr_enabled:
		return "stratified"
	
	if sampler_lower == "adaptive" and not ts_enabled and not tr_enabled:
		return "No Temporal Sampling/Reconstruction"
	
	if sampler_lower == "adaptive" and ts_enabled and not tr_enabled:
		return "Temporal Sampling Only"
	
	if sampler_lower == "adaptive" and not ts_enabled and tr_enabled:
		return "Temporal Reconstruction Only"


	return f"{sampler_value} | TS={ts_value} | TR={tr_value}"


def _build_group_palette(group_labels: list[str]) -> dict[str, tuple[float, float, float] | str]:
	labels = [str(label) for label in group_labels]
	palette: dict[str, tuple[float, float, float] | str] = {}

	# Keep the key reference groups visually distinct while other groups stay in a similar spectrum.
	if "stratified" in labels:
		palette["stratified"] = "#d62728"
	if "baseline" in labels:
		palette["baseline"] = "#2ca02c"

	accent_labels = {"stratified", "baseline"}
	non_accent_labels = [label for label in labels if label not in accent_labels]
	if non_accent_labels:
		close_colors = sns.color_palette("Blues", n_colors=len(non_accent_labels) + 2)[2:]
		for label, color in zip(non_accent_labels, close_colors):
			palette[label] = color

	return palette


def _filter_temporal_reconstruction_by_alpha(df: pd.DataFrame, selected_alpha: float) -> pd.DataFrame:
	if "alpha" not in df.columns:
		return df

	tr_mask = df["withTemporalReconstruction"].str.lower().eq("true")
	if not tr_mask.any():
		return df

	df = df.copy()
	df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
	tolerance = 1e-12
	alpha_match = df["alpha"].sub(float(selected_alpha)).abs().le(tolerance)
	filtered_df = df.loc[~tr_mask | alpha_match].copy()

	filtered_tr_mask = filtered_df["withTemporalReconstruction"].str.lower().eq("true")
	if not filtered_tr_mask.any():
		available_alphas = sorted(df.loc[tr_mask, "alpha"].dropna().unique().tolist())
		raise ValueError(
			f"No temporal reconstruction rows found for alpha={selected_alpha}. "
			f"Available alpha values: {available_alphas}"
		)

	return filtered_df


def _filter_groups(df: pd.DataFrame, include_groups: list[str] | None) -> pd.DataFrame:
	if not include_groups:
		return df

	wanted_groups = {
		str(group).strip().lower()
		for group in include_groups
		if str(group).strip()
	}
	if not wanted_groups:
		return df

	group_labels = df["group_label"].astype(str).str.strip().str.lower()
	filtered_df = df.loc[group_labels.isin(wanted_groups)].copy()
	if filtered_df.empty:
		available_groups = sorted(df["group_label"].dropna().astype(str).str.strip().unique().tolist())
		raise ValueError(
			f"No rows matched include_groups={sorted(wanted_groups)}. "
			f"Available groups: {available_groups}"
		)

	return filtered_df

def _filter_scanned_pixel_percent_range(
	df: pd.DataFrame,
	scanned_pixel_percent_range: tuple[float, float] | list[float] | None,
) -> pd.DataFrame:
	if not scanned_pixel_percent_range:
		return df

	if len(scanned_pixel_percent_range) != 2:
		raise ValueError("scanned_pixel_percent_range must contain exactly two values.")

	lower_bound = float(scanned_pixel_percent_range[0])
	upper_bound = float(scanned_pixel_percent_range[1])
	if lower_bound > upper_bound:
		lower_bound, upper_bound = upper_bound, lower_bound

	filtered_df = df.loc[df["scanned_pixel_percent"].between(lower_bound, upper_bound, inclusive="both")].copy()
	if filtered_df.empty:
		available_values = sorted(df["scanned_pixel_percent"].dropna().unique().tolist())
		raise ValueError(
			f"No rows matched scanned_pixel_percent_range=({lower_bound}, {upper_bound}). "
			f"Available scanned_pixel_percent values: {available_values}"
		)

	return filtered_df


def generate_framewise_line_plots(
	csv_path: str | Path = "plots/per_frame_results.csv",
	output_dir: str | Path = "plots/statistics",
	selected_alpha: float = 2.0,
	include_groups: list[str] | None = None,
	scanned_pixel_percent_range: tuple[float, float] | list[float] | None = None,
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
	df["withTemporalReconstruction"] = df["withTemporalReconstruction"].fillna("False").astype(str).str.strip()
	df.loc[df["withTemporalReconstruction"].eq(""), "withTemporalReconstruction"] = "False"
	df["sampler"] = df["sampler"].astype(str)
	df = _filter_scanned_pixel_percent_range(df, scanned_pixel_percent_range)
	df = df.dropna(subset=["frame_idx", "scanned_pixel_percent", "PSNR", "SSIM"])
	df = _filter_temporal_reconstruction_by_alpha(df, selected_alpha)

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
	df = _filter_groups(df, include_groups)

	for scanned_pixel_percent, gt_df in df.groupby("scanned_pixel_percent", sort=True):
		gt_df = gt_df.sort_values(["gt_name", "group_label", "frame_idx"])
		group_order = sorted(gt_df["group_label"].dropna().unique().tolist())
		group_palette = _build_group_palette(group_order)

		for metric in ["PSNR", "SSIM"]:
			plt.figure(figsize=(12, 6))
			sns.lineplot(
				data=gt_df,
				x="frame_idx",
				y=metric,
				hue="group_label",
				hue_order=group_order,
				palette=group_palette,
				style="gt_name",
				markers=True,
				dashes=False,
				linewidth=1.4,
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
				title="ground truth / method",
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
	selected_alpha: float = 2.0,
	include_groups: list[str] | None = None,
	scanned_pixel_percent_range: tuple[float, float] | list[float] | None = None,
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
	df = _filter_scanned_pixel_percent_range(df, scanned_pixel_percent_range)
	df = _filter_temporal_reconstruction_by_alpha(df, selected_alpha)
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
	aggregated_df = _filter_groups(aggregated_df, include_groups)
	group_order = sorted(aggregated_df["group_label"].dropna().unique().tolist())
	group_palette = _build_group_palette(group_order)

	csv_table_df = aggregated_df.copy()
	csv_table_df["row_label"] = csv_table_df["gt_name"].astype(str) + " " + csv_table_df["group_label"].astype(str)
	for metric in ["PSNR", "SSIM"]:
		metric_table = (
			csv_table_df.pivot_table(
				index="row_label",
				columns="scanned_pixel_percent",
				values=metric,
				aggfunc="mean",
			)
			.sort_index(axis=0)
			.sort_index(axis=1)
		)
		metric_table.columns = [str(col) for col in metric_table.columns]
		metric_table = metric_table.reset_index().rename(columns={"row_label": "ground_truth_condition"})
		metric_table.to_csv(output_dir / f"{metric.lower()}_avg_vs_scanned_pixel_percent_table.csv", index=False)

	sns.set_theme(style="whitegrid", context="talk")

	for gt_name, gt_df in aggregated_df.groupby("gt_name", sort=True):
		for metric in ["PSNR", "SSIM"]:
			plt.figure(figsize=(12, 6))
			sns.lineplot(
				data=gt_df,
				x="scanned_pixel_percent",
				y=metric,
				hue="group_label",
				hue_order=group_order,
				palette=group_palette,
				marker="X",
				markersize=5,
				markeredgewidth=0,
				dashes=False,
				linewidth=1.4,
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
				title="method",
				bbox_to_anchor=(1.02, 1),
				loc="upper left",
				borderaxespad=0,
			)
			plt.tight_layout()

			output_path = output_dir / f"{gt_name}_{metric.lower()}_avg_vs_scanned_pixel_percent.png"
			plt.savefig(output_path, dpi=300)
			plt.close()


def generate_alpha_comparison_framewise_plots(
	csv_path: str | Path = "plots/per_frame_results.csv",
	output_dir: str | Path = "plots/statistics",
	scanned_pixel_percent_range: tuple[float, float] | list[float] | None = None,
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
	df["sampler"] = df["sampler"].astype(str)
	df["alpha"] = pd.to_numeric(df.get("alpha"), errors="coerce")
	df = _filter_scanned_pixel_percent_range(df, scanned_pixel_percent_range)
	df = df.dropna(subset=["gt_name", "frame_idx", "scanned_pixel_percent", "PSNR", "SSIM"])
	df["frame_idx"] = df["frame_idx"].astype(int)

	sampler_lower = df["sampler"].str.strip().str.lower()
	ts_true = df["withTemporalSampler"].str.lower().eq("true")
	tr_true = df["withTemporalReconstruction"].str.lower().eq("true")

	spatiotemporal_mask = sampler_lower.eq("adaptive") & ts_true & tr_true
	no_temporal_mask = sampler_lower.eq("adaptive") & (~ts_true) & (~tr_true)

	alpha_df = df.loc[spatiotemporal_mask].dropna(subset=["alpha"]).copy()
	baseline_df = df.loc[no_temporal_mask].copy()

	if alpha_df.empty:
		raise ValueError(
			"No spatiotemporal adaptive rows with valid alpha values were found in the CSV."
		)

	if baseline_df.empty:
		raise ValueError(
			"No rows found for the 'No Temporal Sampling/Reconstruction' adaptive condition."
		)

	sns.set_theme(style="whitegrid", context="talk")

	for gt_name, gt_alpha_df in alpha_df.groupby("gt_name", sort=True):
		gt_baseline_df = baseline_df.loc[baseline_df["gt_name"] == gt_name].copy()

		for scanned_pixel_percent, spp_alpha_df in gt_alpha_df.groupby("scanned_pixel_percent", sort=True):
			spp_baseline_df = gt_baseline_df.loc[
				gt_baseline_df["scanned_pixel_percent"] == scanned_pixel_percent
			].copy()

			if spp_baseline_df.empty:
				continue

			alpha_values = sorted(spp_alpha_df["alpha"].dropna().unique().tolist())
			alpha_palette = sns.color_palette("viridis", n_colors=len(alpha_values))

			for metric in ["PSNR", "SSIM"]:
				plt.figure(figsize=(12, 6))

				for alpha_value, alpha_color in zip(alpha_values, alpha_palette):
					alpha_slice = spp_alpha_df.loc[spp_alpha_df["alpha"] == alpha_value]
					if alpha_slice.empty:
						continue
					sns.lineplot(
						data=alpha_slice,
						x="frame_idx",
						y=metric,
						color=alpha_color,
						marker="o",
						dashes=False,
						linewidth=1.4,
						errorbar=None,
						label=f"alpha={alpha_value:g}",
					)

				sns.lineplot(
					data=spp_baseline_df,
					x="frame_idx",
					y=metric,
					color="#444444",
					marker="X",
					dashes=True,
					linewidth=1.6,
					errorbar=None,
					label="No Temporal Sampling/Reconstruction",
				)

				plt.title(
					f"{metric} vs frame_idx | {gt_name} | {scanned_pixel_percent}% scanned pixels"
				)
				plt.xlabel("frame_idx")
				plt.ylabel(metric)
				if metric == "SSIM":
					plt.ylim(0, 1.0)
				else:
					plt.ylim(bottom=0)
				plt.legend(
					title="alpha / condition",
					bbox_to_anchor=(1.02, 1),
					loc="upper left",
					borderaxespad=0,
				)
				plt.tight_layout()

				output_path = output_dir / (
					f"{gt_name}_{scanned_pixel_percent}_{metric.lower()}_alpha_comparison_framewise.png"
				)
				plt.savefig(output_path, dpi=300)
				plt.close()


if __name__ == "__main__":
	selected_alpha = 0.5  # Example: 2.0
	include_groups = ['baseline', 'stratified', 'spatiotemporal']  # Example: ["baseline", "spatiotemporal"]
	scanned_pixel_percent_range = (0.1, 10.0)  # Example: (2.0, 10.0)
	generate_averaged_metric_vs_scanned_pixel_plots(
		selected_alpha=selected_alpha,
		include_groups=include_groups,
		scanned_pixel_percent_range=scanned_pixel_percent_range,
	)
	generate_framewise_line_plots(
		selected_alpha=selected_alpha,
		include_groups=include_groups,
		scanned_pixel_percent_range=scanned_pixel_percent_range,
	)
	generate_alpha_comparison_framewise_plots(
		scanned_pixel_percent_range=scanned_pixel_percent_range,
	)
