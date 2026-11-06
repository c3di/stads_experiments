#!/usr/bin/env python3
"""
Per-frame comparison of the best adaptive configuration vs. baselines.

For each (gt_name, scanned_pixel_percent) tuple found in the adaptive runs of
per_frame_results.csv, this script:

  1. Aggregates frame-level PSNR/SSIM per experiment identity (same identity
     columns and ranking scheme as parameter_tune.py: psnr_rank/ssim_rank with
     method='min' ascending=False, combined = psnr_rank**2 + ssim_rank**2).
  2. Picks the "best configuration" for the tuple with the following rule:
       - if the best-combined config is identical to the best-PSNR config
         -> use the best-PSNR config
       - else if the best-combined config is identical to the best-SSIM config
         -> use the best-SSIM config
       - else (all three differ) -> use the best-PSNR config
  3. Computes mean and stddev (np.std, ddof=0, matching parameter_tune.py) of
     PSNR and SSIM over ALL frames of that chosen configuration. The same
     single configuration is used for both the PSNR and the SSIM numbers.
  4. Does the same mean/stddev-over-frames calculation for the stratified
     random baseline (sampler == 'stratified') and the low dwell baseline
     (sampler == 'low_dwell') of the same tuple.

Output: one CSV row per tuple with columns
  gt name | sparsity | adaptive psnr average (stdev) | adaptive ssim average (stdev)
  | stratified random psnr average (stdev) | stratified random ssim average (stdev)
  | low dwell psnr average (stdev) | low dwell ssim average (stdev)
where each metric cell is formatted as "<mean> (<std>)" (4 decimals).

Usage:
    python best_config_frame_stats.py [csv_file] [--output OUT_CSV]

Defaults:
    csv_file : <repo_root>/plots/per_frame_results.csv
    output   : <script_dir>/best_config_frame_comparison.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Identity columns: same experiment identity definition as parameter_tune.py
IDENTITY_COLUMNS = [
    'withTemporalSampler', 'withTemporalReconstruction', 'gt_name',
    'scanned_pixel_percent', 'alpha', 'beta', 'adaptiveFraction',
    'minDensityGamma', 'sampleSequence', 'temporalMethod',
    'temporalResidualCutoff', 'temporalResidualConfidenceScale'
]

OUTPUT_COLUMNS = [
    'gt name', 'sparsity',
    'adaptive psnr average (stdev)', 'adaptive ssim average (stdev)',
    'stratified random psnr average (stdev)', 'stratified random ssim average (stdev)',
    'low dwell psnr average (stdev)', 'low dwell ssim average (stdev)',
]


def fmt(mean, std):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return ''
    return f"{mean:.4f} ({std:.4f})"


def frame_stats(df):
    """Mean/std (ddof=0) of PSNR and SSIM over all frames, NaNs excluded."""
    psnr = df['PSNR'].dropna()
    ssim = df['SSIM'].dropna()
    return (
        float(psnr.mean()) if len(psnr) else np.nan,
        float(psnr.std(ddof=0)) if len(psnr) else np.nan,
        float(ssim.mean()) if len(ssim) else np.nan,
        float(ssim.std(ddof=0)) if len(ssim) else np.nan,
    )


def choose_best_config(exp_stats):
    """
    Given per-experiment stats for one (gt, sparsity) tuple, return the chosen
    experiment identity row plus a human-readable reason.
    """
    df = exp_stats.copy()
    df['psnr_rank'] = df['psnr_mean'].rank(method='min', ascending=False)
    df['ssim_rank'] = df['ssim_mean'].rank(method='min', ascending=False)
    df['combined_rank'] = df['psnr_rank'] ** 2 + df['ssim_rank'] ** 2

    best_psnr = df.loc[df['psnr_mean'].idxmax()]
    best_ssim = df.loc[df['ssim_mean'].idxmax()]
    best_comb = df.loc[df['combined_rank'].idxmin()]

    def identity(row):
        return tuple(row[c] for c in IDENTITY_COLUMNS)

    if identity(best_comb) == identity(best_psnr):
        return best_psnr, 'combined == best PSNR -> use best PSNR config'
    if identity(best_comb) == identity(best_ssim):
        return best_ssim, 'combined == best SSIM -> use best SSIM config'
    return best_psnr, 'all three differ -> use best PSNR config'


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv_file', nargs='?',
                        default=os.path.join(repo_root, 'plots', 'per_frame_results.csv'),
                        help='Input per-frame results CSV (default: plots/per_frame_results.csv)')
    parser.add_argument('--output', default=os.path.join(script_dir, 'best_config_frame_comparison.csv'),
                        help='Output CSV path (default: statistics/best_config_frame_comparison.csv)')
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        sys.exit(1)

    print(f"Loading {args.csv_file} ...")
    df = pd.read_csv(args.csv_file, low_memory=False)
    print(f"  {len(df)} rows, samplers: {sorted(df['sampler'].unique())}")

    # --- adaptive: per-experiment aggregation -------------------------------
    df_adaptive = df[df['sampler'] == 'adaptive'].copy()
    exp_stats = (df_adaptive
                 .groupby(IDENTITY_COLUMNS, dropna=False)
                 .agg(psnr_mean=('PSNR', 'mean'), ssim_mean=('SSIM', 'mean'),
                      num_frames=('PSNR', 'size'))
                 .reset_index())
    print(f"  {len(exp_stats)} adaptive experiment configurations")

    # --- choose best config per (gt, sparsity) tuple ------------------------
    tuples = (exp_stats[['gt_name', 'scanned_pixel_percent']]
              .drop_duplicates()
              .sort_values(['gt_name', 'scanned_pixel_percent']))

    chosen_rows = []
    for _, t in tuples.iterrows():
        gt, sparsity = t['gt_name'], t['scanned_pixel_percent']
        subset = exp_stats[(exp_stats['gt_name'] == gt) &
                           (exp_stats['scanned_pixel_percent'] == sparsity)]
        chosen, reason = choose_best_config(subset)
        print(f"  {gt} @ {sparsity}: {reason}")
        chosen_rows.append(chosen)

    chosen_df = pd.DataFrame(chosen_rows)[IDENTITY_COLUMNS]

    # --- frame-level stats of the chosen adaptive configs -------------------
    chosen_frames = df_adaptive.merge(chosen_df, on=IDENTITY_COLUMNS, how='inner')
    adaptive_stats = (chosen_frames
                      .groupby(['gt_name', 'scanned_pixel_percent'])
                      .apply(lambda g: pd.Series(frame_stats(g), index=['p', 'ps', 's', 'ss']),
                             include_groups=False)
                      .reset_index())

    # --- baselines -----------------------------------------------------------
    def baseline_stats(sampler_name):
        sub = df[df['sampler'] == sampler_name]
        if len(sub) == 0:
            return pd.DataFrame(columns=['gt_name', 'scanned_pixel_percent', 'p', 'ps', 's', 'ss'])
        return (sub.groupby(['gt_name', 'scanned_pixel_percent'])
                .apply(lambda g: pd.Series(frame_stats(g), index=['p', 'ps', 's', 'ss']),
                       include_groups=False)
                .reset_index())

    strat_stats = baseline_stats('stratified')
    lowdwell_stats = baseline_stats('low_dwell')

    # --- assemble output table ----------------------------------------------
    rows = []
    for _, t in tuples.iterrows():
        gt, sparsity = t['gt_name'], t['scanned_pixel_percent']
        a = adaptive_stats[(adaptive_stats['gt_name'] == gt) &
                           (adaptive_stats['scanned_pixel_percent'] == sparsity)]
        s = strat_stats[(strat_stats['gt_name'] == gt) &
                        (strat_stats['scanned_pixel_percent'] == sparsity)] if len(strat_stats) else strat_stats
        l = lowdwell_stats[(lowdwell_stats['gt_name'] == gt) &
                           (lowdwell_stats['scanned_pixel_percent'] == sparsity)] if len(lowdwell_stats) else lowdwell_stats

        a_p, a_ps, a_s, a_ss = (a.iloc[0][['p', 'ps', 's', 'ss']] if len(a) else (np.nan,) * 4)
        s_p, s_ps, s_s, s_ss = (s.iloc[0][['p', 'ps', 's', 'ss']] if len(s) else (np.nan,) * 4)
        l_p, l_ps, l_s, l_ss = (l.iloc[0][['p', 'ps', 's', 'ss']] if len(l) else (np.nan,) * 4)

        rows.append({
            'gt name': gt,
            'sparsity': sparsity,
            'adaptive psnr average (stdev)': fmt(a_p, a_ps),
            'adaptive ssim average (stdev)': fmt(a_s, a_ss),
            'stratified random psnr average (stdev)': fmt(s_p, s_ps),
            'stratified random ssim average (stdev)': fmt(s_s, s_ss),
            'low dwell psnr average (stdev)': fmt(l_p, l_ps),
            'low dwell ssim average (stdev)': fmt(l_s, l_ss),
        })

    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out_df.to_csv(args.output, index=False)
    print(f"\nWrote {len(out_df)} rows to {args.output}")


if __name__ == '__main__':
    main()
