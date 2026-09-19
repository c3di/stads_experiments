#!/usr/bin/env python3
"""
Parameter vs Response Variable Plots Generator

Generates scatter plots of a single parameter against PSNR, SSIM, and combined_score,
grouped by ground truth and scanned_pixel_percent buckets.

Input:
    - parameter: Name of the parameter to plot (must match column in aggregated CSV)
    - --gt: Ground truth names to include (optional, defaults to all)
    - --sparsity: Sparsity levels to include (optional, defaults to all)
    - --all: Include all ground truths and sparsity levels

Output:
    - Three scatter plots saved in: statistics/plots/{parameter_name}/{gt_name}/sparsity_X.X/
      * parameter_vs_psnr_mean.png
      * parameter_vs_ssim_mean.png
      * parameter_vs_combined_score.png

Usage:
    python parameter_vs_response_plots.py alpha --gt HYDRATION_ONE --sparsity 0.1 0.5 1.0
    
    OR with all ground truths and sparsities:
    python parameter_vs_response_plots.py alpha --all

The combined_score is calculated as: psnr_rank^2 + ssim_rank^2 (lower is better)
where ranks are descending (rank 1 = highest PSNR/SSIM mean = best)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def load_aggregated_data(tune_results_dir, gt_names=None, sparsity_levels=None):
    """Load aggregated CSV files from tune_results directory."""
    if not os.path.exists(tune_results_dir):
        print(f"Error: tune_results directory not found: {tune_results_dir}")
        return None
    
    csv_files = [f for f in os.listdir(tune_results_dir) if f.endswith('_parameter_analysis.csv')]
    if not csv_files:
        print(f"No aggregated CSV files found in {tune_results_dir}")
        return None
    
    all_data = []
    for csv_file in csv_files:
        file_path = os.path.join(tune_results_dir, csv_file)
        df = pd.read_csv(file_path)
        gt_name = csv_file.replace('_parameter_analysis.csv', '')
        df['gt_name'] = gt_name
        if gt_names and gt_name not in gt_names:
            continue
        all_data.append(df)
    
    if not all_data:
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    if sparsity_levels:
        combined_df = combined_df[combined_df['scanned_pixel_percent'].isin(sparsity_levels)]
    
    return combined_df


def compute_combined_score(df_bucket):
    """Compute combined_score within a bucket as psnr_rank^2 + ssim_rank^2."""
    if 'psnr_mean' not in df_bucket.columns or 'ssim_mean' not in df_bucket.columns:
        return None
    psnr_rank = df_bucket['psnr_mean'].rank(method='min', ascending=False)
    ssim_rank = df_bucket['ssim_mean'].rank(method='min', ascending=False)
    return psnr_rank**2 + ssim_rank**2


def create_scatter_plot(df_bucket, parameter_name, response_var, response_values, output_path):
    """Create a scatter plot of parameter vs response variable."""
    if len(df_bucket) == 0:
        return
    
    x_values = df_bucket[parameter_name].values
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(
        x=x_values,
        y=response_values,
        hue=df_bucket.get('gt_name', 'all'),
        palette='tab10',
        s=60,
        alpha=0.7,
        edgecolor='w',
        linewidth=0.5
    )
    
    plt.xlabel(parameter_name, fontsize=12)
    plt.ylabel(response_var, fontsize=12)
    
    if response_var == 'combined_score':
        plt.title(f'{parameter_name} vs {response_var}\n(Lower combined_score = better)', fontsize=14)
    else:
        plt.title(f'{parameter_name} vs {response_var}', fontsize=14)
    
    plt.legend(title='Ground Truth', fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_plots_for_parameter(df, parameter_name, gt_names, sparsity_levels, output_base_dir):
    """Create all plots for a specific parameter across all buckets."""
    if parameter_name not in df.columns:
        print(f"Error: Parameter '{parameter_name}' not found in data columns")
        print(f"Available columns: {list(df.columns)}")
        return
    
    available_gt_names = sorted(df['gt_name'].unique())
    available_sparsities = sorted(df['scanned_pixel_percent'].unique())
    
    if gt_names:
        available_gt_names = [gt for gt in available_gt_names if gt in gt_names]
    if sparsity_levels:
        available_sparsities = [s for s in available_sparsities if s in sparsity_levels]
    
    print(f"\nProcessing parameter: {parameter_name}")
    print(f"Ground truths: {available_gt_names}")
    print(f"Sparsity levels: {available_sparsities}")
    
    for gt_name in available_gt_names:
        for sparsity in available_sparsities:
            bucket_mask = (df['gt_name'] == gt_name) & (df['scanned_pixel_percent'] == sparsity)
            df_bucket = df[bucket_mask].copy()
            
            if len(df_bucket) == 0:
                print(f"  Skipping {gt_name} @ {sparsity}% (no data)")
                continue
            
            print(f"\n  Bucket: {gt_name} @ {sparsity}%")
            print(f"  Experiments: {len(df_bucket)}")
            
            combined_scores = compute_combined_score(df_bucket)
            sparsity_dir = f"sparsity_{sparsity}"
            output_dir = os.path.join(output_base_dir, parameter_name, gt_name, sparsity_dir)
            
            response_vars = ['psnr_mean', 'ssim_mean', 'combined_score']
            for resp_var in response_vars:
                if resp_var == 'combined_score':
                    if combined_scores is None:
                        continue
                    response_values = combined_scores
                else:
                    response_values = df_bucket[resp_var].values
                
                filename = f"{parameter_name}_vs_{resp_var}.png"
                output_path = os.path.join(output_dir, filename)
                create_scatter_plot(df_bucket, parameter_name, resp_var, response_values, output_path)


def main():
    parser = argparse.ArgumentParser(description='Generate parameter vs response variable plots')
    
    parser.add_argument('parameter', type=str, help='Parameter name to plot on x-axis')
    parser.add_argument('--gt', type=str, nargs='*', default=None, 
                        help='Ground truth names to include (optional, defaults to all)')
    parser.add_argument('--sparsity', type=float, nargs='*', default=None,
                        help='Sparsity levels to include (optional, defaults to all)')
    parser.add_argument('--all', action='store_true',
                        help='Include all ground truths and sparsity levels')
    parser.add_argument('--output-dir', type=str, 
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameter_plots'),
                        help='Base output directory (default: statistics/parameter_plots)')
    
    args = parser.parse_args()
    
    tune_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tune_results")
    df = load_aggregated_data(tune_results_dir)
    
    if df is None:
        print("Error: Could not load data. Please run parameter_tune.py --analyze first.")
        sys.exit(1)
    
    print(f"Loaded {len(df)} aggregated experiments from {tune_results_dir}")
    
    gt_names = None if args.all else args.gt
    sparsity_levels = None if args.all else args.sparsity
    
    if not args.all and not gt_names and not sparsity_levels:
        print("Error: Please provide --gt and/or --sparsity arguments, or use --all")
        print("Example: python parameter_vs_response_plots.py alpha --gt HYDRATION_ONE --sparsity 0.1 0.5")
        sys.exit(1)
    
    create_plots_for_parameter(df, args.parameter, gt_names, sparsity_levels, args.output_dir)
    print(f"\nDone! Plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
