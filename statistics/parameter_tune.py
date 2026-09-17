#!/usr/bin/env python3
"""
Parameter Tuning Analysis Script

Workflow 1 (--analyze):
  Takes a per_frame_results CSV, filters for adaptive sampler runs, and:
  1. Identifies all ground truths used
  2. For each ground truth:
     a. Aggregates frame-level results by experiment identity
     b. Computes statistics (mean, std, max, min) for PSNR and SSIM
     c. Writes results to statistics/tune_results/{gt_name}_parameter_analysis.csv
  3. For each ground truth and scanned_pixel_percent, prints best configuration
     by PSNR mean, SSIM mean, and combined rank (sum of squared ranks)

Workflow 2 (--plot):
  Reads all aggregated CSVs from statistics/tune_results/ and creates:
  - Summary scatter plots for each scanned_pixel_percent
  - PSNR on X axis, SSIM on Y axis
  - Ground truth distinguished by color

Usage:
    python parameter_tune.py [--analyze [csv_file]] [--plot]
    
Default (no args): Runs both workflows in sequence (analyze then plot)
Default CSV for --analyze: plots/per_frame_results.csv
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns


def create_summary_plots():
    """Workflow 2: Create summary scatter plots from aggregated CSV files."""
    tune_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tune_results")
    
    if not os.path.exists(tune_results_dir):
        print(f"Error: tune_results directory not found: {tune_results_dir}")
        print("Please run --analyze first to generate the aggregated CSV files.")
        return
    
    # Find all aggregated CSV files
    csv_files = [f for f in os.listdir(tune_results_dir) if f.endswith('_parameter_analysis.csv')]
    
    if not csv_files:
        print(f"No aggregated CSV files found in {tune_results_dir}")
        print("Please run --analyze first to generate the aggregated CSV files.")
        return
    
    print(f"Found {len(csv_files)} aggregated CSV files")
    
    # Load all data
    all_data = []
    for csv_file in csv_files:
        file_path = os.path.join(tune_results_dir, csv_file)
        df = pd.read_csv(file_path)
        # Extract gt_name from filename (remove _parameter_analysis.csv)
        gt_name = csv_file.replace('_parameter_analysis.csv', '')
        df['source_gt'] = gt_name
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Get unique scanned_pixel_percent values
    sparsity_levels = sorted(combined_df['scanned_pixel_percent'].unique())
    
    # Create a plot for each sparsity level
    for sparsity in sparsity_levels:
        sparsity_df = combined_df[combined_df['scanned_pixel_percent'] == sparsity]
        
        if len(sparsity_df) == 0:
            continue
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Use seaborn scatterplot with hue for ground truth
        ax = sns.scatterplot(
            data=sparsity_df,
            x='psnr_mean',
            y='ssim_mean',
            hue='source_gt',
            palette='tab10',
            alpha=0.7,
            s=100
        )
        
        plt.xlabel('PSNR (mean)', fontsize=12)
        plt.ylabel('SSIM (mean)', fontsize=12)
        plt.title(f'PSNR vs SSIM - scanned_pixel_percent = {sparsity}', fontsize=14)
        plt.legend(title='Ground Truth', fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(tune_results_dir, f'psnr_vs_ssim_sparsity_{sparsity}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot: {plot_file}")
    
    print(f"\nSummary plots created in {tune_results_dir}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Parameter tuning analysis and visualization'
    )
    
    # Mutually exclusive group for workflow selection
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--analyze',
        action='store_true',
        help='Run analysis workflow: aggregate results and find best configurations'
    )
    group.add_argument(
        '--plot',
        action='store_true',
        help='Run plotting workflow: create summary scatter plots from aggregated CSVs'
    )
    
    parser.add_argument(
        'csv_file',
        nargs='?',
        default=os.path.join('plots', 'per_frame_results.csv'),
        help='Input CSV file for --analyze (default: plots/per_frame_results.csv)'
    )
    
    args = parser.parse_args()
    
    # Default behavior: run both workflows if no arguments provided
    if not args.analyze and not args.plot:
        run_analysis(args.csv_file)
        create_summary_plots()
    else:
        # Route to appropriate workflow
        if args.analyze:
            run_analysis(args.csv_file)
        elif args.plot:
            create_summary_plots()


def run_analysis(csv_path):
    """Workflow 1: Analyze adaptive sampler results and find best configurations."""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Filter for adaptive sampler only
    df_adaptive = df[df['sampler'] == 'adaptive'].copy()
    
    if len(df_adaptive) == 0:
        print("No adaptive sampler data found in CSV")
        return
    
    # Identify all ground truths
    ground_truths = sorted(df_adaptive['gt_name'].unique())
    print(f"Found {len(ground_truths)} ground truths: {ground_truths}")
    
    # Columns that define experiment identity (all except frame_idx and sampler)
    identity_columns = [
        'withTemporalSampler', 'withTemporalReconstruction', 'gt_name',
        'scanned_pixel_percent', 'alpha', 'beta', 'adaptiveFraction',
        'minDensityGamma', 'sampleSequence', 'temporalMethod',
        'temporalResidualCutoff', 'temporalResidualConfidenceScale'
    ]
    
    # For each ground truth, process experiments
    gt_aggregated = {}  # {gt_name: list of aggregated stats}
    
    for gt_name in ground_truths:
        print(f"\nProcessing {gt_name}...")
        
        # Filter for this ground truth
        gt_df = df_adaptive[df_adaptive['gt_name'] == gt_name]
        
        # Group by experiment identity
        grouped = gt_df.groupby(identity_columns)
        
        aggregated_data = []
        
        for exp_id, group in grouped:
            # Extract identity parameters - exp_id is a tuple, convert to dict
            exp_params = dict(zip(identity_columns, exp_id))
            
            # Get PSNR and SSIM values
            psnr_values = group['PSNR'].values
            ssim_values = group['SSIM'].values
            
            # Remove NaN values for statistics
            psnr_clean = psnr_values[~pd.isna(psnr_values)]
            ssim_clean = ssim_values[~pd.isna(ssim_values)]
            
            # Compute statistics
            stats = {
                **exp_params,
                'num_frames': len(group),
                'psnr_mean': np.mean(psnr_clean) if len(psnr_clean) > 0 else np.nan,
                'psnr_std': np.std(psnr_clean) if len(psnr_clean) > 0 else np.nan,
                'psnr_max': np.max(psnr_clean) if len(psnr_clean) > 0 else np.nan,
                'psnr_min': np.min(psnr_clean) if len(psnr_clean) > 0 else np.nan,
                'ssim_mean': np.mean(ssim_clean) if len(ssim_clean) > 0 else np.nan,
                'ssim_std': np.std(ssim_clean) if len(ssim_clean) > 0 else np.nan,
                'ssim_max': np.max(ssim_clean) if len(ssim_clean) > 0 else np.nan,
                'ssim_min': np.min(ssim_clean) if len(ssim_clean) > 0 else np.nan,
            }
            aggregated_data.append(stats)
            gt_aggregated.setdefault(gt_name, []).append(stats)
        
        # Create DataFrame from aggregated data
        result_df = pd.DataFrame(aggregated_data)
        
        # Reorder columns: identity params first, then stats
        identity_cols = list(exp_params.keys())
        stat_cols = ['num_frames', 'psnr_mean', 'psnr_std', 'psnr_max', 'psnr_min',
                     'ssim_mean', 'ssim_std', 'ssim_max', 'ssim_min']
        result_df = result_df[identity_cols + stat_cols]
        
        # Write to CSV in statistics/tune_results subfolder
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tune_results")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{gt_name}_parameter_analysis.csv")
        result_df.to_csv(output_file, index=False)
        print(f"  Saved {len(result_df)} aggregated experiments to {output_file}")
    
    # Now find best configurations for each ground truth and scanned_pixel_percent
    print("\n" + "="*60)
    print("BEST CONFIGURATIONS BY ground_truth AND scanned_pixel_percent")
    print("="*60)
    
    if not gt_aggregated:
        print("No data to analyze")
        return
    
    stat_cols = ['num_frames', 'psnr_mean', 'psnr_std', 'psnr_max', 'psnr_min',
                 'ssim_mean', 'ssim_std', 'ssim_max', 'ssim_min']
    
    for gt_name, gt_data in gt_aggregated.items():
        print(f"\n--- {gt_name} ---")
        
        # Convert to DataFrame for this ground truth
        gt_df = pd.DataFrame(gt_data)
        
        # Get unique scanned_pixel_percent values for this ground truth
        sparsity_levels = sorted(gt_df['scanned_pixel_percent'].unique())
        
        for sparsity in sparsity_levels:
            print(f"\n  --- scanned_pixel_percent = {sparsity} ---")
            
            # Filter for this sparsity level
            sparsity_df = gt_df[gt_df['scanned_pixel_percent'] == sparsity].copy()
            
            if len(sparsity_df) == 0:
                continue
            
            # Sort by PSNR mean (descending) and assign ranks
            sparsity_df['psnr_rank'] = sparsity_df['psnr_mean'].rank(method='min', ascending=False)
            # Sort by SSIM mean (descending) and assign ranks
            sparsity_df['ssim_rank'] = sparsity_df['ssim_mean'].rank(method='min', ascending=False)
            # Compute combined rank (sum of squared ranks)
            sparsity_df['combined_rank'] = sparsity_df['psnr_rank']**2 + sparsity_df['ssim_rank']**2
            
            # Find best by PSNR mean
            best_psnr_idx = sparsity_df['psnr_mean'].idxmax()
            best_psnr = sparsity_df.loc[best_psnr_idx]
            
            # Find best by SSIM mean
            best_ssim_idx = sparsity_df['ssim_mean'].idxmax()
            best_ssim = sparsity_df.loc[best_ssim_idx]
            
            # Find best combined (lowest sum of squared ranks)
            best_combined_idx = sparsity_df['combined_rank'].idxmin()
            best_combined = sparsity_df.loc[best_combined_idx]
            
            # Print best PSNR configuration
            print(f"    Best PSNR (mean={best_psnr['psnr_mean']:.4f}):")
            identity_str = ", ".join([f"{k}={v}" for k, v in best_psnr.items() if k not in stat_cols + ['psnr_rank', 'ssim_rank', 'combined_rank']])
            print(f"      {identity_str}")
            
            # Print best SSIM configuration
            print(f"    Best SSIM (mean={best_ssim['ssim_mean']:.4f}):")
            identity_str = ", ".join([f"{k}={v}" for k, v in best_ssim.items() if k not in stat_cols + ['psnr_rank', 'ssim_rank', 'combined_rank']])
            print(f"      {identity_str}")
            
            # Print best combined configuration
            print(f"    Best Combined (PSNR rank={int(best_combined['psnr_rank'])}, SSIM rank={int(best_combined['ssim_rank'])}, score={best_combined['combined_rank']:.0f}):")
            identity_str = ", ".join([f"{k}={v}" for k, v in best_combined.items() if k not in stat_cols + ['psnr_rank', 'ssim_rank', 'combined_rank']])
            print(f"      {identity_str}")


if __name__ == '__main__':
    main()
