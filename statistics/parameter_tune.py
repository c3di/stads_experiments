#!/usr/bin/env python3
"""
Parameter Tuning Analysis Script

Takes a per_frame_results CSV, filters for adaptive sampler runs, and:
1. Identifies all ground truths used
2. For each ground truth:
   a. Aggregates frame-level results by experiment identity
   b. Computes statistics (mean, std, max, min) for PSNR and SSIM
   c. Writes results to a separate CSV per ground truth
3. For each scanned_pixel_percent, prints best configuration by PSNR and SSIM mean

Usage:
    python parameter_tune.py [csv_file]
    
Default CSV: plots/per_frame_results.csv
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Analyze adaptive sampler results for parameter tuning'
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        default=os.path.join('plots', 'per_frame_results.csv'),
        help='Input CSV file (default: plots/per_frame_results.csv)'
    )
    args = parser.parse_args()
    
    csv_path = args.csv_file
    
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
            sparsity_df = gt_df[gt_df['scanned_pixel_percent'] == sparsity]
            
            if len(sparsity_df) == 0:
                continue
            
            # Find best by PSNR mean
            best_psnr_idx = sparsity_df['psnr_mean'].idxmax()
            best_psnr = sparsity_df.loc[best_psnr_idx]
            
            # Find best by SSIM mean
            best_ssim_idx = sparsity_df['ssim_mean'].idxmax()
            best_ssim = sparsity_df.loc[best_ssim_idx]
            
            # Print best PSNR configuration
            print(f"    Best PSNR (mean={best_psnr['psnr_mean']:.4f}):")
            identity_str = ", ".join([f"{k}={v}" for k, v in best_psnr.items() if k not in stat_cols])
            print(f"      {identity_str}")
            
            # Print best SSIM configuration
            print(f"    Best SSIM (mean={best_ssim['ssim_mean']:.4f}):")
            identity_str = ", ".join([f"{k}={v}" for k, v in best_ssim.items() if k not in stat_cols])
            print(f"      {identity_str}")


if __name__ == '__main__':
    main()
