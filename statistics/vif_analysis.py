#!/usr/bin/env python3
"""
Variance Inflation Factor (VIF) Analysis for Parameter Tuning

This script implements Phase 1.2 of the statistical analysis proposal:
Computes Variance Inflation Factors to quantify multicollinearity among
parameters within each gt_name x scanned_pixel_percent bucket.

VIF quantifies how much the variance of regression coefficient estimates
is inflated due to correlations among predictors. High VIF (>5-10) indicates
problematic multicollinearity that can make coefficient estimates unstable.

Usage:
    python vif_analysis.py [csv_file]
    
Default CSV: plots/per_frame_results.csv (or aggregated files from tune_results/)

Output:
    - VIF values for each parameter in each bucket
    - Condition number for parameter matrix
    - Identifies parameters with VIF > threshold (default: 5, 10)
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd


def compute_vif(data, numeric_cols, threshold_low=5.0, threshold_high=10.0):
    """
    Compute Variance Inflation Factors for a set of numeric predictors.
    
    VIF Formula: VIF_i = 1 / (1 - R^2_i)
    where R^2_i is from regressing predictor i on all other predictors.
    
    This implementation uses numpy only (no statsmodels dependency).
    
    Args:
        data: DataFrame containing the predictor columns
        numeric_cols: List of column names to compute VIF for
        threshold_low: VIF > this value indicates moderate multicollinearity
        threshold_high: VIF > this value indicates severe multicollinearity
    
    Returns:
        Dictionary with VIF values and condition number
    """
    # Filter to numeric columns only and drop NA
    # Select only numeric dtype columns
    numeric_data = data.select_dtypes(include=[np.number])
    # Intersect with requested columns
    available_numeric = [col for col in numeric_cols if col in numeric_data.columns]
    
    if len(available_numeric) == 0:
        return {
            'vif': None,
            'condition_number': None,
            'high_vif_params_low': [],
            'high_vif_params_high': [],
            'vif_summary': None
        }
    
    vif_data = numeric_data[available_numeric].dropna()
    
    if len(vif_data) == 0 or len(vif_data.columns) == 0:
        return {
            'vif': None,
            'condition_number': None,
            'high_vif_params_low': [],
            'high_vif_params_high': [],
            'vif_summary': None
        }
    
    # Convert to numpy array and ensure float type
    X = vif_data.values.astype(float)
    
    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    col_names = vif_data.columns.tolist()
    n_cols = len(col_names)
    
    # Update numeric_cols to only include available numeric columns
    numeric_cols = col_names
    
    # Calculate VIF for each predictor
    vif_values = []
    for i, col in enumerate(col_names):
        try:
            # Regress column i on all other columns
            # X_i = X_other * beta + epsilon
            # We want R^2 from this regression
            
            # Create design matrix: all columns except i
            X_other = np.delete(X, i, axis=1)
            
            # Ensure X_other is 2D
            if X_other.ndim == 1:
                X_other = X_other.reshape(-1, 1)
            
            y = X[:, i]
            
            # Ensure y is a proper 1D array
            y = np.asarray(y).flatten()
            
            # Ensure y is numeric
            y = y.astype(float)
            
            # Ensure X_other is numeric
            X_other = X_other.astype(float)
            
            # Add intercept term - use float type explicitly
            intercept = np.ones(len(y), dtype=float)
            X_other_with_intercept = np.column_stack([intercept, X_other])
            
            # Solve: beta = (X^T X)^-1 X^T y
            XtX = np.dot(X_other_with_intercept.T, X_other_with_intercept)
            Xty = np.dot(X_other_with_intercept.T, y)
            
            # Check if matrix is invertible
            try:
                beta = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                # Matrix is singular, cannot compute VIF
                vif_values.append(np.nan)
                continue
            
            # Calculate predicted values
            y_pred = np.dot(X_other_with_intercept, beta)
            
            # Calculate R^2
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            
            if ss_tot == 0:
                r_squared = 0
            else:
                r_squared = 1 - (ss_res / ss_tot)
            
            # Calculate VIF
            if r_squared >= 1:
                vif = np.inf
            elif r_squared < 0:  # Can happen due to numerical issues
                vif = 1.0
            else:
                vif = 1.0 / (1.0 - r_squared)
            
            vif_values.append(vif)
            
        except Exception as e:
            warnings.warn(f"Could not compute VIF for {col}: {e}")
            vif_values.append(np.nan)
    
    # Create VIF series
    vif_series = pd.Series(vif_values, index=col_names, name='VIF')
    
    # Calculate condition number of the parameter matrix
    try:
        if X.shape[0] >= X.shape[1] and X.shape[1] > 0:
            # Ensure X is float
            X = X.astype(float)
            # Center the data
            means = np.nanmean(X, axis=0)
            X_centered = X - means
            # Standardize by std for condition number
            stds = np.nanstd(X, axis=0, ddof=1)
            # Avoid division by zero
            stds[stds == 0] = 1.0
            X_scaled = X_centered / stds
            # Replace inf/nan with 0 for safety
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Condition number is ratio of largest to smallest singular value
            _, s, _ = np.linalg.svd(X_scaled)
            if len(s) > 0 and s[-1] > 0:
                condition_number = float(s[0]) / float(s[-1])
            else:
                condition_number = np.inf if len(s) > 0 else None
        else:
            condition_number = None
    except Exception as e:
        warnings.warn(f"Could not compute condition number: {e}")
        condition_number = None
    
    # Identify parameters with high VIF
    high_vif_low = vif_series[vif_series > threshold_low].index.tolist()
    high_vif_high = vif_series[vif_series > threshold_high].index.tolist()
    
    return {
        'vif': vif_series,
        'condition_number': condition_number,
        'high_vif_params_low': high_vif_low,
        'high_vif_params_high': high_vif_high,
        'vif_summary': vif_series.describe()
    }


def analyze_bucket(df_bucket, gt_name, sparsity, vif_params):
    """
    Analyze VIF for a single gt_name x scanned_pixel_percent bucket.
    
    Args:
        df_bucket: DataFrame of experiments for this bucket
        gt_name: Ground truth name
        sparsity: scanned_pixel_percent value
        vif_params: List of parameter names to include in VIF analysis
    """
    print(f"\n{'='*60}")
    print(f"Bucket: {gt_name} | scanned_pixel_percent = {sparsity}")
    print(f"Experiments: {len(df_bucket)}")
    print(f"{'='*60}")
    
    # Check sample size - need at least p + 2 experiments
    # vif_params includes temporalResidualCutoff and temporalResidualConfidenceScale
    # which will be replaced by temporalSensitivity
    # So actual numeric params = 3 (alpha, adaptiveFraction, temporalSensitivity)
    min_required = 5  # At least 5 experiments for reliable VIF
    if len(df_bucket) < min_required:
        print(f"WARNING: Only {len(df_bucket)} experiments. Need at least {min_required}.")
        print("VIF calculation may be unreliable.")
        return None
    
    # Create temporalSensitivity composite and filter parameters
    # vif_params = ['alpha', 'adaptiveFraction', 'temporalResidualCutoff', 'temporalResidualConfidenceScale']
    df_bucket = df_bucket.copy()
    
    # Create temporalSensitivity if both components exist
    if 'temporalResidualCutoff' in df_bucket.columns and \
       'temporalResidualConfidenceScale' in df_bucket.columns:
        df_bucket['temporalSensitivity'] = \
            df_bucket['temporalResidualCutoff'] * df_bucket['temporalResidualConfidenceScale']
    
    # Parameters to analyze: alpha, adaptiveFraction, temporalSensitivity
    params_to_analyze = ['alpha', 'adaptiveFraction']
    if 'temporalSensitivity' in df_bucket.columns:
        params_to_analyze.append('temporalSensitivity')
    
    # Filter to only parameters that exist in the data
    available_params = [p for p in params_to_analyze if p in df_bucket.columns]
    
    # Debug: Check correlations first
    corr_matrix = df_bucket[available_params].corr()
    print(f"\nParameter correlations:")
    print(corr_matrix.round(4))
    
    # Compute VIF
    vif_result = compute_vif(df_bucket, available_params)
    
    if vif_result['vif'] is None:
        print("Could not compute VIF (no numeric parameters or insufficient data)")
        return None
    
    # Print results
    print(f"\nVIF Results:")
    print(f"-" * 60)
    for param, vif in vif_result['vif'].sort_values(ascending=False).items():
        marker = " ***" if vif > 10 else (" **" if vif > 5 else "")
        print(f"  {param:30s}: {vif:8.2f}{marker}")
    
    print(f"\nCondition Number: {vif_result['condition_number']:.2f}" if vif_result['condition_number'] else "Condition Number: N/A")
    
    if vif_result['condition_number'] and vif_result['condition_number'] > 30:
        print("  WARNING: High condition number (>30) indicates near-linear dependencies")
    
    if vif_result['high_vif_params_high']:
        print(f"\nSEVERE multicollinearity (VIF > 10): {vif_result['high_vif_params_high']}")
    
    if vif_result['high_vif_params_low']:
        print(f"MODERATE multicollinearity (VIF > 5): {vif_result['high_vif_params_low']}")
    
    if not vif_result['high_vif_params_low'] and not vif_result['high_vif_params_high']:
        print("\nNo significant multicollinearity detected (all VIF < 5)")
    
    return vif_result


def load_aggregated_data(tune_results_dir):
    """
    Load aggregated CSV files from tune_results directory.
    
    Args:
        tune_results_dir: Path to directory containing _parameter_analysis.csv files
    
    Returns:
        DataFrame with all aggregated data
    """
    if not os.path.exists(tune_results_dir):
        return None
    
    csv_files = [f for f in os.listdir(tune_results_dir) if f.endswith('_parameter_analysis.csv')]
    
    if not csv_files:
        return None
    
    all_data = []
    for csv_file in csv_files:
        file_path = os.path.join(tune_results_dir, csv_file)
        df = pd.read_csv(file_path)
        # Extract gt_name from filename
        gt_name = csv_file.replace('_parameter_analysis.csv', '')
        df['gt_name'] = gt_name
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description='VIF Analysis for Parameter Tuning - Detects multicollinearity among parameters'
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        default=os.path.join('plots', 'per_frame_results.csv'),
        help='Input CSV file (default: plots/per_frame_results.csv)'
    )
    args = parser.parse_args()
    
    # Define PRIMARY parameters to analyze (as specified by user)
    # These are the only parameters that should be included in VIF analysis
    primary_params = ['alpha', 'adaptiveFraction']
    temporal_params = ['temporalResidualCutoff', 'temporalResidualConfidenceScale']
    # sampleSequence is nominal/categorical, will be excluded from VIF but kept for info
    
    # Parameters for VIF analysis: alpha, adaptiveFraction, temporalSensitivity
    vif_params = ['alpha', 'adaptiveFraction'] + temporal_params
    
    # Load data
    print("Loading data...")
    
    # Try loading aggregated data from tune_results first (preferred)
    tune_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tune_results")
    df = load_aggregated_data(tune_results_dir)
    
    if df is not None:
        print(f"Loaded {len(df)} aggregated experiments from {tune_results_dir}")
    else:
        # Fall back to per_frame_results.csv
        if os.path.exists(args.csv_file):
            df = pd.read_csv(args.csv_file)
            # Filter for adaptive sampler
            df = df[df['sampler'] == 'adaptive'].copy()
            print(f"Loaded {len(df)} adaptive experiments from {args.csv_file}")
            print("NOTE: Using per-frame data. For per-experiment analysis, run parameter_tune.py first.")
        else:
            print(f"Error: Could not find data at {args.csv_file}")
            print(f"Also tried: {tune_results_dir}")
            return
    
    # Identify ground truths and sparsity levels
    ground_truths = sorted(df['gt_name'].unique())
    sparsity_levels = sorted(df['scanned_pixel_percent'].unique())
    
    print(f"Found {len(ground_truths)} ground truths: {ground_truths}")
    print(f"Found {len(sparsity_levels)} sparsity levels: {sparsity_levels}")
    
    # Analyze each bucket
    results = {}
    for gt_name in ground_truths:
        for sparsity in sparsity_levels:
            df_bucket = df[(df['gt_name'] == gt_name) & (df['scanned_pixel_percent'] == sparsity)]
            if len(df_bucket) == 0:
                continue
            
            result = analyze_bucket(df_bucket, gt_name, sparsity, vif_params)
            if result:
                results[(gt_name, sparsity)] = result
    
    # Summary across all buckets
    print(f"\n\n{'='*60}")
    print("SUMMARY ACROSS ALL BUCKETS")
    print(f"{'='*60}")
    
    all_vif_values = []
    buckets_with_high_vif = 0
    
    for (gt_name, sparsity), result in results.items():
        if result['high_vif_params_high']:
            buckets_with_high_vif += 1
            print(f"\n{gt_name} (sparsity={sparsity}): Severe VIF > 10 for {result['high_vif_params_high']}")
        
        if result['vif'] is not None:
            all_vif_values.extend(result['vif'].tolist())
    
    print(f"\nTotal buckets analyzed: {len(results)}")
    print(f"Buckets with severe multicollinearity (VIF > 10): {buckets_with_high_vif}")
    
    if all_vif_values:
        vif_array = np.array(all_vif_values)
        print(f"\nOverall VIF Statistics:")
        print(f"  Mean: {np.mean(vif_array):.2f}")
        print(f"  Median: {np.median(vif_array):.2f}")
        print(f"  Max: {np.max(vif_array):.2f}")
        print(f"  % > 5: {100 * np.sum(vif_array > 5) / len(vif_array):.1f}%")
        print(f"  % > 10: {100 * np.sum(vif_array > 10) / len(vif_array):.1f}%")


if __name__ == '__main__':
    main()
