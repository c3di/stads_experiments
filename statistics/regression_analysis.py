#!/usr/bin/env python3
"""
Standardized Multiple Regression Analysis for Parameter Tuning

Implements the screening stage analysis for full factorial experimental designs.
For each gt_name x scanned_pixel_percent bucket, fits standardized regression
models to identify parameter importance, interactions, and non-linear effects.

Goals:
- Rank parameters by impact on PSNR/SSIM (standardized coefficients)
- Detect parameter interactions
- Identify non-linear relationships

Updated Approach (2026-09-18):
- Response variables: psnr_mean, ssim_mean, combined_score (3 separate models per bucket)
- combined_score = psnr_rank^2 + ssim_rank^2 (lower is better, rewards balanced performance)
- Interaction term: temporalResidualCutoff * temporalResidualConfidenceScale added as predictor
- Response variables are NEVER used as predictors

Usage:
    python regression_analysis.py

Output:
    - Standardized regression coefficients per bucket
    - Parameter rankings by effect size
    - All output saved to regression_analysis.txt
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

# Output file path
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regression_analysis.txt')

# Parameters to include in analysis
# minDensityGamma is fixed in current experiments, so excluded
CONTINUOUS_PARAMS = ['alpha', 'adaptiveFraction', 'temporalResidualCutoff', 'temporalResidualConfidenceScale']
CATEGORICAL_PARAMS = ['sampleSequence']
RESPONSE_PARAMS = ['psnr_mean', 'ssim_mean']

# Interaction terms to include (product of continuous parameters)
INTERACTION_TERMS = [
    ('temporalResidualCutoff', 'temporalResidualConfidenceScale')
]

# Sample size threshold
MIN_SAMPLES = 5


def standardize_series(series):
    """Manually standardize a series (z-score normalization)."""
    if len(series) == 0:
        return series
    mean = np.mean(series)
    std = np.std(series, ddof=1)
    if std == 0:
        return series - mean  # Avoid division by zero
    return (series - mean) / std


# Try to import statsmodels, but provide fallback
TRY_STATSMODELS = True  # Disabled since not available
STATSMODELS_AVAILABLE = True


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


def prepare_data(df_bucket):
    """
    Prepare data for regression analysis:
    - Select only predictor columns (continuous + categorical)
    - Standardize continuous parameters
    - Dummy-code categorical parameters
    - Compute combined_score as a SEPARATE response variable (not a predictor)
    - Return: predictor_df, response_series_dict, ranks_dict
    
    The key fix: Response variables (psnr_mean, ssim_mean, combined_score) 
    are NEVER included in the predictor DataFrame.
    """
    if len(df_bucket) == 0:
        return None, None, None
    
    # Select only predictor parameters that exist in the data
    available_cont = [p for p in CONTINUOUS_PARAMS if p in df_bucket.columns]
    available_cat = [p for p in CATEGORICAL_PARAMS if p in df_bucket.columns]
    available_resp = [p for p in RESPONSE_PARAMS if p in df_bucket.columns]
    
    if not available_cont or not available_resp:
        return None, None, None
    
    # Drop rows with NaN in response variables
    df_clean = df_bucket.dropna(subset=available_resp)
    
    if len(df_clean) == 0:
        return None, None, None
    
    # ========================================================================
    # PREPARE PREDICTORS (only continuous + categorical, NO response variables)
    # ========================================================================
    predictor_df = df_clean[available_cont + available_cat].copy()
    
    # Add interaction terms
    for (param1, param2) in INTERACTION_TERMS:
        if param1 in predictor_df.columns and param2 in predictor_df.columns:
            interaction_col = f"{param1}_x_{param2}"
            predictor_df[interaction_col] = predictor_df[param1] * predictor_df[param2]
            # Add to available_cont so it gets standardized
            available_cont.append(interaction_col)
    
    # Standardize continuous parameters (including interaction terms)
    for col in available_cont:
        if col in predictor_df.columns:
            predictor_df[col] = standardize_series(predictor_df[col])
    
    # Dummy-code categorical parameters (drop first to avoid multicollinearity)
    if available_cat:
        predictor_df = pd.get_dummies(predictor_df, columns=available_cat, drop_first=True)
    
    # ========================================================================
    # PREPARE RESPONSE VARIABLES AND RANKS
    # ========================================================================
    # Response variables: psnr_mean, ssim_mean, and combined_score
    response_series = {}
    ranks = {}
    
    for resp in available_resp:
        response_series[resp] = df_clean[resp].copy()
        # Rank experiments by this response variable (1 = best = highest value)
        ranks[f'{resp}_rank'] = df_clean[resp].rank(method='min', ascending=False)
    
    # Compute combined_score as a RESPONSE variable (not a predictor)
    # combined_score = psnr_rank^2 + ssim_rank^2 (lower is better)
    if 'psnr_mean_rank' in ranks and 'ssim_mean_rank' in ranks:
        combined_score = ranks['psnr_mean_rank']**2 + ranks['ssim_mean_rank']**2
        response_series['combined_score'] = combined_score
        ranks['combined_score'] = combined_score  # Store for reference
    elif 'psnr_mean_rank' in ranks:
        combined_score = ranks['psnr_mean_rank']**2
        response_series['combined_score'] = combined_score
        ranks['combined_score'] = combined_score
    elif 'ssim_mean_rank' in ranks:
        combined_score = ranks['ssim_mean_rank']**2
        response_series['combined_score'] = combined_score
        ranks['combined_score'] = combined_score
    
    return predictor_df, response_series, ranks


def fit_regression(df, response_var, response_values):
    """
    Fit standardized regression model.
    Uses statsmodels if available, otherwise pure numpy.
    
    Args:
        df: DataFrame with standardized predictors and dummy-coded categoricals
        response_var: Response variable name (e.g., 'psnr_mean')
        response_values: Series or array of response values
    
    Returns:
        Dictionary with regression results, or None if failed
    """
    if len(df) == 0 or len(response_values) == 0:
        return None
    
    # Get all predictors (continuous + dummy categoricals)
    predictor_cols = [col for col in df.columns]
    
    if len(predictor_cols) == 0:
        return None
    
    # Prepare data - convert all to float
    X = df[predictor_cols].astype(float).values
    y = response_values.astype(float).values
    
    # Add constant for intercept
    X_with_intercept = np.column_stack([np.ones(len(y)), X])
    
    try:
        if STATSMODELS_AVAILABLE and TRY_STATSMODELS:
            # Use statsmodels for better statistics
            import statsmodels.api as sm
            model = sm.OLS(y, X_with_intercept)
            result = model.fit()
            
            # Extract results (excluding intercept at index 0)
            coeffs = result.params[1:]
            pvalues = result.pvalues[1:]
            std_errors = result.bse[1:]
            r_squared = result.rsquared
            adj_r_squared = result.rsquared_adj
            n_obs = result.nobs
        else:
            # Pure numpy implementation
            # OLS: beta = (X^T X)^-1 X^T y
            XtX = np.dot(X_with_intercept.T, X_with_intercept)
            Xty = np.dot(X_with_intercept.T, y)
            
            # Check if matrix is singular
            try:
                # Check condition number
                det = np.linalg.det(XtX)
                if np.isclose(det, 0):
                    warnings.warn("Matrix near-singular (det ≈ 0), cannot compute OLS")
                    return None
                
                coeffs_full = np.linalg.solve(XtX, Xty)
            except (np.linalg.LinAlgError, ValueError):
                warnings.warn("Matrix singular, cannot compute OLS")
                return None
            
            # Separate intercept and predictor coefficients
            intercept = coeffs_full[0]
            coeffs = coeffs_full[1:]
            
            # Calculate predicted values and residuals
            y_pred = np.dot(X_with_intercept, coeffs_full)
            residuals = y - y_pred
            
            # Calculate R-squared
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum(residuals ** 2)
            r_squared = 1 - (ss_res / ss_total) if ss_total > 0 else 0
            
            # Calculate adjusted R-squared
            n_obs = len(y)
            p = len(coeffs) + 1  # +1 for intercept
            adj_r_squared = 1 - ((1 - r_squared) * (n_obs - 1) / (n_obs - p)) if n_obs > p else r_squared
            
            # Calculate standard errors
            mse = ss_res / (n_obs - p) if (n_obs - p) > 0 else 0
            var_cov = mse * np.linalg.inv(XtX)
            std_errors = np.sqrt(np.diag(var_cov))[1:]  # Exclude intercept
            
            # Calculate p-values (t-test: beta / se ~ t(df_residual))
            t_stats = coeffs / std_errors if np.any(std_errors > 0) else np.ones_like(coeffs) * np.nan
            pvalues = [2 * (1 - stats.t.cdf(abs(t), df=n_obs - p)) for t in t_stats]
            
            n_obs = len(y)
        
        # Create results dictionary
        reg_result = {
            'coefficients': coeffs,
            'pvalues': np.array(pvalues),
            'std_errors': std_errors,
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'n_obs': n_obs,
            'predictors': predictor_cols,
            'response': response_var
        }
        
        return reg_result
        
    except Exception as e:
        warnings.warn(f"Regression failed: {e}")
        return None


def print_bucket_results(df_bucket, gt_name, sparsity, results_dict, ranks):
    """
    Print formatted results for a bucket.
    
    Args:
        df_bucket: Original DataFrame for the bucket (for size info)
        gt_name: Ground truth name
        sparsity: Sparsity level
        results_dict: Dictionary of regression results keyed by response variable
        ranks: Dictionary of rank information
    """
    print(f"\n{'='*70}")
    print(f"Bucket: {gt_name} | scanned_pixel_percent = {sparsity}")
    print(f"Experiments: {len(df_bucket)}")
    print(f"{'='*70}")
    
    if not results_dict:
        print("No results to display")
        return
    
    # Print results for each response variable
    response_order = ['psnr_mean', 'ssim_mean', 'combined_score']
    
    for resp_var in response_order:
        result = results_dict.get(resp_var)
        if result is None:
            continue
        
        # For combined_score, note that lower is better
        if resp_var == 'combined_score':
            print(f"\n{resp_var.replace('_', ' ').title()} Model (R^2 = {result['r_squared']:.4f}, Adj R^2 = {result['adj_r_squared']:.4f}):")
            print(f"  (Response: sum of squared ranks for PSNR and SSIM, lower is better)")
        else:
            print(f"\n{resp_var.replace('_', ' ').title()} Model (R^2 = {result['r_squared']:.4f}, Adj R^2 = {result['adj_r_squared']:.4f}):")
        
        print(f"-" * 70)
        print(f"{'Parameter':<30} {'Coeff':<12} {'Std Err':<12} {'p-value':<12} {'|Coeff| Rank':<10}")
        print(f"-" * 70)
        
        # Combine coefficients, pvalues, std_errors
        coeff_data = list(zip(
            result['predictors'],
            result['coefficients'],
            result['std_errors'],
            result['pvalues']
        ))
        
        # Sort by absolute coefficient for ranking
        coeff_data_sorted = sorted(coeff_data, key=lambda x: abs(x[1]), reverse=True)
        
        for i, (param, coeff, std_err, pval) in enumerate(coeff_data_sorted, 1):
            pval_str = f"{pval:.4f}" if pval >= 0.001 else f"{pval:.2e}"
            print(f"{param:<30} {coeff:<12.4f} {std_err:<12.4f} {pval_str:<12} {i:<10}")
        
        # Note for combined_score: negative coefficients are GOOD
        if resp_var == 'combined_score':
            print(f"\n  Note: Negative coefficients improve performance (lower combined_score = better)")
    
    # Print parameter importance ranking (combined across all models)
    print(f"\nParameter Importance (Average |Coeff| across all models):")
    print(f"-" * 70)
    
    # Get all unique parameters across all models
    all_params = set()
    all_coeffs = {}  # param -> list of coefficients
    
    for resp_var in response_order:
        result = results_dict.get(resp_var)
        if result:
            for param, coeff in zip(result['predictors'], result['coefficients']):
                all_params.add(param)
                if param not in all_coeffs:
                    all_coeffs[param] = []
                all_coeffs[param].append(abs(coeff))
    
    # Compute average |coeff| for each parameter
    avg_coeffs = {}
    for param in all_params:
        avg_coeffs[param] = np.mean(all_coeffs[param])
    
    # Sort by average |coeff|
    sorted_by_avg = sorted(avg_coeffs.items(), key=lambda x: x[1], reverse=True)
    
    for i, (param, avg_coeff) in enumerate(sorted_by_avg, 1):
        print(f"{i}. {param:<30} Avg |Coeff| = {avg_coeff:.4f}")


def analyze_bucket(df_bucket, gt_name, sparsity):
    """
    Analyze a single gt_name x scanned_pixel_percent bucket.
    Fits 3 separate models: PSNR, SSIM, and combined_score.
    """
    if len(df_bucket) < MIN_SAMPLES:
        print(f"\n{'='*70}")
        print(f"Bucket: {gt_name} | scanned_pixel_percent = {sparsity}")
        print(f"Experiments: {len(df_bucket)} - SKIPPED (below minimum {MIN_SAMPLES})")
        print(f"{'='*70}")
        return None
    
    # Prepare data - this now properly separates predictors from responses
    predictor_df, response_series, ranks = prepare_data(df_bucket)
    
    if predictor_df is None or response_series is None:
        print(f"\n{'='*70}")
        print(f"Bucket: {gt_name} | scanned_pixel_percent = {sparsity}")
        print(f"Experiments: {len(df_bucket)} - No valid data")
        print(f"{'='*70}")
        return None
    
    # Fit regression for each response variable
    # 3 models: psnr_mean, ssim_mean, combined_score
    results_dict = {}
    
    available_responses = list(response_series.keys())
    
    for resp_var in available_responses:
        result = fit_regression(predictor_df, resp_var, response_series[resp_var])
        if result:
            results_dict[resp_var] = result
    
    # Print results
    print_bucket_results(df_bucket, gt_name, sparsity, results_dict, ranks)
    
    return {
        'results': results_dict,
        'n_obs': len(df_bucket)
    }


def main():
    # Redirect stdout to file
    original_stdout = sys.stdout
    
    with open(OUTPUT_FILE, 'w') as f:
        sys.stdout = f
        try:
            _main_inner()
        finally:
            sys.stdout = original_stdout
    
    # Print file contents to console
    with open(OUTPUT_FILE, 'r') as f:
        print(f.read(), end='')


def _main_inner():
    """Inner main function that does the actual work"""
    print("="*70)
    print("STANDARDIZED REGRESSION ANALYSIS FOR PARAMETER TUNING")
    print("="*70)
    print(f"\nCorrected Approach:")
    print(f"  - Response variables: psnr_mean, ssim_mean, combined_score (3 models per bucket)")
    print(f"  - combined_score = psnr_rank^2 + ssim_rank^2 (lower is better)")
    print(f"  - Interaction term: temporalResidualCutoff * temporalResidualConfidenceScale")
    print(f"  - Response variables are NEVER used as predictors")
    print(f"\nLoading data...")
    
    # Load aggregated data from tune_results
    tune_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tune_results")
    df = load_aggregated_data(tune_results_dir)
    
    if df is None:
        print(f"Error: Could not load data from {tune_results_dir}")
        print("Please run parameter_tune.py --analyze first to generate aggregated CSV files.")
        return
    
    print(f"Loaded {len(df)} aggregated experiments from {tune_results_dir}")
    
    # Identify ground truths and sparsity levels
    ground_truths = sorted(df['gt_name'].unique())
    sparsity_levels = sorted(df['scanned_pixel_percent'].unique())
    
    print(f"Found {len(ground_truths)} ground truths: {ground_truths}")
    print(f"Found {len(sparsity_levels)} sparsity levels: {sparsity_levels}")
    print(f"\nContinuous parameters: {CONTINUOUS_PARAMS}")
    print(f"Categorical parameters: {CATEGORICAL_PARAMS}")
    print(f"Interaction terms: {INTERACTION_TERMS}")
    print(f"Response variables: {RESPONSE_PARAMS} + combined_score")
    
    # Analyze each bucket
    results = {}
    for gt_name in ground_truths:
        for sparsity in sparsity_levels:
            df_bucket = df[(df['gt_name'] == gt_name) & (df['scanned_pixel_percent'] == sparsity)]
            if len(df_bucket) == 0:
                continue
            
            result = analyze_bucket(df_bucket, gt_name, sparsity)
            if result:
                results[(gt_name, sparsity)] = result
    
    # Summary across all buckets
    print(f"\n\n{'='*70}")
    print("SUMMARY ACROSS ALL BUCKETS")
    print(f"{'='*70}")
    print(f"\nTotal buckets analyzed: {len(results)}")
    print(f"Minimum samples per bucket: {MIN_SAMPLES}")
    print(f"\nNote: For each bucket, 3 models were fitted:")
    print(f"  1. PSNR model (response = psnr_mean)")
    print(f"  2. SSIM model (response = ssim_mean)")
    print(f"  3. Combined Score model (response = psnr_rank^2 + ssim_rank^2)")


if __name__ == '__main__':
    main()
