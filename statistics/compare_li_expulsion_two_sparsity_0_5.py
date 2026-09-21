#!/usr/bin/env python3
"""
Specific comparison script for LI_EXPULSION_TWO_ORIGINAL at sparsity 0.5

Compares 6 experiment configurations with unique parameter combinations.

All with: scanned_pixel_percent=0.5, temporalMethod=temporal_variance
"""

import os
import sys

# Add current directory to path for importing experiment_comparison
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_comparison import compare_experiments


def main():
    # All configs share these parameters
    base_config = {
        "sampler_type": "adaptive",
        "interpol_method": "cubic",
        "scanned_pixel_percent": 0.5,
        "gt_name": "LI_EXPULSION_TWO_ORIGINAL",
        "has_temporal_sampler": True,
        "has_temporal_reconstruction": True,
        "minDensityGamma": 0.1,
        "temporal_method": "temporal_variance"
    }
    
    # Define the 6 configurations with unique parameter combinations
    configs = [
        # Config 1
        {
            **base_config,
            "alpha": 5.0,
            "adaptive_fraction": 0.5,
            "sample_sequence": "halton",
            "temporal_residual_cutoff": 12.0,
            "temporal_residual_confidence_scale": 250.0
        },
        # Config 2
        {
            **base_config,
            "alpha": 3.0,
            "adaptive_fraction": 0.3,
            "sample_sequence": "halton",
            "temporal_residual_cutoff": 12.0,
            "temporal_residual_confidence_scale": 100.0
        },
        # Config 3
        {
            **base_config,
            "alpha": 5.0,
            "adaptive_fraction": 0.5,
            "sample_sequence": "halton",
            "temporal_residual_cutoff": 25.0,
            "temporal_residual_confidence_scale": 250.0
        },
        # Config 4
        {
            **base_config,
            "alpha": 0.25,
            "adaptive_fraction": 0.0,
            "sample_sequence": "uniform",
            "temporal_residual_cutoff": 25.0,
            "temporal_residual_confidence_scale": 250.0
        },
        # Config 5
        {
            **base_config,
            "alpha": 0.25,
            "adaptive_fraction": 0.1,
            "sample_sequence": "stratified",
            "temporal_residual_cutoff": 25.0,
            "temporal_residual_confidence_scale": 250.0
        },
        # Config 6
        {
            **base_config,
            "alpha": 0.25,
            "adaptive_fraction": 0.0,
            "sample_sequence": "halton",
            "temporal_residual_cutoff": 25.0,
            "temporal_residual_confidence_scale": 100.0
        }
    ]
    
    # Generate output path
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "comparison_li_expulsion_two_sparsity_0_5_reconstruction_with_gt.tiff"
    )
    
    print(f"Comparing {len(configs)} configurations for LI_EXPULSION_TWO_ORIGINAL at sparsity 0.5...")
    print(f"Output will be saved to: {output_path}")
    print("Including both noise-equivalent (sparsity=0.5) and full-scan ground truths")
    
    # Run comparison with reconstruction result type and both GTs
    result_path = compare_experiments(
        experiment_configs=configs,
        result_type="reconstruction",
        gt_name="LI_EXPULSION_TWO_ORIGINAL",
        noise_equivalent_sparsity=0.5,  # Noise-equivalent GT at 0.5 sparsity
        include_full_gt=True,             # Include full-scan GT
        output_path=output_path,
        base_dir="plots"
    )
    
    print(f"Done! Composite TIF saved to: {result_path}")
    
    # Print config summary
    print("\nConfigurations compared:")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. alpha={config['alpha']}, adaptive={config['adaptive_fraction']}, "
              f"sample={config['sample_sequence']}, cutoff={config['temporal_residual_cutoff']}, "
              f"confidence={config['temporal_residual_confidence_scale']}")


if __name__ == '__main__':
    main()
