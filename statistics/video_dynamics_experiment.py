import os

from video_dynamics_analysis import (
    analyze_all_ground_truths,
    plot_dynamics_scatter,
    plot_dynamics_side_by_side,
)

# When False, the entire calculation stack runs without any noise
# correction: no noise model, no change cut. Output files are suffixed
# "_uncorrected" so both modes can be compared side by side.
NOISE_CORRECTION = True

# Optional display names for the plots: ground truth name -> display name
# (used verbatim; names without an entry keep the in-code name)
LABELS = {
    "LI_EXPULSION_ONE_ORIGINAL": "Li expulsion 1",
    "LI_EXPULSION_ONE_10FPS": "Li expulsion 1 (10 fps)",
    "LI_EXPULSION_TWO_ORIGINAL": "Li expulsion 2",
    "LI_EXPULSION_TWO_10FPS": "Li expulsion 2 (10 fps)",
    "EDS_AEROSPACE_ONE_downscale": "EDS aerospace",
    "EDS_AEROSPACE_ONE_10FPS_downscale": "EDS aerospace (10 fps)",
    "TITANIUM_STRAIN_ONE_downscale": "Titanium strain",
    # Entries for ground truths currently commented out in GROUNDTRUTH_MAP:
    # "HYDRATION_ONE": "Hydration 1",
    # "LI_EXPULSION_ONE_50FPS": "Li expulsion 1 (50 fps)",
    # "SI_LITHIATION_ONE": "Si lithiation 1",
    # "EDS_AEROSPACE_TWO_downscale": "EDS aerospace 2 (half res)",
    # "EDS_AEROSPACE_TWO_10FPS_downscale": "EDS aerospace 2 (10 fps, half res)",
}

# Anchor the output to this script's folder so the results land in
# <repo>/statistics/video_dynamics/ regardless of the working directory
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "video_dynamics"
)

results = analyze_all_ground_truths(
    noise_factor=(3.0 if NOISE_CORRECTION else None)
)

suffix = "" if NOISE_CORRECTION else "_uncorrected"
plot_dynamics_scatter(results=results, show=False, labels=LABELS,
                      output_path=os.path.join(OUTPUT_DIR, f"results{suffix}.png"))
plot_dynamics_scatter(results=results, relative=True, show=False, labels=LABELS,
                      output_path=os.path.join(OUTPUT_DIR, f"results{suffix}_relative.png"))
plot_dynamics_side_by_side(results=results, show=False, labels=LABELS,
                           output_path=os.path.join(OUTPUT_DIR, f"results{suffix}_side_by_side.png"))
