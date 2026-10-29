import os

from video_dynamics_analysis import analyze_all_ground_truths, plot_dynamics_scatter

# When False, the entire calculation stack runs without any noise
# correction: no noise model, no change cut. Output files are suffixed
# "_uncorrected" so both modes can be compared side by side.
NOISE_CORRECTION = True

# Optional display names for the plots: ground truth name -> display name
# (used verbatim; names without an entry keep the in-code name)
LABELS = {
    # "LI_EXPULSION_ONE_ORIGINAL": "Li expulsion I",
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
