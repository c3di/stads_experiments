"""Restore acquisition noise to the AI-upsampled Li_Expulsion stacks.

`Li_Expulsion_1.tif` is the real acquisition: 20 frames at 1018x1510. The
`x10` and `x50` stacks were produced from it by an AI model that binned 2x2
and interpolated 9 (resp. 49) synthetic frames between each pair of real
ones. Measured in a flat region, the real acquisition carries about 3.0 grey
levels of noise and the upsampled stacks about 2.2 -- and the interpolated
frames barely differ from their neighbours, where consecutive real frames
differ substantially. A sampling experiment run on them therefore sees a
specimen both quieter and far more temporally placid than the microscope
delivers, which flatters every reconstruction metric and the temporal
machinery most of all.

This script measures the real stack and adds independent white Gaussian noise
to every upsampled frame to bring it back to that level, writing corrected
copies beside the originals. Run it once:

    python correct_upsampled_noise.py

Two deliberate simplifications. The noise is modelled as additive and white,
where the real noise is correlated over two to three pixels and its sigma
varies with brightness. And the target is the *unbinned* level, so the
corrected stacks carry the noise of the acquisition they came from rather
than the lower level binning really produces -- which is what lets the rest
of the pipeline stay unaware that these stacks are binned at all.
"""
import os

import numpy as np
import tifffile

from stads.video_downloader import DEFAULT_SAVE_DIR

#: The real acquisition every target is measured from.
ORIGINAL = "Li_Expulsion_1.tif"

#: filename -> the stride at which its frames are real rather than
#: interpolated. Each stack holds the original's 20 frames, so the stride is
#: (frames - 1) / 19 and frames 0, stride, 2*stride, ... are the real ones.
UPSAMPLED = {
    "Li_Expulsion_1x10.tif": 10,
    "Li_Expulsion_1x10_square.tif": 10,
    "Li_Expulsion_1_x50.tif": 50,
}

SUFFIX = "_noise_corrected"

#: (top, left, bottom, right) of a uniform patch of background in the
#: original's own pixels, picked by eye and well clear of both ends of the
#: intensity range. Chosen rather than searched: ranking patches by how
#: little they vary picks quiet *texture* as readily as true background, and
#: reads about 1.6x high when it does.
FLAT_REGION = (150, 865, 321, 1055)


def flat_sigma(image):
    """The noise sigma of a flat `image` patch, in its own units.

    Plain standard deviation after subtracting a fitted plane, which removes
    the gentle shading the patch carries without touching the noise.

    A Laplacian estimator (Immerkaer) would need no flat region at all, but
    it assumes white noise and this dataset's is correlated over two to three
    pixels: on synthetic noise of known amplitude it under-reads by 1.4x at a
    correlation length of 2 pixels and 2.0x at 3.5, and on these frames it
    reports barely a third of the true sigma.
    """
    image = image.astype(np.float32)
    rows, columns = np.mgrid[:image.shape[0], :image.shape[1]]
    design = np.c_[rows.ravel(), columns.ravel(), np.ones(image.size)]
    plane, *_ = np.linalg.lstsq(design, image.ravel(), rcond=None)
    return float((image.ravel() - design @ plane).std())


def median_sigma(frames, region):
    """Median flat_sigma of `region` over `frames`.

    Median because frame 0 of the real acquisition reads about a third above
    every other frame -- its spectrum is flat to Nyquist where the rest roll
    off, so it is the one frame that reaches us unsmoothed.
    """
    top, left, bottom, right = region
    return float(np.median([flat_sigma(frame[top:bottom, left:right])
                            for frame in frames]))


def bin2(frame):
    """`frame` averaged 2x2, the binning the upsampled stacks were made with."""
    height, width = frame.shape[-2:]
    frame = frame[..., :height // 2 * 2, :width // 2 * 2]
    return frame.reshape(*frame.shape[:-2], height // 2, 2, width // 2, 2
                         ).mean(axis=(-3, -1))


def crop_offset(binnedOriginal, frame):
    """Where `frame` sits in the binned original, as (top, left).

    The upsampled stacks are cropped as well as binned, and not all to the
    same window -- the square one is inset 120 binned pixels from the left --
    so the flat region has to be located per stack rather than just halved.
    """
    height, width = frame.shape
    best = None
    for top in range(binnedOriginal.shape[0] - height + 1):
        for left in range(binnedOriginal.shape[1] - width + 1):
            error = np.mean((binnedOriginal[top:top + height,
                                            left:left + width] - frame) ** 2)
            if best is None or error < best[0]:
                best = (error, top, left)
    return best[1], best[2]


def binned_region(offset):
    """FLAT_REGION in the pixels of a stack cropped at `offset`."""
    top, left, bottom, right = FLAT_REGION
    offsetTop, offsetLeft = offset
    return (top // 2 - offsetTop, left // 2 - offsetLeft,
            bottom // 2 - offsetTop, right // 2 - offsetLeft)


def added_sigma(target, present):
    """The sigma to add to reach `target` given `present` already there.

    Independent noise adds in quadrature, so this is a leg of a right
    triangle, not a difference. A stack already at the target gets nothing.
    """
    return float(np.sqrt(max(target ** 2 - present ** 2, 0.0)))


def corrected_path(filename):
    stem, extension = os.path.splitext(filename)
    return os.path.join(str(DEFAULT_SAVE_DIR), stem + SUFFIX + extension)


def correct_stack(filename, stride, target, binnedOriginal, rng):
    """Write the noise-corrected copy of one upsampled stack."""
    source = os.path.join(str(DEFAULT_SAVE_DIR), filename)
    destination = corrected_path(filename)

    with tifffile.TiffFile(source) as handle:
        frameCount = len(handle.pages)
        realIndices = set(range(0, frameCount, stride))

        first = handle.pages[0].asarray().astype(np.float32)
        region = binned_region(crop_offset(binnedOriginal, first))

        # The two classes are measured apart: interpolating two frames
        # averages their noise as well as their content, so a synthetic frame
        # starts quieter than the real ones it sits between.
        realFrames = [handle.pages[i].asarray().astype(np.float32)
                      for i in sorted(realIndices)]
        synthFrames = [handle.pages[i].asarray().astype(np.float32)
                       for i in range(frameCount) if i not in realIndices][:20]

        realSigma = median_sigma(realFrames, region)
        synthSigma = median_sigma(synthFrames, region)
        addReal = added_sigma(target, realSigma)
        addSynth = added_sigma(target, synthSigma)

        print(f"\n{filename}  {frameCount} frames, every {stride}th real")
        print(f"  flat region here  {region}")
        print(f"  real       present {realSigma:.3f}   add {addReal:.3f}")
        print(f"  synthetic  present {synthSigma:.3f}   add {addSynth:.3f}")

        with tifffile.TiffWriter(destination) as writer:
            for index, page in enumerate(handle.pages):
                frame = page.asarray().astype(np.float32)
                sigma = addReal if index in realIndices else addSynth
                frame += rng.normal(0.0, sigma, frame.shape)
                writer.write(np.clip(frame, 0, 255).astype(np.uint8),
                             contiguous=True)

    print(f"  -> {os.path.basename(destination)} "
          f"({os.path.getsize(destination) / 1e6:.0f} MB)")


def main():
    rng = np.random.default_rng(20260917)

    original = tifffile.imread(os.path.join(str(DEFAULT_SAVE_DIR), ORIGINAL)
                               ).astype(np.float32)
    target = median_sigma(original, FLAT_REGION)

    print(f"{ORIGINAL}: {original.shape}")
    print(f"  flat region {FLAT_REGION}")
    print(f"  target sigma {target:.3f} grey levels")
    print("  per frame: " + " ".join(
        f"{flat_sigma(f[FLAT_REGION[0]:FLAT_REGION[2], FLAT_REGION[1]:FLAT_REGION[3]]):.2f}"
        for f in original))

    binnedOriginal = bin2(original[0])
    for filename, stride in UPSAMPLED.items():
        correct_stack(filename, stride, target, binnedOriginal, rng)


if __name__ == "__main__":
    main()
