import cv2
import numpy as np
from .config import SI_LITHIATION_ONE, LI_EXPULSION_ONE, LI_EXPULSION_TWO, HYDRATION_ONE
from .image_processing import gaussian_kernel_2d


class Microscope:
    SUPPORTED_GROUND_TRUTHS = {
        "si_lithiation_one": SI_LITHIATION_ONE,
        "li_expulsion_one": LI_EXPULSION_ONE,
        "li_expulsion_two": LI_EXPULSION_TWO,
        "hydration_one": HYDRATION_ONE,
    }

    def __init__(self, groundTruthName):

        self.perPixelDwellTime = np.float32(20)

        if groundTruthName not in self.SUPPORTED_GROUND_TRUTHS:
            raise ValueError(
                f"Unsupported groundTruthName: {groundTruthName}. "
                f"Supported: {list(self.SUPPORTED_GROUND_TRUTHS.keys())}"
            )
        self.groundTruthVideo  = self.SUPPORTED_GROUND_TRUTHS[groundTruthName]

        if self.groundTruthVideo is None or len(self.groundTruthVideo) == 0:
            raise ValueError(f"Ground truth video for '{groundTruthName}' is not loaded or empty.")

        # Automatically determine meta-data from the ground truth videos
        self.imageShape = self.groundTruthVideo[0].shape
        self.dataType = self.groundTruthVideo[0].dtype
        info = np.iinfo(self.dataType)
        self.maxValue = info.max

    def sample_image(self, yCoords, xCoords, frameNumber, sigma=None):
        previousFrame = self.groundTruthVideo[frameNumber-1]
        nextFrame = self.groundTruthVideo[frameNumber]

        H, W = previousFrame.shape

        # STEP 1: Build the merged frame
        sampled = previousFrame.astype(np.float32).copy()
        sampled[yCoords, xCoords] = nextFrame[yCoords, xCoords].astype(np.float32)

        # STEP 2: Determine sigma
        if sigma is None:
            sigma = 1.0 / np.sqrt(self.perPixelDwellTime)

        # STEP 3: Add noise across the ENTIRE image
        noise = np.random.normal(0, sigma, size=(H, W)).astype(np.float32)

        if sigma > 0:
            kernel = gaussian_kernel_2d(sigma)
            noise = cv2.filter2D(noise, -1, kernel)

        # STEP 4: Apply noise globally
        sampled += noise

        return np.clip(sampled, 0, self.maxValue).astype(self.dataType)
