import os

import numpy as np
from concurrent.futures import ProcessPoolExecutor

from . import evaluation
from .image_processing import generate_random_pixel_locations, compute_sample_size
from .interpolator import ImageInterpolator
from .microscope import Microscope
from .monitor import visualize_microscope_image


class RandomSampler:
    def __init__(self, interpolMethod, sparsityPercent, numberOfFrames = 1,
                 groundTruthName = "li_expulsion_one", imageShape=None):


        self.sparsityPercent = sparsityPercent
        self.interpolMethod = interpolMethod
        self.numberOfFrames = numberOfFrames


        self.microscope = Microscope(groundTruthName)
        self.imageShape = imageShape if imageShape else self.microscope.imageShape
        self.numberOfSamples = compute_sample_size(self.imageShape, sparsityPercent)

        self.validate_inputs()

        self.sampledFrames = []
        self.reconstructedFrames = []
        self.psnrs = []
        self.ssims = []

    def validate_inputs(self):
        if not isinstance(self.imageShape, (tuple, list)) or len(self.imageShape) != 2:
            raise ValueError("Image shape must be a tuple or list of two integers.")
        if not all(isinstance(x, int) and x > 0 for x in self.imageShape):
            raise ValueError("Image dimensions must be positive integers.")

        if not isinstance(self.sparsityPercent, (int, float)):
            raise ValueError("Sparsity percent must be a number.")
        if not (0 <= self.sparsityPercent <= 100):
            raise ValueError("Sparsity percent must be between 0 and 100.")

        # interpolMethod validation
        valid_methods = {"linear", "nearest", "cubic"}
        if not isinstance(self.interpolMethod, str) or self.interpolMethod not in valid_methods:
            raise ValueError(f"Interpolation method must be one of {valid_methods}.")

    def get_coordinates(self):
        yCoords, xCoords = generate_random_pixel_locations(self.imageShape, self.sparsityPercent)

        H, W = self.imageShape
        yCoords = np.clip(yCoords, 0, H - 1)
        xCoords = np.clip(xCoords, 0, W - 1)

        return yCoords, xCoords


    def process_frame(self, frameNumber):
        yCoords, xCoords = self.get_coordinates()
        sampledImage = self.microscope.sample_image(yCoords, xCoords, frameNumber)
        pixelIntensities = sampledImage[yCoords, xCoords]

        knownPoints = np.column_stack((xCoords, yCoords))
        imageInterpolator = ImageInterpolator(self.imageShape, knownPoints, pixelIntensities, self.interpolMethod)
        reconstructedImage = imageInterpolator.interpolate_image()
        reconstructedImage = np.clip(reconstructedImage, 0, 255).astype(np.uint8)

        psnr = evaluation.calculate_psnr(self.microscope.groundTruthVideo[frameNumber], reconstructedImage)
        ssim = evaluation.calculate_ssim(reconstructedImage, self.microscope.groundTruthVideo[frameNumber])

        return sampledImage, reconstructedImage, psnr, ssim


    def run(self):
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self.process_frame, range(self.numberOfFrames)))

        self.sampledFrames, self.reconstructedFrames, self.psnrs, self.ssims = zip(*results)
        return np.array(self.reconstructedFrames), list(self.psnrs), list(self.ssims)


    def show_figures(self, frameNumber=4):
        if frameNumber >= len(self.reconstructedFrames):
            raise IndexError("Frame number exceeds number of reconstructed frames.")

        sampled = self.sampledFrames[frameNumber]
        reconstructed = self.reconstructedFrames[frameNumber]

        print(f"Displaying results for frame {frameNumber}")
        print(f"PSNR: {self.psnrs[frameNumber]:.2f}, SSIM: {self.ssims[frameNumber]:.4f}")

        visualize_microscope_image(sampled, imageTitle="Sampled Image (Random Sampling)")
        visualize_microscope_image(reconstructed, imageTitle="Reconstructed Image")

    # UPDATED: Uses visualize_microscope_image with savePlot
    def save_figures(self, frameNumber=4, save_path="plots"):
        if frameNumber >= len(self.reconstructedFrames):
            raise IndexError("Frame number exceeds number of reconstructed frames.")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        sampled = self.sampledFrames[frameNumber]
        reconstructed = self.reconstructedFrames[frameNumber]

        sampled_path = os.path.join(save_path, f"frame_{frameNumber}_sampled.png")
        recon_path = os.path.join(save_path, f"frame_{frameNumber}_reconstructed.png")

        visualize_microscope_image(sampled, imageTitle="Sampled Image (Random Sampling)",
                                   savePlot=True, savePath=sampled_path)
        visualize_microscope_image(reconstructed, imageTitle="Reconstructed Image",
                                   savePlot=True, savePath=recon_path)
