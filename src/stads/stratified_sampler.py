import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from multiprocessing import Pool, cpu_count

from . import evaluation
from .delaunay_helpers import create_2d_triangulation
from .image_processing import rasterize_coordinates
from .interpolator import ImageInterpolator, DelaunayInterpolator
from .microscope import Microscope
from .monitor import visualize_microscope_image
from .stratified_sampler_helpers import sample_child, direct_child_buckets, is_leaf, random_rejection_sampling, \
    deterministic_sample_counts, non_deterministic_sample_counts
from .utility_functions import compute_sample_size, calculate_area_fractions_of_buckets


class StratifiedSampler:
    def __init__(self, interpolMethod, sparsityPercent, numberOfFrames=1, groundTruthName="li_expulsion_one"):


        self.sparsityPercent = sparsityPercent
        self.interpolMethod = interpolMethod
        self.numberOfFrames = numberOfFrames

        self.microscope = Microscope(groundTruthName)
        self.imageShape = self.microscope.imageShape
        self.numberOfSamples = compute_sample_size(self.imageShape, sparsityPercent)
        self.parentBucket = np.array([[0, 0], [self.imageShape[0] - 1, self.imageShape[1] - 1]])

        self.validate_inputs()

        self.ssims = []
        self.psnrs = []
        self.sampledFrames = []
        self.reconstructedFrames = []


    def validate_inputs(self):
        if not isinstance(self.imageShape, (tuple, list)) or len(self.imageShape) != 2:
            raise ValueError("Image shape must be a tuple or list of two integers.")
        if not all(isinstance(x, int) and x > 0 for x in self.imageShape):
            raise ValueError("Image dimensions must be positive integers.")
        if not (0 <= self.sparsityPercent <= 100):
            raise ValueError("Sparsity percent must be between 0 and 100.")
        if not isinstance(self.sparsityPercent, (int, float)):
            raise ValueError("Sparsity percent must be a number.")
        if not (0 <= self.sparsityPercent <= 100):
            raise ValueError("Sparsity percent must be between 0 and 100.")

    def get_coordinates(self):
        if self.numberOfSamples == 0:
            return np.array([], dtype=np.uint8), np.array([], dtype=np.uint8)

        stratifiedPixels = self.stratified_sampling(self.parentBucket, self.numberOfSamples)
        yCoords, xCoords = np.array(stratifiedPixels).T

        H, W = self.imageShape
        yCoords = np.clip(yCoords, 0, H - 1)
        xCoords = np.clip(xCoords, 0, W - 1)

        return rasterize_coordinates(self.imageShape, yCoords, xCoords)

    def stratified_sampling(self, bucket, numberOfSamples):

        children = direct_child_buckets(bucket)
        if is_leaf(children):
            return random_rejection_sampling(bucket, numberOfSamples)

        areaFractions = calculate_area_fractions_of_buckets(children, bucket)
        deterministic = deterministic_sample_counts(areaFractions, numberOfSamples)
        remainder = numberOfSamples - np.sum(deterministic)
        nondeterministic = non_deterministic_sample_counts(areaFractions, remainder)
        total = np.add(deterministic, nondeterministic)

        samplingArguments = [(child, samples, self.imageShape)
                for child, samples in zip(children, total) if samples > 0]

        if len(samplingArguments) > 0:
            with Pool(min(cpu_count(), len(samplingArguments))) as pool:
                results = pool.map(sample_child, samplingArguments)
                return np.vstack(results)

    def process_frame(self, frameNumber):
        yCoords, xCoords = self.get_coordinates()
        sampledImage = self.microscope.sample_image(yCoords, xCoords, frameNumber)
        pixelIntensities = sampledImage[yCoords, xCoords]

        knownPoints = np.column_stack((xCoords, yCoords))
        triangleObject = create_2d_triangulation(knownPoints)
        delaunayInterpolator = DelaunayInterpolator(triangleObject, pixelIntensities, self.imageShape,
                                                    imageDataType=self.microscope.dataType)
        reconstructedImage = delaunayInterpolator.interpolate_from_triangles()
        reconstructedImage =  np.clip(reconstructedImage, 0, self.microscope.maxValue).astype(self.microscope.dataType)

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
        visualize_microscope_image(sampled, imageTitle="Sampled Image (Stratified Sampling)")
        visualize_microscope_image(reconstructed, imageTitle="Reconstructed Image")

    def save_figures(self, frameNumber=4, save_path="plots"):
        if frameNumber >= len(self.reconstructedFrames):
            raise IndexError("Frame number exceeds number of reconstructed frames.")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        sampled = self.sampledFrames[frameNumber]
        reconstructed = self.reconstructedFrames[frameNumber]

        sampled_path = os.path.join(save_path, f"frame_{frameNumber}_sampled.png")
        recon_path = os.path.join(save_path, f"frame_{frameNumber}_reconstructed.png")

        visualize_microscope_image(sampled, imageTitle="Sampled Image (Stratified Sampling)",
                                   savePlot=True, savePath=sampled_path)
        visualize_microscope_image(reconstructed, imageTitle="Reconstructed Image",
                                   savePlot=True, savePath=recon_path)

