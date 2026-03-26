import os
import numpy as np
import logging
from . import evaluation
from .image_processing import generate_scan_pattern_from_pdf, extract_points
from .interpolator import DelaunayInterpolator
from .delaunay_helpers import update_triangles_backwards
from .monitor import visualize_microscope_image, save_absolute_error_map, save_pixel_wise_psnr_plots
from .stads_helpers import (
    compute_local_moments_of_image,
    compute_pdf_from_gradients_image,
    compute_optical_flow,
)
from .microscope import Microscope
from .random_sampler import RandomSampler
from .stratified_sampler import StratifiedSampler

logging.basicConfig(level=logging.INFO)


class AdaptiveSampler:
    def __init__(self, initialSampling, interpolMethod, sparsityPercent, numberOfFrames,
                 groundTruthName="li_expulsion_one", withTemporal=True):
        self.temporalWeight = None
        self.windowSize = 8
        self.initialSampling = initialSampling
        self.interpolMethod = interpolMethod
        self.sparsityPercent = sparsityPercent
        self.numberOfFrames = numberOfFrames
        self.withTemporal = withTemporal
        self.groundTruthName = groundTruthName

        self.microscope = Microscope(self.groundTruthName)
        self.imageShape = self.microscope.imageShape

        self.reconstructedFrames = []
        self.gradientsMaps = []
        self.yCoords, self.xCoords = self.initialize_sampling()
        self.currentPoints = extract_points(self.xCoords, self.yCoords, self.imageShape)

        self.pastPoints2D = []
        self.pastValues = []
        self.triangleObject = []
        self.history = numberOfFrames

        self.flowMap = np.zeros(self.imageShape, dtype=np.float32)
        self.flowMaps = []
        self.temporalVarianceMap = np.ones(self.imageShape, dtype=np.float32)
        self.spatialVarianceMap = np.ones(self.imageShape, dtype=np.float32)

        self.pdfs = [np.ones(self.imageShape, dtype=np.float32)]

        self.psnrs = []
        self.ssims = []


    def initialize_sampling(self):
        if self.initialSampling == "uniform":
            randomSampler = RandomSampler(
                self.interpolMethod, self.sparsityPercent, imageShape=self.imageShape, groundTruthName=self.groundTruthName
            )
            return randomSampler.get_coordinates()
        elif self.initialSampling == "stratified":
            stratifiedSampler = StratifiedSampler(self.interpolMethod, self.sparsityPercent, groundTruthName=self.groundTruthName)
            return stratifiedSampler.get_coordinates()
        else:
            raise ValueError("Invalid initial sampling method. Choose 'uniform' or 'stratified'.")

    def get_samples(self, frameNumber=0):
        self.currentPoints = extract_points(self.xCoords, self.yCoords, self.imageShape)

        self.xCoords = self.currentPoints[:, 0].astype(int)
        self.yCoords = self.currentPoints[:, 1].astype(int)

        pixelIntensities = self.microscope.groundTruthVideo[frameNumber, self.yCoords, self.xCoords]
        return pixelIntensities

    def interpolate_sparse_image(self, allValues):
        delaunayInterpolator = DelaunayInterpolator(self.triangleObject, allValues, self.imageShape,
                                                    imageDataType=self.microscope.dataType)
        reconstructedImage = delaunayInterpolator.interpolate_from_triangles()
        return np.clip(reconstructedImage, 0, self.microscope.maxValue).astype(self.microscope.dataType)

    def update_reconstructed_frames(self, reconstructedImage):
        self.reconstructedFrames.append(reconstructedImage)

    def compute_pdf_based_on_flow_magnitude(self):
        epsilon = 1e-12
        flowBasedPDF = compute_pdf_from_gradients_image(self.flowMap)
        flowBasedPDF = flowBasedPDF / (np.sum(flowBasedPDF) + epsilon)
        return flowBasedPDF

    def compute_pdf_based_on_spatial_variance(self):
        epsilon = 1e-12
        varianceBasedPDF = compute_pdf_from_gradients_image(self.spatialVarianceMap)
        varianceBasedPDF = varianceBasedPDF / (np.sum(varianceBasedPDF) + epsilon)
        return varianceBasedPDF

    def update_scan_pattern(self):
        self.yCoords, self.xCoords = generate_scan_pattern_from_pdf(self.pdfs[-1], self.sparsityPercent)

    def update_flow_map(self, frameNumber=1):
        self.flowMap = compute_optical_flow(
            self.gradientsMaps[frameNumber - 1], self.gradientsMaps[frameNumber], self.windowSize
        )
        self.flowMaps.append(self.flowMap.copy())
        print(f"[DEBUG] Flow map appended at frame {frameNumber}. Total now: {len(self.flowMaps)}")

    def generate_scan_pattern_for_next_frame(self, frameNumber):
        pixelIntensities = self.get_samples(frameNumber)

        if len(self.pastPoints2D) > self.history:
            self.pastPoints2D = self.pastPoints2D[-self.history:]
            self.pastValues = self.pastValues[-self.history:]

        self.triangleObject, allValues = update_triangles_backwards(self.currentPoints, pixelIntensities,
                                                                    self.pastPoints2D, self.pastValues, self.history)

        self.pastPoints2D.append(self.currentPoints.copy())
        self.pastValues.append(pixelIntensities.copy())

        reconstructedImage = self.interpolate_sparse_image(allValues)
        self.update_reconstructed_frames(reconstructedImage)

        if frameNumber < self.numberOfFrames - 1:
            meanMap, gradientMap, spatialVariance = compute_local_moments_of_image(
                reconstructedImage, self.windowSize
            )
            self.gradientsMaps.append(gradientMap)
            self.spatialVarianceMap = spatialVariance

            if self.withTemporal and frameNumber > 1:
                self.update_flow_map(frameNumber)
                pdf = self.compute_pdf_based_on_flow_magnitude()
                self.pdfs.append(pdf)

            elif (not self.withTemporal) or (frameNumber <= 1):
                pdf = self.compute_pdf_based_on_spatial_variance()
                self.pdfs.append(pdf)

            self.update_scan_pattern()

    def run(self):
        for frameNumber in range(self.numberOfFrames):
            self.generate_scan_pattern_for_next_frame(frameNumber)

            psnr = evaluation.calculate_psnr(self.microscope.groundTruthVideo[frameNumber],
                                             self.reconstructedFrames[frameNumber])
            ssim = evaluation.calculate_ssim(self.reconstructedFrames[frameNumber],
                                             self.microscope.groundTruthVideo[frameNumber])
            self.psnrs.append(psnr)
            self.ssims.append(ssim)

        return np.array(self.reconstructedFrames), self.psnrs, self.ssims

    def coordinates_to_mask(self):
        mask = np.zeros(self.imageShape, dtype=self.microscope.dataType)
        mask[self.yCoords, self.xCoords] = self.microscope.maxValue
        return mask


    def show_figures(self, frameNumber=8):
        if frameNumber >= len(self.reconstructedFrames):
            raise IndexError("Frame number exceeds number of reconstructed frames.")

        reconstructed = self.reconstructedFrames[frameNumber]
        groundTruth = self.microscope.groundTruthVideo[frameNumber]

        print(f"Displaying results for frame {frameNumber}")
        print(f"PSNR: {self.psnrs[frameNumber]:.2f}, SSIM: {self.ssims[frameNumber]:.4f}")
        visualize_microscope_image(reconstructed, imageTitle="Reconstructed Image")

        sampling_mask = self.coordinates_to_mask()
        visualize_microscope_image(sampling_mask, imageTitle="Sampling Mask")

        # Absolute error map
        save_absolute_error_map(groundTruth, reconstructed)

        # Pixel-wise PSNR map
        save_pixel_wise_psnr_plots(groundTruth, reconstructed)


    def save_figures(self, frameNumber=8, save_path="plots"):
        if frameNumber >= len(self.reconstructedFrames):
            raise IndexError("Frame number exceeds number of reconstructed frames.")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        reconstructed = self.reconstructedFrames[frameNumber]
        groundTruth = self.microscope.groundTruthVideo[frameNumber]

        if len(self.pdfs) > 0:
            pdf_map = self.pdfs[frameNumber]
            visualize_microscope_image(pdf_map / pdf_map.sum(), imageTitle="PDF Map",
                                       savePlot=True,
                                       savePath=os.path.join(save_path, f"frame_{frameNumber}_pdf_map.png"))
            np.save(os.path.join(save_path, f"frame_{frameNumber}_pdf_map.npy"), pdf_map)


        visualize_microscope_image(reconstructed, imageTitle="Reconstructed Image",
                                   savePlot=True,
                                   savePath=os.path.join(save_path, f"frame_{frameNumber}_reconstructed.png"))

        sampling_mask = self.coordinates_to_mask()
        visualize_microscope_image(sampling_mask, imageTitle="Sampling Mask",
                                   savePlot=True,
                                   savePath=os.path.join(save_path, f"frame_{frameNumber}_sampling_mask.png"))

        # Absolute error map
        save_absolute_error_map(groundTruth, reconstructed,
                                savePlot=True,
                                savePath=os.path.join(save_path, f"frame_{frameNumber}_abs_error_map.tiff"))

        # Pixel-wise PSNR map
        save_pixel_wise_psnr_plots(groundTruth, reconstructed,
                                   savePlot=True,
                                   savePath=os.path.join(save_path, f"frame_{frameNumber}_pixelwise_psnr.tiff"))
