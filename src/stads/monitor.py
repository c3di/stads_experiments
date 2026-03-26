import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.pyplot import inferno
from scipy.ndimage import uniform_filter

from .image_processing import extract_patch  # Make sure this function is correct

def configure_axis(ax, imageData):
    ax.set_xlim(0, imageData.shape[1])
    ax.set_ylim(0, imageData.shape[0])
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

def draw_detail_rectangle(ax, detail_size, origin=(0, 0)):
    rectangle = patches.Rectangle(origin, detail_size[1], detail_size[0],
                                  linewidth=1.2, edgecolor='red', facecolor='none')
    ax.add_patch(rectangle)

def draw_region_of_interest(ax, detailSize, regionOfInterest):
    ((top, left), (height, width)) = regionOfInterest
    rectangle = patches.Rectangle((left, top), width, height,
                                  linewidth=1.2, edgecolor='red', facecolor='none')
    ax.add_patch(rectangle)

    # Dashed connector lines from full image corner to ROI
    line1 = Line2D([detailSize[1], left + width], [0, top], color='blue', linestyle='dashed')
    line2 = Line2D([0, left], [detailSize[0], top + height], color='blue', linestyle='dashed')
    ax.add_line(line1)
    ax.add_line(line2)

def draw_roi_highlight(ax, detailSize, regionOfInterest):
    draw_detail_rectangle(ax, detailSize)
    draw_region_of_interest(ax, detailSize, regionOfInterest)


def generate_mask_from_coordinates(yCoordinates, xCoordinates, imageShape):
    yCoordinates = np.asarray(yCoordinates, dtype=int)
    xCoordinates = np.asarray(xCoordinates, dtype=int)

    mask = np.zeros(imageShape, dtype=np.float32)
    valid = (
        (yCoordinates >= 0) & (yCoordinates < imageShape[0]) &
        (xCoordinates >= 0) & (xCoordinates < imageShape[1])
    )
    mask[yCoordinates[valid], xCoordinates[valid]] = 1.0
    return mask

def display_masked_image(mask, originalImage):
    plt.imshow(originalImage, cmap='gray')
    plt.title("Original Image")
    plt.show()

    plt.imshow(mask, cmap='binary')
    plt.title("Mask")
    plt.show()

    masked = np.multiply(originalImage, mask).astype(originalImage.dtype)
    plt.imshow(masked, cmap='gray')
    plt.title("Masked Image")
    plt.show()

def visualize_microscope_image(imageData, imageTitle="", savePlot=False, savePath='output.png'):

    top, left = imageData.shape[0]//2 + 10, imageData.shape[1]//2 + 10
    height, width = imageData.shape[1]//10, imageData.shape[1]//10
    bottom, right = top + height, left + width

    fig, ax = plt.subplots()
    ax.set_title(imageTitle)

    # Show full image
    ax.imshow(imageData, cmap='gray')
    fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.08)

    # Axis setup
    ax.set_xlim(0, imageData.shape[1])
    ax.set_ylim(0, imageData.shape[0])
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw red dashed rectangle around ROI
    roi_rect = patches.Rectangle((left, top), width, height,
                                 linewidth=1.0, edgecolor='red',
                                 facecolor='none', linestyle='dashed')
    ax.add_patch(roi_rect)

    # Extract ROI patch and scale it to top-left quarter
    zoom_patch = extract_patch(imageData, ((top, left), (height, width)))
    zoom_patch = cv2.resize(zoom_patch, (
        imageData.shape[1] // 2, imageData.shape[0] // 2),
        interpolation=cv2.INTER_NEAREST)

    # Overlay zoomed-in patch in top-left quarter
    ax.imshow(zoom_patch, extent=(0, imageData.shape[1] // 2,
                                  imageData.shape[0] // 2, 0),
              cmap='gray', interpolation='none')

    # Define corners of zoomed patch and ROI box, excluding top-left corner of zoom patch
    zoom_box_coords = [
        (imageData.shape[1] // 2, 0),                # top-right corner of zoom patch
        (0, imageData.shape[0] // 2),                # bottom-left corner of zoom patch
        (imageData.shape[1] // 2, imageData.shape[0] // 2)  # bottom-right corner of zoom patch
    ]
    roi_box_coords = [
        (right, top),    # top-right corner of ROI
        (left, bottom),  # bottom-left corner of ROI
        (right, bottom)  # bottom-right corner of ROI
    ]

    # Draw red dashed lines connecting the 3 pairs of corners
    for zc, rc in zip(zoom_box_coords, roi_box_coords):
        line = Line2D([zc[0], rc[0]], [zc[1], rc[1]],
                      color='red', linestyle='dashed', linewidth=1.0)
        ax.add_line(line)

    if savePlot:
        fig.savefig(savePath, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        plt.show()

def overlay_images(firstImage, secondImage):
    if firstImage.shape != secondImage.shape:
        raise ValueError("Input images must have the same dimensions")

    firstImage = firstImage.astype(np.uint8)
    secondImage = secondImage.astype(np.uint8)
    greenChannel = np.zeros_like(firstImage)

    firstRGBImage = cv2.merge([greenChannel, greenChannel, firstImage])
    secondRGBImage = cv2.merge([secondImage, greenChannel, greenChannel])

    combinedImage = cv2.addWeighted(firstRGBImage, 1, secondRGBImage, 1, 0)
    combinedImage = np.clip(combinedImage, 0, 255).astype(np.uint8)

    return combinedImage


def save_pixel_wise_psnr_plots(groundTruthImage, reconstructedImage,savePlot=False,savePath = 'psnr_map.tiff',dpi=600):
    neighborhoodRadius = 2
    psnrCap = 75
    dataType = groundTruthImage.dtype
    info = np.iinfo(dataType)
    maxValue = info.max

    squaredError = (groundTruthImage.astype(np.float32) - reconstructedImage.astype(np.float32)) ** 2
    kernelSize = 2 * neighborhoodRadius + 1

    meanSquaredErrorMap = uniform_filter(squaredError, size=kernelSize, mode='reflect')
    meanSquaredErrorMap = np.maximum(meanSquaredErrorMap, 1e-10)

    psnrMap = 10 * np.log10((maxValue ** 2) / meanSquaredErrorMap)
    psnrMap = np.clip(psnrMap, 0, psnrCap)

    psnrOverlay = overlay_images(reconstructedImage, (psnrMap / psnrCap * maxValue).astype(dataType))

    if savePlot:
        if savePath is None:
            raise ValueError("savePath must be provided if savePlot=True")
        os.makedirs(os.path.dirname(savePath), exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(psnrOverlay, cmap="inferno", vmin=0, vmax=psnrCap)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("PSNR (dB)", rotation=90)
        ax.set_title("Pixel-wise PSNR")
        ax.axis("off")
        plt.savefig(savePath, dpi=dpi)
        plt.close(fig)

    else:
        plt.imshow(psnrOverlay, cmap="inferno")
        plt.title("Pixel-wise PSNR Overlay")
        plt.colorbar(label="PSNR (dB)")
        plt.show()

    return psnrOverlay

def save_absolute_error_map(groundTruthImage, reconstructedImage, savePlot=False, savePath='abs_error.tiff', dpi=600):

    absErrorMap = np.abs(groundTruthImage - reconstructedImage)

    if savePlot:
        if savePath is None:
            raise ValueError("savePath must be provided if savePlot=True")
        os.makedirs(os.path.dirname(savePath), exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(absErrorMap, cmap='inferno', vmin=0, vmax=absErrorMap.max())
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Absolute Error", rotation=90)
        ax.set_title("Pixel-wise Absolute Error Map")
        ax.axis("off")
        plt.savefig(savePath, dpi=dpi)
        plt.close(fig)

    else:
        plt.imshow(absErrorMap, cmap='inferno')
        plt.title("Pixel-wise Absolute Error Map")
        plt.colorbar(label="Absolute Error")
        plt.show()

    return absErrorMap



