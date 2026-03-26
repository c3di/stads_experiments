import numpy as np
from skimage import filters

from .utility_functions import compute_sample_size

def extract_patch(imageData, regionOfInterest):
    (top, left), (height, width) = regionOfInterest
    return imageData[top:top+height, left:left+width]

def gaussian_kernel_2d(sigma, radius_factor=3):
    r = int(radius_factor * sigma)
    y, x = np.mgrid[-r:r+1, -r:r+1]
    g = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
    g /= g.sum()
    return g


def normalize_image(image, minValue, maxValue):
    if minValue == 0 and maxValue == 255:
        return image

    if maxValue > minValue:
        return (((image-minValue)*255)/(maxValue-minValue)).astype(np.uint8)
    else:
        return np.zeros_like(image)


def compute_image_of_gradient_magnitudes(image):
    gradientsImage = np.asarray(filters.sobel(image))
    return gradientsImage


def rasterize_coordinates(imageShape, yCoordinates, xCoordinates):

    if yCoordinates.shape != xCoordinates.shape:
        raise ValueError("yCoordinates and xCoordinates must have the same shape.")

    if not (np.all((yCoordinates >= 0) & (yCoordinates < imageShape[0])) and
            np.all((xCoordinates >= 0) & (xCoordinates < imageShape[1]))):
        raise ValueError("Coordinates must lie within the image dimensions.")


    coordinates = np.array(list(zip(yCoordinates, xCoordinates)), dtype=[('y', int), ('x', int)])

    sortedCoordinates = np.sort(coordinates, order=['y', 'x'])

    sortedY = sortedCoordinates['y']
    sortedX = sortedCoordinates['x']

    return sortedY, sortedX


def generate_random_pixel_locations(imageShape, sparsityPercent):
    sampleSize = compute_sample_size(imageShape, sparsityPercent)
    randomPixelIndices = np.random.choice(imageShape[0] * imageShape[1], size=sampleSize, replace=False)
    yRandomPixels, xRandomPixels = divmod(randomPixelIndices, imageShape[1])
    return rasterize_coordinates(imageShape, yRandomPixels, xRandomPixels)


def generate_scan_pattern_from_pdf(pdfImage, sparsityPercent):
    sampleSize = compute_sample_size(pdfImage.shape, sparsityPercent)
    indices = np.arange(pdfImage.shape[0] * pdfImage.shape[1])
    pdfFlat = pdfImage.flatten()
    nonZeroProbabilities = np.nonzero(pdfFlat)[0]
    numberDiceRolls = min(sampleSize, len(nonZeroProbabilities))
    selectedIndices = np.random.choice(indices, size = numberDiceRolls, p = pdfFlat, replace=False)
    yCoordinates, xCoordinates = divmod(selectedIndices, pdfImage.shape[1])
    return rasterize_coordinates(pdfImage.shape,yCoordinates, xCoordinates)

def extract_points(x, y, imageShape):
    H, W = imageShape
    points = np.column_stack((x, y))
    corners = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]])

    return np.vstack([points, corners])
