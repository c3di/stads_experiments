import math
import numpy as np
from scipy.ndimage import gaussian_filter


def compute_sample_size(imageShape, sparsityPercent):
    return int(imageShape[0] * imageShape[1] * sparsityPercent / 100)


def find_corners(imageToSample):
    xCornerCoordinates = np.array([0, 0, len(imageToSample) - 1, len(imageToSample) - 1])
    yCornerCoordinates = np.array([0, len(imageToSample) - 1, 0, len(imageToSample) - 1])
    cornerPixelIntensities = np.array(
        [imageToSample[ yCornerCoordinates[i], xCornerCoordinates[i]] for i in range(len(xCornerCoordinates))])
    return np.array([yCornerCoordinates, xCornerCoordinates, cornerPixelIntensities]).astype(int)


def add_corners_to_samples(cornersToAdd, randomSparseFeatures):
    randomSparseFeatures = np.concatenate((randomSparseFeatures, cornersToAdd), axis=1)
    return randomSparseFeatures


def remove_duplicate_points(randomSparseFeatures):
    return np.unique(randomSparseFeatures, axis=1)


def calculate_number_of_pyramid_levels(imageSize):
    imageSizeY, imageSizeX = imageSize
    if imageSizeX <= 0 or imageSizeY <= 0:
        raise ValueError("illegal image size")
    numberOfPyramidLevels = 1
    pyramidResolution = 2
    while pyramidResolution < imageSizeX and pyramidResolution < imageSizeY:
        numberOfPyramidLevels = numberOfPyramidLevels + 1
        pyramidResolution = pyramidResolution * 2
    return numberOfPyramidLevels

def next_smaller_power_of_two( number ):
    if number <= 2:
        return 2
    return 2**int(math.log2( number-1 ))


def intersection_of_rectangles(a, b):
    upperLeft, lowerRight = a
    ay0, ax0 = upperLeft
    ay1, ax1  = lowerRight

    upperLeft, lowerRight = b
    by0, bx0 = upperLeft
    by1, bx1 = lowerRight

    x0 = max(ax0, bx0)
    x1 = min(ax1, bx1)
    y0 = max(ay0, by0)
    y1 = min(ay1, by1)

    if x0 > x1 or y0 > y1:
        return None

    return np.array([[y0, x0], [y1, x1]])


def calculate_number_of_samples_from_bucket(areaOfBucket, sparsityPercent):
    if sparsityPercent < 0.0 or sparsityPercent > 100.0:
        raise ValueError("invalid sparsity")
    if areaOfBucket <= 0:
        raise ValueError("invalid bucket")
    return int(sparsityPercent * areaOfBucket / 100)


def calculate_area_of_bucket(bucket):
    upper_left, lower_right = bucket
    y0, x0 = upper_left
    y1, x1 = lower_right
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    return width * height


def calculate_area_fractions_of_buckets(children, parentBucket):
    if not math.isclose(np.sum([calculate_area_of_bucket(child) for child in children]), calculate_area_of_bucket(parentBucket)):
        raise ValueError("Areas don't add up")
    areaFractionOfBucket = []
    for childBucket in children:
        area = calculate_area_of_bucket(childBucket) / calculate_area_of_bucket(parentBucket)
        areaFractionOfBucket.append(area)
    return np.array(areaFractionOfBucket)

def percentile_norm(a, smooth_sigma=1.0):

    lo, hi = np.percentile(a, (20, 80))
    if hi - lo < 1e-9:
        return np.zeros_like(a, dtype=np.float32)
    a = np.clip(a, lo, hi)
    a = (a - lo) / (hi - lo)
    a = gaussian_filter(a, sigma=smooth_sigma)
    return np.array(a).astype(np.float32)


def compute_triangle_circumcircles(triangleVertices):

    A = triangleVertices[:, 0, :]
    B = triangleVertices[:, 1, :]
    C = triangleVertices[:, 2, :]

    AB = B - A
    AC = C - A

    normals = np.cross(AB, AC)              # (M,3)
    norm_sq = np.einsum('ij,ij->i', normals, normals)

    centers = np.zeros_like(A)
    radii = np.zeros(A.shape[0], dtype=np.float32)

    # Degenerate triangles
    degenerateTriangles = norm_sq < 1e-12
    nonDegenerateTriangles = ~degenerateTriangles

    if np.any(degenerateTriangles):
        centers[degenerateTriangles] = (A[degenerateTriangles] + B[degenerateTriangles] + C[degenerateTriangles]) / 3
        radii[degenerateTriangles] = np.max(
            np.linalg.norm(
                np.stack([A[degenerateTriangles], B[degenerateTriangles], C[degenerateTriangles]], axis=1)
                - centers[degenerateTriangles][:, None, :],
                axis=2
            ),
            axis=1
        )

    if np.any(nonDegenerateTriangles):
        AB_nd = AB[nonDegenerateTriangles]
        AC_nd = AC[nonDegenerateTriangles]
        A_nd = A[nonDegenerateTriangles]

        cross_AB_AC = np.cross(AB_nd, AC_nd)
        denominator = 2 * np.einsum('ij,ij->i', cross_AB_AC, cross_AB_AC)

        alpha = np.einsum('ij,ij->i',
                          np.cross(AC_nd, cross_AB_AC),
                          AB_nd) / denominator

        beta = np.einsum('ij,ij->i',
                         np.cross(AB_nd, cross_AB_AC),
                         AC_nd) / denominator

        O = A_nd + alpha[:, None] * AB_nd + beta[:, None] * AC_nd

        centers[nonDegenerateTriangles] = O
        radii[nonDegenerateTriangles] = np.max(
            np.linalg.norm(
                np.stack([A_nd, B[nonDegenerateTriangles], C[nonDegenerateTriangles]], axis=1)
                - O[:, None, :],
                axis=2
            ),
            axis=1
        )

    return centers, radii


def calculate_circumcircle(tri_pts):
    N = tri_pts.shape[0]
    centers = np.zeros((N, 3), dtype=np.float32)
    radii = np.zeros(N, dtype=np.float32)

    for i in range(N):
        A, B, C = tri_pts[i]
        AB = B - A
        AC = C - A
        normal = np.cross(AB, AC)
        norm_sq = np.dot(normal, normal)
        if norm_sq < 1e-12:
            # Degenerate triangle fallback
            centers[i] = np.mean(tri_pts[i], axis=0)
            radii[i] = np.max(np.linalg.norm(tri_pts[i] - centers[i], axis=1))
            continue

        AB_cross_AC = np.cross(AB, AC)
        alpha = np.dot(np.cross(AC, AB_cross_AC), AB) / (2 * np.dot(AB_cross_AC, AB_cross_AC))
        beta = np.dot(np.cross(AB, AB_cross_AC), AC) / (2 * np.dot(AB_cross_AC, AB_cross_AC))
        O = A + alpha * AB + beta * AC
        centers[i] = O
        radii[i] = np.max(np.linalg.norm(tri_pts[i] - O, axis=1))

    return centers, radii

def calculate_mean_edge_length(triangleObject):

    allTriangles = triangleObject.simplices
    points = triangleObject.points

    vertices = points[allTriangles]

    AB = np.linalg.norm(vertices[:, 0, :] - vertices[:, 1, :], axis=1)
    BC = np.linalg.norm(vertices[:, 1, :] - vertices[:, 2, :], axis=1)
    CA = np.linalg.norm(vertices[:, 2, :] - vertices[:, 0, :], axis=1)
    return np.mean(np.hstack([AB, BC, CA]))

