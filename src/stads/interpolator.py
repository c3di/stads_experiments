import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.spatial import Delaunay, cKDTree
from .delaunay_helpers import create_2d_triangulation


class ImageInterpolator:
    def __init__(self, imageShape, knownPoints, pixelIntensities, interpolMethod):
        self.imageShape = imageShape
        self.knownPoints = knownPoints
        self.pixelIntensities = pixelIntensities
        self.interpolMethod = interpolMethod
        self.validate_inputs()

    def validate_inputs(self):

        if not isinstance(self.imageShape, tuple) or len(self.imageShape) != 2:
            raise ValueError("imageShape must be a tuple of two integers (height, width)")
        if self.interpolMethod not in ['linear', 'cubic', 'nearest']:
            raise ValueError(f"Unknown interpolation method: {self.interpolMethod}")
        if len(self.knownPoints) != len(self.pixelIntensities):
            raise ValueError("Number of knownPoints must match number of pixelIntensities")
        if self.knownPoints.ndim != 2 or self.knownPoints.shape[1] != 2:
            raise ValueError("knownPoints must be a 2D array with shape (N, 2) for [x, y] coordinates")

        if not np.all(np.isfinite(self.knownPoints)):
            raise ValueError("knownPoints contains NaN or infinite values")

        if not np.all(np.isfinite(self.pixelIntensities)):
            raise ValueError("pixelIntensities contains NaN or infinite values")

        width, height = self.imageShape[1], self.imageShape[0]
        x_valid = (self.knownPoints[:, 0] >= 0) & (self.knownPoints[:, 0] < width)
        y_valid = (self.knownPoints[:, 1] >= 0) & (self.knownPoints[:, 1] < height)

        if not np.all(x_valid & y_valid):
            raise ValueError("Some knownPoints are outside the image bounds")

    def interpolate_image(self):
        height, width = self.imageShape
        gridY, gridX = np.mgrid[0:height, 0:width]

        if self.interpolMethod in ['cubic', 'nearest']:
            interpolatedImage = griddata(self.knownPoints, self.pixelIntensities, (gridX, gridY),
                                         method=self.interpolMethod)
        elif self.interpolMethod == 'linear':
            interpolator = LinearNDInterpolator(self.knownPoints, self.pixelIntensities)
            interpolatedImage = interpolator(gridX, gridY)
        else:
            raise RuntimeError("Interpolation method validation failed to catch an invalid method")

        return np.nan_to_num(interpolatedImage, nan=0.0)


class DelaunayInterpolator:
    def __init__(self, triangleObject, values, imageShape, imageDataType):
        self.triangleObject = triangleObject
        self.imageShape = imageShape
        self.values = values.astype(np.float64)
        self.imageDataType = imageDataType
        self.validate_inputs()


    def validate_inputs(self):

        if not hasattr(self.triangleObject, "points") or not hasattr(self.triangleObject, "simplices"):
            raise ValueError("triangleObject must be a valid Delaunay triangulation")

        if not isinstance(self.triangleObject.points, np.ndarray):
            raise ValueError("triangleObject.points must be a numpy array")

        if self.triangleObject.points.ndim != 2 or self.triangleObject.points.shape[1] != 2:
            raise ValueError("triangleObject.points must have shape (N, 2)")

    def interpolate_inside_hull(self, gridPoints, validSimplices, insideMask, interpolated):

        transform = self.triangleObject.transform[validSimplices]
        validGrid = gridPoints[insideMask]

        delta = validGrid - transform[:, 2]
        bary = np.einsum('ijk,ik->ij', transform[:, :2, :], delta)
        baryCoords = np.c_[bary, 1 - bary.sum(axis=1)]

        vertexIndices = self.triangleObject.simplices[validSimplices]
        vertexValues = self.values[vertexIndices]

        interpolatedValues = np.einsum('ij,ij->i', vertexValues, baryCoords)
        interpolated[insideMask] = interpolatedValues

    def interpolate_outside_hull(self, gridPoints, outsideMask, interpolated):

        tree = cKDTree(self.triangleObject.points)
        _, idx = tree.query(gridPoints[outsideMask])

        interpolated[outsideMask] = self.values[idx]

    def restore_data_type(self, interpolated):
        interpolated = interpolated.reshape(self.imageShape)

        if np.issubdtype(self.imageDataType, np.integer):
            info = np.iinfo(self.imageDataType)
            interpolated = np.clip(interpolated, info.min, info.max)

        interpolated = interpolated.astype(self.imageDataType)

        return interpolated

    def interpolate_from_triangles(self):

        H, W = self.imageShape

        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        gridPoints = np.column_stack([gridX.ravel(), gridY.ravel()])

        simplices = self.triangleObject.find_simplex(gridPoints)
        insideMask = simplices >= 0
        outsideMask = simplices < 0
        validSimplices = simplices[insideMask]

        interpolated = np.zeros(len(gridPoints), dtype=np.float32)

        if np.flatnonzero(insideMask).size > 0:
            self.interpolate_inside_hull(gridPoints, validSimplices, insideMask, interpolated)

        if np.flatnonzero(outsideMask).size > 0:
            print(f"Adding {np.flatnonzero(outsideMask).size} points that were outside the convex hull")
            self.interpolate_outside_hull(gridPoints, outsideMask, interpolated)

        interpolatedImage = self.restore_data_type(interpolated)

        return interpolatedImage