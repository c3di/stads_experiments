import math
import random

import numpy as np

from .utility_functions import next_smaller_power_of_two, intersection_of_rectangles


def select_index(probabilities, dice):
    index = 0
    cumulativeDensity = np.cumsum(probabilities)
    while index < len(probabilities) - 1:
        if dice <= cumulativeDensity[index]:
            return index
        index = index + 1
    return len(probabilities) - 1


def sample_child(samplingArguments):
    bucket, samples, imageShape = samplingArguments
    return random_rejection_sampling(bucket, samples)


def direct_child_buckets(parentBucket):
    (y0, x0), (y1, x1) = parentBucket
    if y1 - y0 < 2 and x1 - x0 < 2:
        return np.array([parentBucket])

    splitPointX = next_smaller_power_of_two(x1 - x0 + 1)
    splitPointY = next_smaller_power_of_two(y1 - y0 + 1)

    s = max(splitPointX,splitPointY)

    buckets = np.array([
        [[y0 + 0 * s, x0 + 0 * s], [y0 + 1 * s - 1, x0 + 1 * s - 1]],
        [[y0 + 0 * s, x0 + 1 * s], [y0 + 1 * s - 1, x0 + 2 * s - 1]],
        [[y0 + 1 * s, x0 + 0 * s], [y0 + 2 * s - 1, x0 + 1 * s - 1]],
        [[y0 + 1 * s, x0 + 1 * s], [y0 + 2 * s - 1, x0 + 2 * s - 1]]
    ])

    children = []
    for bucket in buckets:
        croppedBucket = intersection_of_rectangles(bucket, parentBucket)
        if croppedBucket is not None:
            children.append(croppedBucket)
    return np.array(children)


def is_leaf(buckets):
    return len(buckets) <= 1


def random_rejection_sampling(bucket, numberOfSamples):
    (y0, x0), (y1, x1) = bucket
    points = [(y, x) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    if numberOfSamples > len(points):
        raise ValueError("Sample size exceeds available points.")
    return np.array(random.sample(points, numberOfSamples))


def deterministic_sample_counts(areaFractions, n):
    return np.array([int(f * n) for f in areaFractions])


def non_deterministic_sample_counts(areaFractions, numberOfSamples):
    if not math.isclose(np.sum(areaFractions), 1.0):
        raise ValueError("Area fractions must sum to 1.")
    result = [0] * len(areaFractions)
    chosenIndices = []
    i = 0
    while i < numberOfSamples:
        dice = random.uniform(0, 1)
        index = select_index(areaFractions, dice)
        if index not in chosenIndices:
            chosenIndices.append(index)
            result[index] += 1
            i += 1
            if len(chosenIndices) == len(areaFractions):
                chosenIndices = []
    return np.array(result)
