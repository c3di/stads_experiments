import numpy as np
from scipy.spatial import Delaunay

from .utility_functions import compute_triangle_circumcircles, calculate_mean_edge_length


def create_2d_triangulation(currentPoints):
    tri = Delaunay(currentPoints)
    return tri


def remove_duplicate_points(vertices, relevantPastPoints, relevantPastValues ):
    currentFramePoints = set(map(tuple, vertices[:, :2].astype(int)))
    maskUnique = np.array([tuple(p[:2].astype(int)) not in currentFramePoints for p in relevantPastPoints])
    relevantPastPoints = relevantPastPoints[maskUnique]
    relevantPastValues = relevantPastValues[maskUnique]
    return relevantPastPoints, relevantPastValues


def update_triangles_backwards(
        currentPoints,
        currentValues,
        pastPoints,
        pastValues,
        history,
):


    tri = Delaunay(currentPoints, incremental=True)

    triangles = tri.simplices.astype(np.int32)
    alpha = calculate_mean_edge_length(tri)

    print(f"Initial triangulation: {len(currentValues)} points, {len(tri.simplices)} triangles")

    t_curr = 0.0
    vertices = np.hstack([currentPoints, np.full((currentPoints.shape[0], 1), t_curr, dtype=np.float32)])
    values = currentValues.copy()

    num_past_frames = len(pastPoints)
    frames_to_use = min(history, num_past_frames)

    if frames_to_use == 0:
        return tri,values

    past_pts_list = []
    past_vals_list = []
    t_list = []

    for h in range(1, frames_to_use + 1):
        pts = pastPoints[-h]
        vals = pastValues[-h]
        tPast = alpha * h
        past_pts_list.append(pts)
        past_vals_list.append(vals)
        t_list.append(np.full((pts.shape[0], 1), tPast, dtype=np.float32))

    pastPointsFlat = np.vstack(past_pts_list)
    pastValuesFlat = np.concatenate(past_vals_list)
    pastPoints3D = np.hstack([pastPointsFlat, np.vstack(t_list)])


    # --- Step 4: Find which 2D triangle contains each past point ---
    simplexIDs = tri.find_simplex(pastPoints3D[:, :2])
    validMask = simplexIDs >= 0
    if not np.any(validMask):
        return vertices, values, triangles

    pastPoints3D = pastPoints3D[validMask]
    pastValuesFlat = pastValuesFlat[validMask]
    simplexIDs = simplexIDs[validMask]

    triangleVerticesIDs = triangles[simplexIDs]
    triangleVertices3D = vertices[triangleVerticesIDs]  # shape (num_points, 3, 3)


    # --- Step 5: Circumcircle test in 3D ---
    circleCenters, circleRadii = compute_triangle_circumcircles(triangleVertices3D)
    distanceToCircle = np.linalg.norm(pastPoints3D - circleCenters, axis=1)
    split_mask = distanceToCircle < circleRadii

    points_to_add = pastPoints3D[split_mask]
    values_to_add = pastValuesFlat[split_mask]

    if points_to_add.shape[0] == 0:
        return vertices, values, triangles

    # --- Step 6: Add points incrementally to the 2D Delaunay triangulation ---
    tri.add_points(points_to_add[:, :2])  # only x,y for 2D Delaunay

    # --- Step 7: Update vertices and values arrays ---
    vertices = np.vstack([vertices, points_to_add])
    values = np.append(values, values_to_add)

    print(f"Final triangulation: vertices={vertices.shape[0]}, triangles={triangles.shape[0]}")

    return tri, values