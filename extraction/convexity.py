""" Convexity """
import cv2
import numpy as np
import copy

from global_settings import globalSettings
settings = globalSettings()


def convexity(contours, hull, img_mask_bgr, settings):
    """Finds the point along a contour where it's deviation from the hull of that contour is a local maximum."""
    all_defects = []
    for i in range(len(contours)):
        all_defects.append(cv2.convexityDefects(contours[i], hull[i]))

# filter to only keep significant defects
    defects = []
    defects_f = []
    epsilon = settings.convexity_epsilon
    for piece in range(len(all_defects)):
        piece_defects = []
        piece_defects_f = []
        for defect in range(len(all_defects[piece])):
            defect_val = all_defects[piece][defect][0]
            s, e, f, d = defect_val
            if d > epsilon:
                piece_defects.append(np.asarray(defect_val))
                piece_defects_f.append(np.asarray(f))
        defects.append(piece_defects)
        defects_f.append(piece_defects_f)

# extract the index of the points that make up the convex closure hull
    defect_hulls_index = []
    for piece in range(len(defects)):
        piece_defect_hulls_index = []
        for defect in range(len(defects[piece])):
            defect_val = defects[piece][defect]
            s, e, f, d = defect_val
            piece_defect_hulls_index.append(s)
            piece_defect_hulls_index.append(e)
        defect_hulls_index.append(piece_defect_hulls_index)

# convert the index into actual coordinates
    defect_hulls = []
    for piece in range(len(defect_hulls_index)):
        piece_defect_hulls = []
        for defect in range(len(defect_hulls_index[piece])):
            index = defect_hulls_index[piece][defect]
            point = contours[piece][index][0]
            piece_defect_hulls.append(np.asarray(point))
        defect_hulls.append(np.asarray(piece_defect_hulls))

# plot and save the convexity defect points
    defect_points = []
    for piece in range(len(defects)):
        piece_defect_points = []
        for defect in range(len(defects[piece])):
            s, e, f, d = defects[piece][defect]
            point = contours[piece][f][0]
            piece_defect_points.append(np.asarray(point))
        defect_points.append(np.asarray(piece_defect_points))

    img_defects = copy.deepcopy(img_mask_bgr)
    for piece in range(len(defect_points)):
        for defect in range(len(defect_points[piece])):
            point = defect_points[piece][defect]
            cv2.circle(img=img_defects, center=tuple(point),
                       radius=settings.point_radius, color=[0, 0, 255], thickness=-1)

    return img_defects, defects_f
