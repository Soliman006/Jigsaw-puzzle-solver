""" Clearance Radius"""
import cv2
import numpy as np
import copy


def clearanceRadius(contours, img_mask_bgr, settings):
    """Finds the bounding circle of each piece, so that we know how much clearance we need to rotate a piece."""
    circle_radius = []
    circle_centers = []
    for piece in range(len(contours)):
        circle_center_data, radius = cv2.minEnclosingCircle(contours[piece])
        circle_center = []
        circle_center.append(int(circle_center_data[0]))
        circle_center.append(int(circle_center_data[1]))
        radius = int(radius + 1)
        circle_radius.append(radius)
        circle_centers.append(circle_center)
    img_circles = copy.deepcopy(img_mask_bgr)
    for piece in range(len(contours)):
        cv2.circle(img=img_circles, center=tuple(circle_centers[piece]), radius=radius, color=[
                   0, 0, 255], thickness=settings.line_thickness)
    # we then find the maximum radius and use that for the spacing between pieces:
    radius_max = np.max(circle_radius) + 1
    return img_circles, circle_centers, radius_max
