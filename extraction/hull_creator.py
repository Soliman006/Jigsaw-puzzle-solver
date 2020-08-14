""" Hull Creator """
import cv2
import copy


def hullCreator(contours, img_mask_bgr, settings):
    """Creates a bounding hull for each of the contours."""
    hull = []
    hull_points = []
    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], returnPoints=False))
        hull_points.append(cv2.convexHull(contours[i], False))
    img_hull_mask_bgr = copy.deepcopy(img_mask_bgr)
    for piece in range(len(hull_points)):
        for point in range(len(hull_points[piece])):
            location = tuple(hull_points[piece][point][0])
            cv2.circle(img=img_hull_mask_bgr, center=location,
                       radius=settings.point_radius, color=[0, 0, 255], thickness=-1)
    cv2.drawContours(img_hull_mask_bgr, hull_points, -1,
                     (0, 255, 0), thickness=settings.line_thickness)
    return img_hull_mask_bgr, hull, hull_points
