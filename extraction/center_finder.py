""" Center Finder """
import cv2
import copy


def centerFinder(best_rectangles_sorted, img_corners, settings):
    """Finds the center of each piece by finding the center of the 4 corners of a piece."""
    centers = []
    for piece in range(len(best_rectangles_sorted)):
        center_data1 = 0.5 * (best_rectangles_sorted[piece][0] + best_rectangles_sorted[piece][2])
        center_data2 = 0.5 * (best_rectangles_sorted[piece][1] + best_rectangles_sorted[piece][3])
        center_data3 = 0.5 * (center_data1 + center_data2)
        center = []
        center.append(int(center_data3[0]))
        center.append(int(center_data3[1]))
        centers.append(center)

    img_centers = copy.deepcopy(img_corners)
    for piece in range(len(centers)):
        point = centers[piece]
        cv2.circle(img=img_centers, center=tuple(point),
                   radius=settings.point_radius, color=[0, 0, 255], thickness=-1)

    return img_centers, centers
