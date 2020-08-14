import cv2
import numpy as np
from utils import zoom, move_contour


def cropping(data):
    cropped_pieces = []
    cropped_contours = []
    cropped_masks = []
    cropped_masks_scaled = []
    for piece in range(len(data.grid_centers)):
        cropped = zoom(data.img_processed_bgr, data.grid_centers[piece], data.radius_max)
        cropped_pieces.append(cropped)
        new_contour = move_contour(data.contours_rotated[piece], data.grid_centers[piece], [data.radius_max, data.radius_max])
        cropped_contours.append(new_contour)

        h, w, ch = cropped.shape
        cropped_mask = np.zeros([h, w, ch], dtype=np.uint8)
        cv2.drawContours(cropped_mask, [new_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        cropped_masks.append(cropped_mask)

        scale = 0.9
        precurser = cv2.resize(cropped_mask, (int(scale*w), int(scale*h)), interpolation=cv2.INTER_AREA)
        cropped_mask_scaled = np.zeros([h, w, ch], dtype=np.uint8)
        cropped_mask_scaled[int((h-int(scale*h))/2):h-int((h-int(scale*h))/2),
                            int((w-int(scale*w))/2):w-int((w-int(scale*w))/2)] = precurser
        cropped_masks_scaled.append(cropped_mask_scaled)
    return cropped_pieces, cropped_contours, cropped_masks, cropped_masks_scaled
