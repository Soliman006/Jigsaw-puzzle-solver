""" Contour Finder """
import cv2
import numpy as np
import copy


def contourFinder(img_input, img_blank_comp, settings):
    """Given a BGR input image, this find the contours that follow the edge of the pieces."""
    # convert the image to HSV colour space:
    img_hsv = cv2.cvtColor(img_input, cv2.COLOR_BGR2HSV)
    # Next we use thresholding to create a mask that distinguishes between the pieces and the background:
    mask = cv2.inRange(img_hsv, settings.bg_thresh_low, settings.bg_thresh_high)
    # Now invert the mask
    mask_inv = cv2.bitwise_not(mask)

    # Optional tecniques for cleaning up the contours
    # kernel = np.ones((3, 3), np.uint8)
    # mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)
    # mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)

    # cv2_imshow(image_resize(mask_inv, height = settings.disp_height))

    # Now we perform a contour search on the mask and save a vector of all the contours:
    ret, thresh = cv2.threshold(mask_inv, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Now filter to leave only the actual puzzle piece contours
    piece_contours = contourFilter(contours)

    # Now we will recreate the mask using the contours, which will give us a better mask than the one that used thresholding:
    # create a black image
    img_mask_bgr = copy.deepcopy(img_blank_comp)
    # use the contours to draw filled white shapes
    cv2.drawContours(img_mask_bgr, piece_contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    # create a black and white mask
    lower = np.array([127, 127, 127])
    upper = np.array([255, 255, 255])
    img_mask = cv2.inRange(img_mask_bgr, lower, upper)

    # Now lets combine our mask with the original colour image:
    # Bitwise-AND mask and original image
    img_masked = cv2.bitwise_and(img_input, img_input, mask=img_mask)

    return piece_contours, img_mask_bgr, img_mask, img_masked


def contourFilter(contours):
    """Filters out identified contours that are too big or too small to be pieces."""
    # Calculate the enclosed area of each contour
    contours_area = []
    for piece in range(len(contours)):
        area = cv2.contourArea(contours[piece])
        contours_area.append(area)
    contours_area_sorted = copy.deepcopy(contours_area)
    contours_area_sorted.sort(reverse=True)

    # select a contour we know is a puzzle piece to be the standard
    standard_contour_size = contours_area_sorted[5]
    contour_size_min = 0.2 * standard_contour_size
    contour_size_max = 5 * standard_contour_size

    # only keep pieces with roughly the same area as the standard
    piece_contours = []
    for i in range(len(contours)):
        if ((contours_area[i] > contour_size_min) & (contours_area[i] < contour_size_max)):
            piece_contours.append(contours[i])
    return piece_contours


def approxContours(contours, epsilon_val, state=True):
    """Approximates a contour with less points to within a variance threshold."""
    if epsilon_val == 0:
        approx_contours = contours
    else:
        approx_contours = []
        for piece in range(0, len(contours)):
            approx_contour_set = []
            for side in range(0, 4):
                approx_contour = cv2.approxPolyDP(contours[piece][side], epsilon=epsilon_val, closed=state)
                approx_contour = approx_contour
                new_contour = []
                for index in range(len(approx_contour)):
                    entry = approx_contour[index][0]
                    new_contour.append(entry)
                approx_contour_set.append(np.asarray(new_contour))
            approx_contours.append(np.asarray(approx_contour_set))
    return approx_contours
