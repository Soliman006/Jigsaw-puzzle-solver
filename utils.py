''' General Utility Functions '''
import cv2
import numpy as np
import copy
import os
from global_settings import globalSettings


def imageResize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Upsamples/Downsamples an image to a required pixel width and/or height."""
    dim = None
    (h, w) = image.shape[:2]
    if height is None and width is None:
        return image
    if height is None:
        if w == width:
            return image
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        if h == height:
            return image
        r = height / float(h)
        dim = (int(w * r), height)
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def zoom(image, center, zoom):
    """Crops an image to a square segment given by the center point and zoom = 0.5*height."""
    x = center[0]
    y = center[1]
    height = 2 * zoom
    width = 2 * zoom
    h_min = y - zoom
    h_max = y + zoom
    w_min = x - zoom
    w_max = x + zoom
    cropped = np.zeros([height, width, 3], dtype=np.uint8)
    cropped = image[h_min:h_max, w_min:w_max]
    return cropped


def retrieveExample(i):
    settings = globalSettings()
    i_str = str(i)
    filename = i_str + '_unsolved.jpg'
    absolute_path = os.path.join(os.getcwd(), 'datasets', filename)
    img = cv2.imread(absolute_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        filename = i_str + '_unsolved.png'
        absolute_path = os.path.join(os.getcwd(), 'datasets', filename)
        img = cv2.imread(absolute_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("could not find image ", i_str, "_unsolved")
    # settings
    if i == 1:
        settings.bg_thresh_low = hsv_to_cvhsv(180, 50, 45)
        settings.bg_thresh_high = hsv_to_cvhsv(220, 100, 100)
        settings.e_contour_smoothing = 0.5
        settings.score_thresh = 1.72  # if no options are under this then it trigger the backtracker
        settings.max_options = 3
        settings.inc = 5
    elif i == 2:
        settings.e_contour_smoothing = 0.5
        settings.max_options = 2
        settings.interpolate_ref = False
    elif i == 3:
        settings.compute_height = 993
        settings.e_contour_smoothing = 0.5
        settings.max_options = 3
    elif i == 4:
        settings.e_contour_smoothing = 0.5
        settings.score_thresh = 1.9  # if no options are under this then it trigger the backtracker
        settings.helper_threshold = 1.3  # all options within this multiplier are considered
        settings.max_options = 2
        settings.interpolate_ref = False
        settings.interpolate_cand = False
    elif i == 5:
        settings.e_contour_smoothing = 0.5
        settings.max_options = 1
        settings.interpolate_ref = False
        settings.interpolate_cand = False
    elif i == 7:
        settings.bg_thresh_low = hsv_to_cvhsv(23, 3, 60)
        settings.bg_thresh_high = hsv_to_cvhsv(60, 15, 95)
    elif i == 8:
        settings.bg_thresh_low = hsv_to_cvhsv(23, 3, 60)
        settings.bg_thresh_high = hsv_to_cvhsv(60, 15, 95)
        
    elif i == 10: # dataset 10 216 piece
        settings.compute_height = 3000
        settings.convexity_epsilon = 1500
        #settings.bg_thresh_low = hsv_to_cvhsv(0, 0, 0)
        #settings.bg_thresh_high = hsv_to_cvhsv(360, 1, 1)
        settings.e_contour_smoothing = 0.5
        settings.inc = 5
        #settings.shape_score_frac = 0.7
        #settings.score_colour_thresh = 0.7
        settings.score_thresh = 0.4 # if no options are under this then it triggers the backtracker
        settings.helper_threshold = 1.2 # all options within this multiplier are considered
        settings.max_options = 2


    elif i == 11: # dataset 11 96 piece
        #settings.bg_thresh_low = hsv_to_cvhsv(0, 0, 0)
        #settings.bg_thresh_high = hsv_to_cvhsv(360, 1, 1)
        settings.compute_height = 2000
        settings.convexity_epsilon = 1500
        settings.e_contour_smoothing = 0.5
        settings.inc = 5
        #settings.shape_score_frac = 0.6
        #settings.score_colour_thresh = 0.6
        settings.score_thresh = 0.3 # if no options are under this then it triggers the backtracker
        settings.helper_threshold = 1.3 # all options within this multiplier are considered
        settings.max_options = 1


    elif i == 12: #dataset 12 88 piece
        #settings.bg_thresh_low = hsv_to_cvhsv(0, 0, 0)
        #settings.bg_thresh_high = hsv_to_cvhsv(360, 1, 1)
        settings.compute_height = 2500
        settings.convexity_epsilon = 1500
        settings.e_contour_smoothing = 0.5
        settings.inc = 5
        #settings.shape_score_frac = 0.6
        #settings.corner_thresh = 1.5
        settings.score_colour_thresh = 1
        settings.score_thresh = 0.4 # if no options are under this then it triggers the backtracker
        settings.helper_threshold = 1.3 # all options within this multiplier are considered
        settings.max_options = 2

    elif i == 13: # dataset 13 140 piece
        #settings.bg_thresh_low = hsv_to_cvhsv(0, 0, 0)
        #settings.bg_thresh_high = hsv_to_cvhsv(360, 1, 1)
        settings.compute_height = 3000
        settings.convexity_epsilon = 1500
        settings.e_contour_smoothing = 0.5
        settings.inc = 5
        #settings.shape_score_frac = 0.6
        #settings.corner_thresh = 1.5
        settings.score_colour_thresh = 0.7
        settings.score_thresh = 0.4 # if no options are under this then it triggers the backtracker
        settings.helper_threshold = 1.3 # all options within this multiplier are considered
        settings.max_options = 2

    elif i == 14: # dataset 14 532 piece
        #settings.bg_thresh_low = hsv_to_cvhsv(0, 0, 0)
        #settings.bg_thresh_high = hsv_to_cvhsv(360, 1, 1)
        settings.compute_height = 4000
        settings.convexity_epsilon = 1500
        settings.e_contour_smoothing = 0.5
        settings.inc = 5
        #settings.shape_score_frac = 0.6
        #settings.corner_thresh = 1.5
        settings.score_colour_thresh = 0.6
        settings.score_thresh = 0.4 # if no options are under this then it triggers the backtracker
        settings.helper_threshold = 1.3 # all options within this multiplier are considered
        settings.max_options = 2

    elif i == 15: # dataset 15 330 piece
        settings.compute_height = 3500
        settings.convexity_epsilon = 1500
        #settings.bg_thresh_low = hsv_to_cvhsv(0, 0, 0)
        #settings.bg_thresh_high = hsv_to_cvhsv(360, 1, 1)
        settings.e_contour_smoothing = 0.5
        settings.inc = 5
        #settings.shape_score_frac = 0.7
        #settings.score_colour_thresh = 0.7
        settings.score_thresh = 0.4 # if no options are under this then it triggers the backtracker
        settings.helper_threshold = 1.2 # all options within this multiplier are considered
        settings.max_options = 2

    '''elif i == 17: #dataset 17 49 7x7
        settings.bg_thresh_low = hsv_to_cvhsv(100, 40, 10)
        settings.bg_thresh_high = hsv_to_cvhsv(150, 80, 70)
        settings.convexity_epsilon = 2500
        settings.e_contour_smoothing = 0.5
        settings.inc = 5
        #settings.shape_score_frac = 0.6
        settings.score_colour_thresh = 1
        settings.score_thresh = 0.4 # if no options are under this then it triggers the backtracker
        settings.helper_threshold = 1.3 # all options within this multiplier are considered
        settings.max_options = 2'''
        
        elif i == 24:
            settings.bg_thresh_low = hsv_to_cvhsv(85, 30, 30)
            settings.bg_thresh_high = hsv_to_cvhsv(140, 100, 100)
            settings.e_contour_smoothing = 0.5
            settings.score_thresh = 1.9  # if no options are under this then it trigger the backtracker
            settings.helper_threshold = 1.3  # all options within this multiplier are considered
            settings.max_options = 2
            settings.interpolate_ref = False
            settings.interpolate_cand = False
        
    return img, settings


def retrieveImage(filename):
    """Imports one of the built-in example images."""
    absolute_path = os.path.join(os.getcwd(), 'datasets', filename)
    img = cv2.imread(absolute_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("could not find image")
    return img


def cart2pol(x, y):
    """Converts from cartesian to polar coordinates."""
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    """Converts from polar to cartesian coordinates."""
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def edgeOrientation(piece, edge_type):
    """Determines which edges of a piece are borders."""
    edge1_index = -1
    edge2_index = -1
    count = 0
    for side in range(0, 4):
        if edge_type[piece][side] == 0:
            count = count + 1
            if count == 1:
                edge1_index = side
            else:
                edge2_index = side
    if edge1_index == 0 and edge2_index == 3:
        temp = edge1_index
        edge1_index = edge2_index
        edge2_index = temp
    return count, edge1_index, edge2_index


def move_contour(contour, center_old, center_new):
    """Moves a contour based on the relative distance between 2 points."""
    contour_norm = contour - center_old
    contour_moved = contour_norm + center_new
    contour_moved = contour_moved.astype(np.int32)
    return contour_moved


def rotate_contour(contour, center, angle):
    """Rotates a contour by a certain angle about a certain coordinate."""
    contour_norm = contour - center
    coordinates = contour_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)
    thetas = np.rad2deg(thetas)
    thetas = (thetas - angle) % 360
    thetas = np.deg2rad(thetas)
    xs, ys = pol2cart(thetas, rhos)
    contour_norm[:, 0, 0] = xs
    contour_norm[:, 0, 1] = ys
    contour_rotated = contour_norm + center
    contour_rotated = contour_rotated.astype(np.int32)
    return contour_rotated


def rotate_contour_segment(contour, center, angle):
    """Rotates a segment/curve by a certain angle about a certain coordinate."""
    contour_norm = contour - center
    coordinates = contour_norm[:, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)
    thetas = np.rad2deg(thetas)
    thetas = (thetas - angle)
    thetas = np.deg2rad(thetas)
    xs, ys = pol2cart(thetas, rhos)
    contour_norm[:, 0] = xs
    contour_norm[:, 1] = ys
    contour_rotated = contour_norm + center
    contour_rotated = contour_rotated.astype(np.int32)
    return contour_rotated


def move_bgr(contour_old, image_old, center_old, image_new, center_new, img_blank):
    """Moves a BGR image based on the relative distance between 2 points."""
    h_old, w_old, ch_old = image_old.shape
    h_new, w_new, ch_new = image_new.shape
    img_blank_old = np.zeros([h_old, w_old, ch_old], dtype=np.uint8)
    img_blank_new = np.zeros([h_new, w_new, ch_new], dtype=np.uint8)

    # create an image of just the piece at it's current location.
    img_piece_mask_bgr = copy.deepcopy(img_blank_old)
    cv2.drawContours(img_piece_mask_bgr, [contour_old], -1, (255, 255, 255), thickness=cv2.FILLED)
    lower = np.array([127, 127, 127])
    upper = np.array([255, 255, 255])
    img_piece_mask = cv2.inRange(img_piece_mask_bgr, lower, upper)
    img_piece_masked = cv2.bitwise_and(image_old, image_old, mask=img_piece_mask)

    # create a mask of just that piece, in it's final orientation.
    moved_contour = move_contour(contour_old, center_old, center_new)
    contour_new = moved_contour
    img_piece_mask_moved_bgr = copy.deepcopy(img_blank_new)
    cv2.drawContours(img_piece_mask_moved_bgr, [moved_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    lower = np.array([127, 127, 127])
    upper = np.array([255, 255, 255])
    img_piece_mask_moved = cv2.inRange(img_piece_mask_moved_bgr, lower, upper)

    # move the image in img_piece_masked to its final orientation. Use a bounding rectange and some sort of copy function
    movement = copy.deepcopy(center_old)
    movement[0] = center_new[0] - center_old[0]
    movement[1] = center_new[1] - center_old[1]
    img_piece_masked_moved = copy.deepcopy(img_blank_new)  # img_piece_masked)
    locs = np.where(img_piece_masked != 0)  # Get the non-zero mask locations
    img_piece_masked_moved[locs[0] + movement[1], locs[1] + movement[0]] = img_piece_masked[locs[0], locs[1]]

    # then mask the image onto the master collection of pieces:
    locs = np.where(img_piece_mask_moved != 0)  # Get the non-zero mask locations
    image_new[locs[0], locs[1]] = img_piece_masked_moved[locs[0], locs[1]]
    return image_new, contour_new


def rotate_bgr(contour_old, image_old, center_old, image_new, angle, img_blank):
    """Rotates a BGR image by a certain angle about a certain point."""
    h_old, w_old, ch_old = image_old.shape
    h_new, w_new, ch_new = image_new.shape
    img_blank_old = np.zeros([h_old, w_old, ch_old], dtype=np.uint8)
    img_blank_new = np.zeros([h_new, w_new, ch_new], dtype=np.uint8)

    img_piece_mask_bgr = copy.deepcopy(img_blank_old)
    cv2.drawContours(img_piece_mask_bgr, [contour_old], -1, (255, 255, 255), thickness=cv2.FILLED)
    lower = np.array([127, 127, 127])
    upper = np.array([255, 255, 255])
    img_piece_mask = cv2.inRange(img_piece_mask_bgr, lower, upper)
    img_piece_masked = cv2.bitwise_and(image_old, image_old, mask=img_piece_mask)

    # create a mask of just that piece, in it's final orientation.
    rotated_contour = rotate_contour(contour_old, center_old, angle)
    contour_new = rotated_contour
    img_piece_mask_rot_bgr = copy.deepcopy(img_blank_new)
    cv2.drawContours(img_piece_mask_rot_bgr, [rotated_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    lower = np.array([127, 127, 127])
    upper = np.array([255, 255, 255])
    img_piece_mask_rot = cv2.inRange(img_piece_mask_rot_bgr, lower, upper)

    # move the image in img_piece_masked to its final orientation. Use a bounding rectange and some sort of copy function
    rot_mat = cv2.getRotationMatrix2D(tuple(center_old), angle, 1.0)
    img_piece_masked_rot = cv2.warpAffine(img_piece_masked, rot_mat, img_piece_masked.shape[1::-1], flags=cv2.INTER_LINEAR)

    # then mask the image onto the master collection of pieces:
    locs = np.where(img_piece_mask_rot != 0)  # Get the non-zero mask locations
    image_new[locs[0], locs[1]] = img_piece_masked_rot[locs[0], locs[1]]
    return image_new, contour_new


def hsv_to_cvhsv(h, s, v):
    """Converts typical HSV ranges of (0<H<360,0<S<100,0<V<100) to the ranges opencv expects of (0<H<179,0<S<255,0<V<255)"""
    cv_h = int(179 * h / 360)
    cv_s = int(255 * s / 100)
    cv_v = int(255 * v / 100)
    colour = np.array([cv_h, cv_s, cv_v])
    return colour


def takeFourth(elem):
    """Returns the fourth element."""
    return elem[3]
