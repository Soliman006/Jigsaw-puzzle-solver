import cv2
import numpy as np
import copy

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

def move_contour(contour, center_old, center_new):
    """Moves a contour based on the relative distance between 2 points."""
    contour_norm = contour - center_old
    contour_moved = contour_norm + center_new
    contour_moved = contour_moved.astype(np.int32)
    return contour_moved


def rotate_contour(contour, center, angle):
    """Rotates a contour by a certain angle about a certain coordinate."""
    contour_norm = contour - center
    coordinates = contour_norm[:, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)
    thetas = np.rad2deg(thetas)
    thetas = (thetas - angle) % 360
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
    #img_blank_new = np.zeros([h_new, w_new, ch_new], dtype=np.uint8)

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
