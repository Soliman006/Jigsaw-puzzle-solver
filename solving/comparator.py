import numpy as np
from utils import move_contour, cart2pol, pol2cart, imageResize
from graphics import imshow

""" Overarching Comparison """


def normaliseContours(contour1, contour2, av_length):
    """Takes 2 contours that are going to be compared and tries to orientate them overlap with one another."""
    contour1, peak_point1 = normaliseContour(contour1, 0, av_length)
    contour2, peak_point2 = normaliseContour(contour2, 1, av_length)
    """ NHA is making everything worse
    peak1_x = peak_point1[0]
    peak2_x = peak_point2[0]
    peak_diff = peak1_x - peak2_x
    contour2 = move_contour(contour2, [0, 0], [peak_diff, 0])
    peak_point2[0] = peak_point2[0] + peak_diff
    """
    return contour1, contour2, peak_point1, peak_point2


def normaliseContour(contour, dir, av_length):
    """Tries to move a contour to a known standard position and orientation."""
    # normalise the contour by:
    # moving the contour so that point 0 is on the origin
    origin = [100, 100]
    if dir == 1:
        norm_curve = move_contour(contour, contour[0], origin)
    else:
        norm_curve = move_contour(contour, contour[len(contour) - 1], origin)
    #   rotating so that the last point is on the positive horizontal axis
    origin = [100, 100]
    norm_curve = norm_curve - origin
    coordinates = norm_curve[:, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    if dir == 1:
        angle = thetas[len(thetas) - 1]
    else:
        angle = thetas[0]
    thetas = (thetas - angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    norm_curve[:, 0] = xs
    norm_curve[:, 1] = ys
    norm_curve = norm_curve + origin
    norm_curve = norm_curve.astype(np.int32)

    trim_length = 0.1 * av_length
    if dir == 1:
        trim_point_approx_start = [norm_curve[0][0] + trim_length, norm_curve[0][1]]
        trim_point_approx_end = [norm_curve[len(contour) - 1][0] - trim_length, norm_curve[len(contour) - 1][1]]
    else:
        trim_point_approx_start = [norm_curve[len(contour) - 1][0] + trim_length, norm_curve[len(contour) - 1][1]]
        trim_point_approx_end = [norm_curve[0][0] - trim_length, norm_curve[0][1]]

    trim_index_start, min_point, min_e_dist, min_x_dist, min_y_dist = closestDist(trim_point_approx_start, norm_curve)
    trim_index_end, min_point, min_e_dist, min_x_dist, min_y_dist = closestDist(trim_point_approx_end, norm_curve)

    trim_point_start = norm_curve[trim_index_start]
    # trim_point_end = norm_curve[trim_index_end]

    norm_curve = move_contour(norm_curve, trim_point_start, origin)
    #   rotating so that the last point is on the positive horizontal axis
    origin = [100, 100]
    norm_curve = norm_curve - origin
    coordinates = norm_curve[:, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    angle = thetas[trim_index_end]
    thetas = (thetas - angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    max_y = 0
    max_index_list = []
    for index in range(len(norm_curve)):
        point = norm_curve[index]
        if (abs(point[1]) > max_y):
            max_y = abs(point[1])
    for index in range(len(norm_curve)):
        point = norm_curve[index]
        if (abs(point[1]) >= 0.9 * max_y):
            max_index_list.append(index)

    norm_curve[:, 0] = xs
    norm_curve[:, 1] = ys
    norm_curve = norm_curve + origin

    norm_curve = move_contour(norm_curve, origin, trim_point_start)
    norm_curve = move_contour(norm_curve, trim_point_start, [trim_point_start[0], origin[1]])

    peak_point = [0, 0]
    for i in range(len(max_index_list)):
        peak_point = peak_point + norm_curve[max_index_list[i]]
    peak_point = peak_point/len(max_index_list)
    peak_point_int = [0, 0]
    peak_point_int[0] = int(peak_point[0])
    peak_point_int[1] = int(peak_point[1])
    peak_point_int = np.asarray(peak_point_int)

    norm_curve = norm_curve.astype(np.int32)

    return norm_curve, peak_point_int


def compareContours(contour1, contour2, colour_curve1, colour_curve2, colour_contour_xy1, colour_contour_xy2, av_length, settings):
    """Takes 2 contours and compares them on shape and colour. A lower score indicates a better match."""
    score_shape = compareShapeContours(contour1, contour2) / (av_length * settings.score_shape_scalar)
    score_colour = compareColourContours(colour_contour_xy1, colour_contour_xy2, colour_curve1, colour_curve2, settings)
    score_total = (settings.score_mult_shape * score_shape) + \
        (settings.score_mult_colour * score_colour)
    return score_shape, score_colour, score_total


""" Shape Comparison """


def dist(point1, point2):
    """Calculates the horizontal, vertical and euclidian distance between 2 points."""
    x_dist = point1[0] - point2[0]
    y_dist = point1[1] - point2[1]
    euclidean_dist = np.sqrt((x_dist)**2 + (y_dist)**2)
    return x_dist, y_dist, euclidean_dist


def closestDist(point1, curve):
    """Finds the closest point in a list of points to a specified coordinate."""
    min_point = 100000
    min_index = 100000
    min_e_dist = 100000
    min_x_dist = 100000
    min_y_dist = 100000
    for index in range(len(curve)):
        point2 = curve[index]
        x_dist, y_dist, e_dist = dist(point1, point2)
        if e_dist < min_e_dist:
            min_e_dist = e_dist
            min_x_dist = x_dist
            min_y_dist = y_dist
            min_point = point2
            min_index = index
    return min_index, min_point, min_e_dist, min_x_dist, min_y_dist


def compareShapeContours(contour1, contour2):
    """Calculates a shape match score between 2 contours.
    The score is based on the average of the closest distance one curve is from every point on the other curve."""
    score = 0
    x_disp = 0
    y_disp = 0
    x_score = 0
    y_score = 0
    for index in range(len(contour1)):
        point1 = contour1[index]
        index2, point2, e_dist, x_dist, y_dist = closestDist(point1, contour2)
        score = score + e_dist
        x_score = x_score + np.abs(x_dist)
        y_score = y_score + np.abs(y_dist)
        x_disp = x_disp + x_dist
        y_disp = y_disp + y_dist
    score = score / len(contour1)
    x_score = x_score / len(contour1)
    y_score = y_score / len(contour1)
    x_score = x_score - x_disp
    y_score = y_score - y_disp
    return score


""" Colour Comparison """


def colourDist(colour1, colour2):
    """Calculates the difference between 2 colours."""
    b_dist = colour1[0] - colour2[0]
    g_dist = colour1[1] - colour2[1]
    r_dist = colour1[2] - colour2[2]
    euclidean_dist = np.sqrt((b_dist)**2 + (g_dist)**2 + (r_dist)**2)
    return b_dist, g_dist, r_dist, euclidean_dist


def colourClosestDist(point1, curve1, colour1, curve2, colour_curve2):
    """Calculates the colour difference between a point on one curve and the point on another curve that is physically closest."""
    min_dist = 100000
    min_index = -1
    for index2 in range(len(curve2)):
        point2 = curve2[index2]
        x_dist, y_dist, e_dist = dist(point1, point2)
        if e_dist < min_dist:
            min_dist = e_dist
            min_index = index2
    colour2 = colour_curve2[min_index]
    b_dist, g_dist, r_dist, e_dist = colourDist(colour1, colour2)
    return e_dist, colour2


def compareColourContours(contour1, contour2, colour_curve1, colour_curve2, settings):
    """Calculates the colour match between 2 contours.
    The score is based on the average of the how similar the colour of a each point on a curve is
    from the colour of the physically closest point on the other curve."""
    score = 0
    w = int(settings.inc - 1)
    width = w * len(contour1)
    height = 20
    img_colour = np.zeros([height, width, 3], dtype=np.uint8)
    for index in range(len(contour1)):
        point1 = contour1[index]
        colour1 = colour_curve1[index]
        dist, colour2 = colourClosestDist(point1, contour1, colour1, contour2, colour_curve2)
        img_colour[0:10, w * index:w * index + w] = colour1
        img_colour[10:21, w * index:w * index + w] = colour2
        score = score + dist
    score = score / len(contour1)
    score = score / settings.score_colour_scalar  # max possible score is 441, thus convert to range of 0-1.
    if settings.show_colour_comparison:
        imshow(imageResize(img_colour, height=height), settings.env)
        print(" ")
    return score
