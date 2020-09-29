import numpy as np
import statistics as st
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

    '''peak_point = [0, 0]
    for i in range(len(max_index_list)):
        peak_point = peak_point + norm_curve[max_index_list[i]]
    peak_point = peak_point/len(max_index_list)
    peak_point_int = [0, 0]
    peak_point_int[0] = int(peak_point[0])
    peak_point_int[1] = int(peak_point[1])
    peak_point_int = np.asarray(peak_point_int)'''

    norm_curve = norm_curve.astype(np.int32)
    
    peak_point = findlockedge(norm_curve)
    peak_point_int = [0, 0]
    peak_point_int[0] = int(peak_point[0])
    peak_point_int[1] = int(peak_point[1])
    peak_point_int = np.asarray(peak_point_int)

    return norm_curve, peak_point_int

def compareContours(colour_curve1, colour_curve2, colour_contour_xy1, colour_contour_xy2, av_length, settings):
    """Takes 2 contours and compares them on shape and colour. A lower score indicates a better match."""
    #score_shape = compareShapeContours(contour1, contour2) / (av_length * settings.score_shape_scalar)
    score_colour = compareColourContours(colour_contour_xy1, colour_contour_xy2, colour_curve1, colour_curve2, settings)
    #score_total = (settings.score_mult_shape * score_shape) + (settings.score_mult_colour * score_colour)
    return score_colour
    #return score_shape, score_colour, score_total


""" Shape Comparison """
def remove_duplicates(contour):
    contourlist = contour.tolist()
    num_del = 0
    for i in range(1,len(contourlist)):
        j = i - num_del
        if contourlist[j] == contourlist[j-1]:
            del contourlist[j]
            num_del = num_del+1
    
    return contourlist
            
def process_contour(contour):
    contourlist = contour.tolist()
    num_del = 0
    num_add = 0
    test = False
    for i in range(1,len(contourlist)):
        j = i - num_del + num_add
        xd = contourlist[j][0] - contourlist[j-1][0]
        yd = contourlist[j][1] - contourlist[j-1][1]
        
        if contourlist[j] == contourlist[j-1]:
            del contourlist[j]
            num_del = num_del+1
        elif abs(xd) > 1 and abs(xd)>=abs(yd):
            new_points = generate_xpoints(contourlist[j-1],xd,yd)
            contourlist[j:j] = new_points
            num_add = num_add + abs(xd)-1
        elif abs(yd) > 1:
            new_points = generate_ypoints(contourlist[j-1],xd,yd)
            contourlist[j:j] = new_points
            num_add = num_add + abs(yd)-1
     
    return contourlist

def generate_xpoints(startpoint,xd,yd):
    new_points = []
    if xd > 0:
        sign = 1
    else:
        sign = -1
    for i in range(1,abs(xd)):
        point = [startpoint[0]+sign*i,startpoint[1]+(yd/abs(xd)*i)]
        new_points.append(point)
    return new_points

def generate_ypoints(startpoint,xd,yd):
    new_points = []
    if yd > 0:
        sign = 1
    else:
        sign = -1
    for i in range(1,abs(yd)):
        point = [startpoint[0]+(xd/abs(yd)*i),startpoint[1]+sign*i]
        new_points.append(point)
    return new_points
                   
def findeuclid_dist(reference,contour):
    edistances = []
    for i in contour:
        start_dist = edist(reference[0],i)
        end_dist = edist(reference[-1],i)
        if start_dist <= end_dist:
            close_dist = start_dist
            for j in reference:
                new_dist = edist(j,i)
                if  new_dist < close_dist:
                    close_dist = new_dist
                if new_dist > 3*close_dist:
                    break
            if close_dist > 1:
                edistances.append(close_dist)
        else:
            close_dist = end_dist
            for j in reversed(reference):
                new_dist = edist(j,i)
                if  new_dist < close_dist:
                    close_dist = new_dist
                if new_dist > 3*close_dist:
                    break
            if close_dist > 1:
                edistances.append(close_dist)
    if len(edistances) != 0:
        score = sum(edistances)/(len(edistances)*11)
    else:
        score = 0
    return score

def findlockedge(contour):
    currentfar = contour[0]
    otherfar = [0,0]
    fardist = 0
    for i in contour:
        yd = ydist(contour[0],i)
        if yd > fardist:
            fardist = yd
            currentfar = i
            otherfar = [0,0]
        elif yd == fardist:
            otherfar = i
    if otherfar[0] != 0:
        far = [(currentfar[0]+otherfar[0])/2,currentfar[1]]
    else:
        far = [currentfar[0],currentfar[1]]
    return far

def findlocks(contour,far):
    if far[1]>contour[0][1]:
        down = True
        up = False
    else:
        up = True
        down = False
    if contour[0][0] < contour[-1][0]:
        lock_left = contour[0]
        lock_right = contour[-1]
        for i in range(len(contour)):
            current = contour[i]
            nextpoint = contour[i+1]
            if nextpoint[0] < current[0] and ((down and nextpoint[1]>=current[1]) or (up and nextpoint[1]<=current[1])):
                    lock_left = current
                    break
            elif nextpoint[0] == current[0] and ydist(nextpoint,current) > 2:
                lock_left = current
                break
            if current[0] >= far[0]:
                print("left lock not found")
                break
        for i in range(len(contour)):
            current = contour[-i-1]
            nextpoint = contour[-i-2]
            if nextpoint[0] > current[0] and ((down and nextpoint[1]>=current[1]) or (up and nextpoint[1]<=current[1])):
                lock_right = current
                break
            elif nextpoint[0] == current[0] and ydist(nextpoint,current) > 2:
                lock_right = current
                break
            if current[0] <= far[0]:
                print("right lock not found")
                break
    
    else:
        lock_left = contour[-1]
        lock_right = contour[0]
        for i in range(len(contour)):
            current = contour[i]
            nextpoint = contour[i+1]
            if nextpoint[0] > current[0] and ((down and nextpoint[1]>=current[1]) or (up and nextpoint[1]<=current[1])):
                lock_right = current
                break
            elif nextpoint[0] == current[0] and ydist(nextpoint,current) > 2:
                lock_right = current
                break
            if current[0] <= far[0]:
                print("left lock not found")
                break
        for i in range(len(contour)):
            current = contour[-i-1]
            nextpoint = contour[-i-2]
            if nextpoint[0] < current[0] and ((down and nextpoint[1]>=current[1]) or (up and nextpoint[1]<=current[1])):
                lock_left = current
                break
            elif nextpoint[0] == current[0] and ydist(nextpoint,current) > 2:
                lock_left = current
                break
            if current[0] >= far[0]:
                print("right lock not found")
                break
    return lock_left, lock_right
    
        
def xdist(point1, point2):
    x_dist = abs(point1[0] - point2[0])
    return x_dist

def ydist(point1,point2):
    ydist = abs(point1[1] - point2[1])
    return ydist

def edist(point1,point2):
    euclidian_dist = np.sqrt(xdist(point1,point2)**2 + ydist(point1,point2)**2)
    return euclidian_dist

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
    score = score / 150#441.673  # max possible score is 441, thus convert to range of 0-1.
    if settings.show_colour_comparison:
        imshow(imageResize(img_colour, height=height), settings.env)
        print(" ")
    return score
