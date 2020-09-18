""" Corner Finder """
import cv2
import numpy as np
import copy

from global_settings import globalSettings
settings = globalSettings()

# here we take the low point count approximate hull contour, and generate polygons for every combination of 4 hull points on a piece:


def cornerFinder(poi_corner, piece_contours, img_mask_bgr, settings):
    """Finds the corners of each piece by maximising the area and 'rectangularness' of the 4 point polygon created by the corners."""
    rectangles_index = []
    rectangles = []
    for piece in range(len(poi_corner)):
        piece_rectangles_index = []
        piece_rectangles = []
        for index1 in range(len(poi_corner[piece])):
            for index2 in range(len(poi_corner[piece])):
                if index2 == index1:
                    break
                for index3 in range(len(poi_corner[piece])):
                    if ((index3 == index1) | (index3 == index2)):
                        break
                    for index4 in range(len(poi_corner[piece])):
                        if ((index4 == index1) | (index4 == index2) | (index4 == index3)):
                            break
                        polygon_index = []
                        polygon = []
                        pointA = np.asarray(poi_corner[piece][index1][0])
                        pointB = np.asarray(poi_corner[piece][index2][0])
                        pointC = np.asarray(poi_corner[piece][index3][0])
                        pointD = np.asarray(poi_corner[piece][index4][0])
                        polygon_index.append(index1)
                        polygon_index.append(index2)
                        polygon_index.append(index3)
                        polygon_index.append(index4)
                        polygon.append(pointA)
                        polygon.append(pointB)
                        polygon.append(pointC)
                        polygon.append(pointD)
                        piece_rectangles_index.append(np.asarray(polygon_index))
                        piece_rectangles.append(np.asarray(polygon))
        rectangles_index.append(np.asarray(piece_rectangles_index))
        rectangles.append(np.asarray(piece_rectangles))

# Now we go through all those polygons and calculate their area and rectangularity.
# An overall score is then calculated. (score = area x rectangularity^1.5)
    rectangles_area = []
    rectangularity = []
    scores = []
    for piece in range(len(rectangles)):
        piece_rectangles_area = []
        piece_rectangularity = []
        piece_scores = []
        for group in range(len(rectangles[piece])):
            sq_ratio = squareness(rectangles[piece][group])
            if sq_ratio >= 0.75:
                area = cv2.contourArea(rectangles[piece][group])
                min_bounding_rectangle = cv2.minAreaRect(rectangles[piece][group])
                box_points = cv2.boxPoints(min_bounding_rectangle)
                bounding_area = cv2.contourArea(box_points)
                if bounding_area == 0:
                    poly_rectangularity = 0
                else:
                    poly_rectangularity = area / bounding_area

                # combine and save
                score = area * (poly_rectangularity**1.5) * sq_ratio
                piece_rectangles_area.append(area)
                piece_rectangularity.append(poly_rectangularity)
                piece_scores.append(score)
            else:
                piece_rectangles_area.append(0)
                piece_rectangularity.append(0)
                piece_scores.append(0)
        rectangles_area.append(piece_rectangles_area)
        rectangularity.append(piece_rectangularity)
        scores.append(piece_scores)

# Now we make a shortlist of the highest scores for each piece:
    scores_max = []
    for piece in range(len(scores)):
        scores_max.append(np.max(scores[piece]))

# Now we iterate through all the polygons, and when we find the polygon with a score matching the max score of the piece, it is saved.
    best_rectangles = []
    for piece in range(len(rectangles)):
        best_polygon_contenders = []
        for group in range(len(scores[piece])):
            if scores[piece][group] == scores_max[piece]:
                best_polygon_contenders.append(group)
        best_rectangles.append(rectangles[piece][best_polygon_contenders[0]])

# Now that we have our best rectangles, we compare each point to the piece contour,
# and save the index of the point on the piece contour that the point in the rectangle is closest to:
    best_rectangles_index = []
    for piece in range(len(best_rectangles)):
        points_on_contour = []
        for point in range(len(best_rectangles[piece])):
            dist_2 = np.sum((piece_contours[piece] - best_rectangles[piece][point])**2, axis=1)
            distances = []
            for entry in range(len(dist_2)):
                dist = dist_2[entry][0]**2 + dist_2[entry][1]**2
                distances.append(dist)
            point_on_contour = np.argmin(distances)
            points_on_contour.append(np.asarray(point_on_contour))
        best_rectangles_index.append(np.asarray(points_on_contour))

    best_rectangles_sorted = []
    for piece in range(len(best_rectangles_index)):
        best_rectangles_index[piece].sort()
        best_rectangles_sorted_piece = []
        for index in range(len(best_rectangles_index[piece])):
            point = piece_contours[piece][best_rectangles_index[piece][index]][0]
            best_rectangles_sorted_piece.append(point)
        best_rectangles_sorted.append(np.asarray(best_rectangles_sorted_piece))

# Let's also find the average dimensions of the rectangles:
    side_length = 0
    for piece in range(len(best_rectangles)):
        length = cv2.arcLength(best_rectangles[piece], closed=True)
        side_length = side_length + 0.25 * length
    av_length = side_length / len(best_rectangles)
    av_length = int(av_length)

# Now we display the 4 point polygons that scored the highest:
    img_corners = copy.deepcopy(img_mask_bgr)
    for piece in range(len(best_rectangles_sorted)):
        cv2.drawContours(img_corners, best_rectangles_sorted, -1,
                         (0, 0, 255), thickness=settings.line_thickness)

    return best_rectangles_sorted, best_rectangles_index, av_length, img_corners

def squareness(points):
    """Doc."""
    sides = []
    for index in range(len(points)):
        distances = []
        for i in range(len(points)-1):
            if i is not index:
                x, y, e = dist(points[index], points[i])
                distances.append(e)
        distances.remove(max(distances))
        sides = sides + distances
    len_min = min(sides)
    len_max = max(sides)
    ratio = len_min/len_max
    return ratio

def dist(point1, point2):
    """Calculates the horizontal, vertical and euclidian distance between 2 points."""
    x_dist = point1[0] - point2[0]
    y_dist = point1[1] - point2[1]
    euclidean_dist = np.sqrt((x_dist)**2 + (y_dist)**2)
    return x_dist, y_dist, euclidean_dist
