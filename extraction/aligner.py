""" Aligner """
import cv2
import numpy as np
import copy
from utils import edgeOrientation, move_contour, rotate_contour
from utils import rotate_contour_segment
from global_settings import globalSettings
settings = globalSettings()


def reorderSegments(piece_curves, edge1):
    """reorders the list of edge contours in a piece so that a flat edge is always first."""
    temp0 = copy.deepcopy(piece_curves[0])
    temp1 = copy.deepcopy(piece_curves[1])
    temp2 = copy.deepcopy(piece_curves[2])
    temp3 = copy.deepcopy(piece_curves[3])
    piece_curves_new = copy.deepcopy(piece_curves)
    if edge1 == 1:
        piece_curves_new[0] = temp1
        piece_curves_new[1] = temp2
        piece_curves_new[2] = temp3
        piece_curves_new[3] = temp0
    if edge1 == 2:
        piece_curves_new[0] = temp2
        piece_curves_new[1] = temp3
        piece_curves_new[2] = temp0
        piece_curves_new[3] = temp1
    if edge1 == 3:
        piece_curves_new[0] = temp3
        piece_curves_new[1] = temp0
        piece_curves_new[2] = temp1
        piece_curves_new[3] = temp2
    return piece_curves_new


def aligner(radius_max, best_rectangles_sorted, corners, edges, interior, piece_contours, img_blank, edge_type, centers, all_curves,
            puzzle_rows, puzzle_columns, settings):
    """Takes the piece contours from how the are positioned in the original image and rotates them and puts them on a grid."""
    # Now we calculate where the new center points will be:
    grid_centers = []
    for piece in range(len(piece_contours)):
        x = 2 * radius_max + (2 * (piece % puzzle_columns) * radius_max)
        y = 2 * (int(piece / puzzle_columns)) * radius_max + 2 * radius_max
        point = []
        point.append(x)
        point.append(y)
        grid_centers.append(point)

    # We calculate the rotation angle of the piece:
    angles = []
    for piece in range(len(best_rectangles_sorted)):
        angle = []
        low_flag = 0
        high_flag = 0
        for side in range(0, 4):
            x1 = best_rectangles_sorted[piece][side][0]
            y1 = best_rectangles_sorted[piece][side][1]
            x2 = best_rectangles_sorted[piece][(side + 1) % 4][0]
            y2 = best_rectangles_sorted[piece][(side + 1) % 4][1]
            num = y2 - y1
            den = x2 - x1
            if den != 0:
                theta_r = np.arctan(num / den)
            else:
                if num > 0:
                    theta_r = np.pi / 2
                    print("positive divide by 0 encountered at piece", piece, "side", side, "Don't worry it has been handled")
                else:
                    theta_r = -np.pi / 2
                    print("negative divide by 0 encountered at piece", piece, "side", side, "Don't worry it has been handled")
            theta_d = theta_r * 180 / (np.pi)
            theta_d = theta_d % 90
            angle.append(theta_d)
            if theta_d < 20:
                low_flag = 1
            if theta_d > 70:
                high_flag = 1
            if side == 0:
                if ((num > 0) and (den > 0)):  # pos
                    rot = 180
                if ((num > 0) and (den < 0)):  # neg
                    rot = 270
                if ((num < 0) and (den < 0)):  # pos
                    rot = 45
                if ((num < 0) and (den > 0)):  # neg
                    rot = 45
                if ((num > 0) and (den == 0)):  # +ve pi/2
                    rot = 270
                if ((num < 0) and (den == 0)):  # -ve pi/2
                    rot = 270
        angle0 = angle[0]
        if low_flag and high_flag:
            print("piece", piece, "has dual flags")
            for side in range(len(angle)):
                if angle[side] < 20:
                    angle[side] = angle[side] + 90
        av_angle = sum(angle) / len(angle)
        av_angle = av_angle % 90
        if angle0 < 20 and av_angle > 70:
            rot = rot - 90
        if angle0 > 70 and av_angle < 20:
            rot = rot + 90
        av_angle = av_angle + rot
        angles.append(av_angle)

    # rotate contours
    contours_rotated = []
    for index in range(len(corners)):
        piece = corners[index]
        count, edge1, edge2 = edgeOrientation(piece, edge_type)
        angle = angles[piece] - (90 * edge1)
        contour_moved = move_contour(piece_contours[piece], centers[piece], grid_centers[index])
        contour_rotated = rotate_contour(contour_moved, grid_centers[index], angle)
        contour_rotated = reorderSegments(contour_rotated, edge1)
        contours_rotated.append(contour_rotated)

    for index in range(len(edges)):
        piece = edges[index]
        count, edge1, edge2 = edgeOrientation(piece, edge_type)
        angle = angles[piece] - (90 * edge1)
        contour_moved = move_contour(
            piece_contours[piece], centers[piece], grid_centers[index + len(corners)])
        contour_rotated = rotate_contour(contour_moved, grid_centers[index + len(corners)], angle)
        contour_rotated = reorderSegments(contour_rotated, edge1)
        contours_rotated.append(contour_rotated)

    for index in range(len(interior)):
        piece = interior[index]
        count, edge1, edge2 = edgeOrientation(piece, edge_type)
        angle = angles[piece] - (90 * edge1)
        contour_moved = move_contour(
            piece_contours[piece], centers[piece], grid_centers[index + len(corners) + len(edges)])
        contour_rotated = rotate_contour(
            contour_moved, grid_centers[index + len(corners) + len(edges)], angle)
        contour_rotated = reorderSegments(contour_rotated, edge1)
        contours_rotated.append(contour_rotated)

    img_processed_pieces = copy.deepcopy(img_blank)
    cv2.drawContours(img_processed_pieces, contours_rotated, -1, (255, 255, 255), thickness=cv2.FILLED)

    # rotate segments
    all_segments_rotated = []
    for index in range(len(corners)):
        piece = corners[index]
        curves_rotated = []
        count, edge1, edge2 = edgeOrientation(piece, edge_type)
        for side in range(0, 4):
            segment_rotated = move_contour(
                all_curves[piece][side], centers[piece], grid_centers[index])
            angle = angles[piece] - (90 * edge1)
            segment_rotated = rotate_contour_segment(segment_rotated, grid_centers[index], angle)
            curves_rotated.append(segment_rotated)
        curves_rotated = reorderSegments(curves_rotated, edge1)
        all_segments_rotated.append(curves_rotated)

    for index in range(len(edges)):
        piece = edges[index]
        curves_rotated = []
        count, edge1, edge2 = edgeOrientation(piece, edge_type)
        for side in range(0, 4):
            segment_rotated = move_contour(
                all_curves[piece][side], centers[piece], grid_centers[index + len(corners)])
            angle = angles[piece] - (90 * edge1)
            segment_rotated = rotate_contour_segment(
                segment_rotated, grid_centers[index + len(corners)], angle)
            curves_rotated.append(segment_rotated)
        curves_rotated = reorderSegments(curves_rotated, edge1)
        all_segments_rotated.append(curves_rotated)

    for index in range(len(interior)):
        piece = interior[index]
        curves_rotated = []
        for side in range(0, 4):
            segment_rotated = move_contour(
                all_curves[piece][side], centers[piece], grid_centers[index + len(corners) + len(edges)])
            angle = angles[piece] + 90
            segment_rotated = rotate_contour_segment(
                segment_rotated, grid_centers[index + len(corners) + len(edges)], angle)
            curves_rotated.append(segment_rotated)
        all_segments_rotated.append(curves_rotated)

    # corners rotated
    all_corners_rotated = []
    for index in range(len(corners)):
        piece = corners[index]
        count, edge1, edge2 = edgeOrientation(piece, edge_type)
        points_rotated = move_contour(
            best_rectangles_sorted[piece], centers[piece], grid_centers[index])
        angle = angles[piece] - (90 * edge1)
        points_rotated = rotate_contour_segment(points_rotated, grid_centers[index], angle)
        points_rotated = reorderSegments(points_rotated, edge1)
        all_corners_rotated.append(points_rotated)

    for index in range(len(edges)):
        piece = edges[index]
        count, edge1, edge2 = edgeOrientation(piece, edge_type)
        points_rotated = move_contour(
            best_rectangles_sorted[piece], centers[piece], grid_centers[index + len(corners)])
        angle = angles[piece] - (90 * edge1)
        points_rotated = rotate_contour_segment(
            points_rotated, grid_centers[index + len(corners)], angle)
        points_rotated = reorderSegments(points_rotated, edge1)
        all_corners_rotated.append(points_rotated)

    for index in range(len(interior)):
        piece = interior[index]
        count, edge1, edge2 = edgeOrientation(piece, edge_type)
        points_rotated = move_contour(
            best_rectangles_sorted[piece], centers[piece], grid_centers[index + len(corners) + len(edges)])
        angle = angles[piece] + 90
        points_rotated = rotate_contour_segment(
            points_rotated, grid_centers[index + len(corners) + len(edges)], angle)
        all_corners_rotated.append(points_rotated)

    # processed edge types
    processed_edge_types = []
    for index in range(len(corners)):
        piece = corners[index]
        count, edge1, edge2 = edgeOrientation(piece, edge_type)
        types = reorderSegments(edge_type[piece], edge1)
        processed_edge_types.append(types)

    for index in range(len(edges)):
        piece = edges[index]
        count, edge1, edge2 = edgeOrientation(piece, edge_type)
        types = reorderSegments(edge_type[piece], edge1)
        processed_edge_types.append(types)

    for index in range(len(interior)):
        piece = interior[index]
        types = edge_type[piece]
        processed_edge_types.append(types)

    # display
    img_align_segments = copy.deepcopy(img_blank)
    for index in range(len(corners) + len(edges) + len(interior)):
        for segment in range(0, 4):
            if segment == 0:
                cv2.polylines(img=img_align_segments, pts=[all_segments_rotated[index][segment]], isClosed=0, color=(
                    255, 0, 0), thickness=settings.line_thickness)
            if segment == 1:
                cv2.polylines(img=img_align_segments, pts=[all_segments_rotated[index][segment]], isClosed=0, color=(
                    0, 255, 0), thickness=settings.line_thickness)
            if segment == 2:
                cv2.polylines(img=img_align_segments, pts=[all_segments_rotated[index][segment]], isClosed=0, color=(
                    0, 0, 255), thickness=settings.line_thickness)
            if segment == 3:
                cv2.polylines(img=img_align_segments, pts=[all_segments_rotated[index][segment]], isClosed=0, color=(
                    255, 0, 255), thickness=settings.line_thickness)
        for corner in range(0, 4):
            point = all_corners_rotated[index][corner]
            cv2.circle(img=img_align_segments, center=tuple(point),
                       radius=settings.point_radius, color=[0, 255, 255], thickness=-1)

    return grid_centers, angles, contours_rotated, all_segments_rotated, all_corners_rotated, processed_edge_types, img_align_segments
