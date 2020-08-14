""" Side Separator """
import cv2
import numpy as np
import copy


def sideSeparator(best_rectangles_index, piece_contours, img_blank, settings):
    """Splits a contour at the corner points to create 4 side contours."""
    for piece in range(len(best_rectangles_index)):
        best_rectangles_index[piece].sort()

    all_curves = []
    for piece in range(len(piece_contours)):
        curves = []
        for segment in range(len(best_rectangles_index[0]) - 1):
            curve = []
            start = best_rectangles_index[piece][segment]
            end = best_rectangles_index[piece][segment + 1]
            for point in range(start, end + 1):
                entry = np.asarray(piece_contours[piece][point][0])
                curve.append(entry)
            curves.append(np.asarray(curve))
        for segment in range(len(best_rectangles_index[piece]) - 1, len(best_rectangles_index[piece])):
            curve = []
            start = best_rectangles_index[piece][segment]
            end = best_rectangles_index[piece][0]
            for point in range(start, len(piece_contours[piece])):
                entry = np.asarray(piece_contours[piece][point][0])
                curve.append(entry)
            for point in range(0, end + 1):
                entry = np.asarray(piece_contours[piece][point][0])
                curve.append(entry)
            curves.append(np.asarray(curve))
        all_curves.append(np.asarray(curves))

    # Let's show them off:
    img_segments = copy.deepcopy(img_blank)
    for piece in range(len(all_curves)):
        for segment in range(len(all_curves[piece])):
            if segment == 0:
                cv2.polylines(img=img_segments, pts=[all_curves[piece][segment]], isClosed=0, color=(
                    255, 0, 0), thickness=settings.line_thickness)
            if segment == 1:
                cv2.polylines(img=img_segments, pts=[all_curves[piece][segment]], isClosed=0, color=(
                    0, 255, 0), thickness=settings.line_thickness)
            if segment == 2:
                cv2.polylines(img=img_segments, pts=[all_curves[piece][segment]], isClosed=0, color=(
                    0, 0, 255), thickness=settings.line_thickness)
            if segment == 3:
                cv2.polylines(img=img_segments, pts=[all_curves[piece][segment]], isClosed=0, color=(
                    255, 0, 255), thickness=settings.line_thickness)

    # Let's calculate the arc lengths of each of the sides. It may be useful when it comes to comparing edges later:
    side_lengths = []
    for piece in range(len(all_curves)):
        piece_lengths = []
        for segment in range(len(all_curves[piece])):
            length = cv2.arcLength(all_curves[piece][segment], 0)
            piece_lengths.append(length)
        side_lengths.append(piece_lengths)

    return all_curves, side_lengths
