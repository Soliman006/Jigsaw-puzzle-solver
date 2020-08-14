""" Piece Types """
import cv2
import numpy as np
import copy

from global_settings import globalSettings
settings = globalSettings()


def setTypeLists(piece_type):
    """Creates separate lists for corner, edge and interior pieces."""
    interior = []
    edges = []
    corners = []
    for piece in range(len(piece_type)):
        entry = piece_type[piece]
        if entry == 0:
            interior.append(piece)
        if entry == 1:
            edges.append(piece)
        if entry == 2:
            corners.append(piece)
    return interior, edges, corners


def pieceType(defects_f, best_rectangles_index, piece_contours, img_blank):
    """Determines the type of each piece, and the type of each side in a piece."""
    # Lets sort the defect_index:
    for piece in range(len(defects_f)):
        defects_f[piece].sort()
    lock_defects = defects_f

    empty_sides = []
    defects_by_side = []

    for piece in range(len(lock_defects)):
        piece_empty_sides = []
        piece_defects_by_side = []

        for side in range(0, 4):
            side_defects = []

            for point in range(len(lock_defects[piece])):
                index = lock_defects[piece][point]

                if side == 3:
                    if ((index > best_rectangles_index[piece][side]) & (index < len(piece_contours[piece]))) | \
                            ((index > 0) & (index < best_rectangles_index[piece][0])):
                        side_defects.append(np.asarray(index))
                else:
                    if ((index > best_rectangles_index[piece][side]) & (index < best_rectangles_index[piece][side + 1])):
                        side_defects.append(np.asarray(index))

            if len(side_defects) == 0:
                index = -1
                side_defects.append(np.asarray(index))
                piece_empty_sides.append(side)
            piece_defects_by_side.append(np.asarray(side_defects))
        empty_sides.append(piece_empty_sides)
        defects_by_side.append(np.asarray(piece_defects_by_side))

# Allocate each piece a type. 0=normal piece, 1=edge piece, 2= corner piece:
    piece_type = []
    for piece in range(len(empty_sides)):
        piece_type.append(len(empty_sides[piece]))

# Now lets create separate lists for the different kinds of pieces:
    interior, edges, corners = setTypeLists(piece_type)

# Now lets store the individual side types within each piece:
    edge_type = []
    for piece in range(len(defects_by_side)):
        piece_defect_count = []
        for side in range(0, 4):
            entry = defects_by_side[piece][side][0]
            if entry == -1:
                defect_count = 0
            else:
                defect_count = len(defects_by_side[piece][side])
            piece_defect_count.append(np.asarray(defect_count))
        edge_type.append(np.asarray(piece_defect_count))

# Display an image showing the classification of edges and corners:
    img_piece_type = copy.deepcopy(img_blank)
    # use the contours to draw filled white shapes
    for piece in range(len(piece_contours)):
        if piece_type[piece] == 0:
            cv2.drawContours(img_piece_type, piece_contours, piece,
                             (255, 255, 255), thickness=cv2.FILLED)
        if piece_type[piece] == 1:
            cv2.drawContours(img_piece_type, piece_contours, piece,
                             (0, 255, 0), thickness=cv2.FILLED)
        if piece_type[piece] == 2:
            cv2.drawContours(img_piece_type, piece_contours, piece,
                             (0, 0, 255), thickness=cv2.FILLED)

    return img_piece_type, piece_type, edge_type, defects_by_side, interior, edges, corners


def puzzleSize(piece_type):
    """Calculated the expected size, dimentions and piece count of the puzzle."""
    unique, counts = np.unique(piece_type, return_counts=True)
    piece_type_count = dict(zip(unique, counts))
    standard_piece_count = int(piece_type_count[0])
    edge_piece_count = int(piece_type_count[1])
    corner_piece_count = int(piece_type_count[2])

    outside_piece_count = edge_piece_count + corner_piece_count
    inside_piece_count = standard_piece_count
    total_piece_count = inside_piece_count + outside_piece_count

    det = (outside_piece_count + 4)**2 - 16 * total_piece_count
    puzzle_columns = int((outside_piece_count + 4 - np.sqrt(det)) / 4)
    puzzle_columns2 = int((outside_piece_count + 4 + np.sqrt(det)) / 4)
    if puzzle_columns2 > puzzle_columns:
        puzzle_columns = puzzle_columns2
    puzzle_rows = int(total_piece_count / puzzle_columns)

    return puzzle_rows, puzzle_columns, corner_piece_count, edge_piece_count, standard_piece_count
