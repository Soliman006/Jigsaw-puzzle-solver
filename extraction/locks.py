""" Locks Searcher """
import cv2
import copy


def locksSearcher(defects_by_side, corner_piece_count, edge_piece_count, standard_piece_count, img_mask_bgr, all_curves, settings):
    """Counts the number of inner locks, outer locks and flat edges across all pieces and creates a colour coded image."""
    outer_count = 0
    inner_count = 0
    edge_count = 0

    for piece in range(len(defects_by_side)):
        for side in range(0, 4):
            count = len(defects_by_side[piece][side])
            if count == 2:
                outer_count = outer_count + 1
            if count == 1:
                if defects_by_side[piece][side] == -1:
                    edge_count = edge_count + 1
                else:
                    inner_count = inner_count + 1

    # The results above should equal the theoretical results here:
    # theory_lock_count = int((2*corner_piece_count+3*edge_piece_count+4*standard_piece_count)/2)
    # print("Outer Locks",theory_lock_count,"Inner Locks",theory_lock_count,"Edges",2*corner_piece_count+edge_piece_count)

    # Now let's split the piece contour into sides:
    img_locks = copy.deepcopy(img_mask_bgr)
    for piece in range(len(all_curves)):
        for segment in range(len(all_curves[piece])):
            count = len(defects_by_side[piece][segment])
            if count == 2:
                # outer lock

                cv2.drawContours(img_locks, all_curves[piece], segment, (0, 0, 255), -1)
            if count == 1:
                if defects_by_side[piece][segment] == -1:
                    # edge
                    cv2.drawContours(
                        img_locks, all_curves[piece], segment, (255, 0, 0), thickness=settings.line_thickness)
                else:
                    # inner lock
                    cv2.drawContours(img_locks, all_curves[piece], segment, (0, 255, 0), -1)

    return img_locks, outer_count, inner_count, edge_count
