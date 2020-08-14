from utils import edgeOrientation, move_bgr, rotate_bgr
import copy


def bgrData(img_blank, corners, edges, interior, angles, piece_contours, img_masked, centers, grid_centers, edge_type):
    """Takes the colour image of the pieces and rotates them and moves them onto a grid."""
    img_processed_bgr0 = copy.deepcopy(img_blank)
    img_processed_bgr = copy.deepcopy(img_blank)
    processed_bgr = []
    for index in range(len(corners)):
        piece = corners[index]
        count, edge1, edge2 = edgeOrientation(piece, edge_type)
        angle = angles[piece] - (90 * edge1)
        img_processed_bgr0, contour_new = move_bgr(
            piece_contours[piece], img_masked, centers[piece], img_processed_bgr0, grid_centers[index], img_blank)
        img_processed_bgr, contour_new = rotate_bgr(
            contour_new, img_processed_bgr0, grid_centers[index], img_processed_bgr, angle, img_blank)
        processed_bgr.append(contour_new)

    for index in range(len(edges)):
        piece = edges[index]
        count, edge1, edge2 = edgeOrientation(piece, edge_type)
        angle = angles[piece] - (90 * edge1)
        img_processed_bgr0, contour_new = move_bgr(
            piece_contours[piece], img_masked, centers[piece], img_processed_bgr0, grid_centers[index + len(corners)], img_blank)
        img_processed_bgr, contour_new = rotate_bgr(
            contour_new, img_processed_bgr0, grid_centers[index + len(corners)], img_processed_bgr, angle, img_blank)
        processed_bgr.append(contour_new)

    for index in range(len(interior)):
        piece = interior[index]
        count, edge1, edge2 = edgeOrientation(piece, edge_type)
        angle = angles[piece] - (90 * edge1)
        img_processed_bgr0, contour_new = move_bgr(piece_contours[piece], img_masked, centers[piece], img_processed_bgr0,
                                                   grid_centers[index + len(corners) + len(edges)], img_blank)
        img_processed_bgr, contour_new = rotate_bgr(contour_new, img_processed_bgr0, grid_centers[index + len(corners) + len(edges)],
                                                    img_processed_bgr, angle, img_blank)
        processed_bgr.append(contour_new)

    return processed_bgr, img_processed_bgr
