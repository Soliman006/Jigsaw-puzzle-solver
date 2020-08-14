""" Processed Data"""
import cv2
import copy


def processedData(input_data, img_blank, processed_edge_types, all_corners_rotated, settings):
    """Creates an image of the contours, colour-coded by side type."""
    processed_pieces = copy.deepcopy(input_data)
    img_processed_segments = copy.deepcopy(img_blank)
    for index in range(len(processed_pieces)):
        for segment in range(0, 4):
            if processed_edge_types[index][segment] == 0:
                cv2.polylines(img=img_processed_segments, pts=[processed_pieces[index][segment]], isClosed=0, color=(
                    255, 0, 0), thickness=settings.line_thickness)
            if processed_edge_types[index][segment] == 1:
                cv2.polylines(img=img_processed_segments, pts=[processed_pieces[index][segment]], isClosed=0, color=(
                    0, 255, 0), thickness=settings.line_thickness)
            if processed_edge_types[index][segment] == 2:
                cv2.polylines(img=img_processed_segments, pts=[processed_pieces[index][segment]], isClosed=0, color=(
                    0, 0, 255), thickness=settings.line_thickness)
            if processed_edge_types[index][segment] == 3:
                cv2.polylines(img=img_processed_segments, pts=[processed_pieces[index][segment]], isClosed=0, color=(
                    255, 0, 255), thickness=settings.line_thickness)
        for corner in range(0, 4):
            point = all_corners_rotated[index][corner]
            cv2.circle(img=img_processed_segments, center=tuple(point),
                       radius=settings.point_radius, color=[0, 255, 255], thickness=-1)

    return processed_pieces, img_processed_segments
