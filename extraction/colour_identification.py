""" Colour Identification """
import numpy as np
from graphics import imshow, imageResize


def colourIdentification(processed_pieces, img_processed_bgr, settings):
    """Creates contours that reprient the colour along each edge of each piece."""
    colour_offset_min = 1 - settings.inc
    colour_offset_max = settings.inc
    colour_contours = []
    colour_contours_xy = []
    for piece in range(len(processed_pieces)):
        colour_piece = []
        colour_piece_xy = []
        for side in range(len(processed_pieces[piece])):
            colour_side = []
            colour_side_xy = []
            for point in range(0, len(processed_pieces[piece][side]), settings.inc):
                bgr_sum = np.array([0, 0, 0])
                count = 0
                xy = processed_pieces[piece][side][point]
                colour_side_xy.append(xy)
                x = processed_pieces[piece][side][point][0]
                y = processed_pieces[piece][side][point][1]
                for y_shift in range(colour_offset_min, colour_offset_max):
                    for x_shift in range(colour_offset_min, colour_offset_max):
                        if ((img_processed_bgr[y + y_shift][x + x_shift][0] != 0) or
                                (img_processed_bgr[y + y_shift][x + x_shift][1] != 255) or
                                (img_processed_bgr[y + y_shift][x + x_shift][2] != 0)):
                            count = count + 1
                            bgr_sum = bgr_sum + img_processed_bgr[y + y_shift][x + x_shift]
                if count != 0:
                    bgr_sum = bgr_sum / count
                bgr_sum = bgr_sum.astype(np.int)
                colour_side.append(np.asarray(bgr_sum))

            point = len(processed_pieces[piece][side]) - 1
            bgr_sum = np.array([0, 0, 0])
            count = 0
            xy = processed_pieces[piece][side][point]
            colour_side_xy.append(xy)
            x = processed_pieces[piece][side][point][0]
            y = processed_pieces[piece][side][point][1]
            for y_shift in range(colour_offset_min, colour_offset_max):
                for x_shift in range(colour_offset_min, colour_offset_max):
                    if ((img_processed_bgr[y + y_shift][x + x_shift][0] != 0) or
                        (img_processed_bgr[y + y_shift][x + x_shift][1] != 255)
                            or (img_processed_bgr[y + y_shift][x + x_shift][2] != 0)):
                        count = count + 1
                        bgr_sum = bgr_sum + img_processed_bgr[y + y_shift][x + x_shift]
            if count != 0:
                bgr_sum = bgr_sum / count
            bgr_sum = bgr_sum.astype(np.int)
            colour_side.append(np.asarray(bgr_sum))

            colour_piece.append(np.asarray(colour_side))
            colour_piece_xy.append(np.asarray(colour_side_xy))

            if (settings.show_extracted_colours):
                colour_disp_w = int(settings.inc - 1)  # 10,1
                width = colour_disp_w * len(colour_side)
                height = 10
                img_colour = np.zeros([height, width, 3], dtype=np.uint8)
                for x in range(len(colour_side)):
                    for y in range(height):
                        for w in range(colour_disp_w):
                            img_colour[y][colour_disp_w * x + w] = colour_side[x]
                imshow(imageResize(img_colour, height=height), settings.env)

        colour_contours.append(np.asarray(colour_piece))
        colour_contours_xy.append(np.asarray(colour_piece_xy))
        if (settings.show_colour_extraction_progress):
            print("Colour Characteristics Extraction Progress:", piece + 1, "/", len(processed_pieces))

    return colour_contours, colour_contours_xy
