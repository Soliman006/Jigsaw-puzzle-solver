import cv2
import numpy as np
import copy
import imageio

from utils import imageResize, move_contour, rotate_contour, move_bgr, rotate_bgr, edgeOrientation

from IPython import display
from PIL import Image as PIL_Image


def imshow(input, env):
    """Called when the program want's to display an image. Handles how to display it depending on the environment."""
    if env == 'JUPYTER':
        cv2_imshow(input)  # for colab
    if env == 'DEFAULT':
        cv2.imshow('', input)  # native cv2
        cv2.waitKey(0)


def cv2_imshow(a):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
    Args:
        a : np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
        (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color image.
    """
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display.display(PIL_Image.fromarray(a))


def createSpacedSolution(data, puzzle):
    """Creates contours of the solution, with pieces spaced apart."""
    solution_contours = []
    for y in range(0, puzzle.y_limit):
        for x in range(0, puzzle.x_limit):
            piece = puzzle.loc[y][x]
            if piece != -1:
                x_val = data.radius_max + 2 * x * data.radius_max
                y_val = data.radius_max + 2 * y * data.radius_max
                puzzle.centers_solution[piece] = [x_val, y_val]
                solution_contour = move_contour(
                    data.contours_rotated[piece], data.grid_centers[piece], puzzle.centers_solution[piece])
                angle = puzzle.rotation[y][x]
                if piece < (len(data.corners) + len(data.edges)):  # it's a border
                    angle = -angle
                else:
                    angle = -angle - 90
                solution_contour = rotate_contour(solution_contour, puzzle.centers_solution[piece], angle)
            solution_contours.append(solution_contour)
    return solution_contours


def displaySpacedSolution(solution_contours, radius_max, x_limit, y_limit, settings):
    """Displays the information created by createSpacedSolution."""
    width = 2 * radius_max * x_limit
    height = 2 * radius_max * y_limit
    img_solution_mask = np.zeros([height, width, 3], dtype=np.uint8)
    cv2.drawContours(img_solution_mask, solution_contours, -1,
                     (0, 0, 255), thickness=settings.line_thickness)
    cv2_imshow(imageResize(img_solution_mask, height=200))


def createSolution(data, puzzle):
    """Creates the contours of the pieces as the have been placed in the puzzle by the solver."""
    solution_contours = []
    for y in range(0, puzzle.y_limit):
        for x in range(0, puzzle.x_limit):
            piece = puzzle.loc[y][x]
            if piece != -1:
                x_val = data.av_length + x * data.av_length
                y_val = data.av_length + y * data.av_length
                puzzle.centers_solution[piece] = [x_val, y_val]
                solution_contour = move_contour(
                    data.contours_rotated[piece], data.grid_centers[piece], puzzle.centers_solution[piece])
                angle = puzzle.rotation[y][x]
                if piece < (len(data.corners) + len(data.edges)):  # it's a border
                    angle = -angle  # weird shit
                else:
                    angle = -angle - 90  # even weirder shit
                solution_contour = rotate_contour(solution_contour, puzzle.centers_solution[piece], angle)
            solution_contours.append(solution_contour)
    return solution_contours


def displaySolution(solution_contours, av_length, x_limit, y_limit, settings):
    """Displays the information created by createSolution."""
    width = av_length * x_limit + av_length
    height = av_length * y_limit + av_length
    img_solution_mask = np.zeros([height, width, 3], dtype=np.uint8)  # copy.deepcopy(img_blank)
    cv2.drawContours(img_solution_mask, solution_contours, -1,
                     (0, 0, 255), thickness=settings.line_thickness)
    cv2_imshow(imageResize(img_solution_mask, height=200))


def createBGRSolution(data, puzzle):
    """Creates a BGR colour image of the pieces as they have been placed in the puzzle."""
    img_solution_bgr0 = copy.deepcopy(data.img_blank_comp)
    img_solution_bgr = copy.deepcopy(data.img_blank_comp)
    solution_contours = []
    for y in range(0, puzzle.y_limit):
        for x in range(0, puzzle.x_limit):
            piece = puzzle.loc[y][x]
            if piece != -1:
                x_val = data.av_length + x * data.av_length
                y_val = data.av_length + y * data.av_length
                puzzle.centers_solution[piece] = [x_val, y_val]
                img_solution_bgr0, contour_new = move_bgr(
                    data.contours_rotated[piece], data.img_processed_bgr, data.grid_centers[piece], img_solution_bgr0,
                    puzzle.centers_solution[piece], data.img_blank_comp)
                angle = puzzle.rotation[y][x]
                if piece < (len(data.corners) + len(data.edges)):  # it's a border
                    angle = -angle  # weird shit
                else:
                    angle = -angle - 90  # even weirder shit
                img_solution_bgr, contour_new = rotate_bgr(
                    contour_new, img_solution_bgr0, puzzle.centers_solution[piece], img_solution_bgr, angle, data.img_blank_comp)
                solution_contours.append(contour_new)
    return img_solution_bgr, solution_contours


def displayBGRSolution(img_solution_bgr, av_length, x_limit, y_limit, settings):
    """Displays the information created by createBGRSolution."""
    width = av_length * x_limit + av_length
    height = av_length * y_limit + av_length
    cropped = np.zeros([height, width, 3], dtype=np.uint8)
    cropped = img_solution_bgr[:height, :width]
    cv2_imshow(imageResize(cropped, height=settings.disp_height))


def createGIFSequential(data, puzzle, filename='sequential_solution.gif'):
    """Creates and saves a GIF of the order in which the solver has placed the pieces in the puzzle."""
    n_placed_pieces = np.size(puzzle.loc)-np.sum(puzzle.loc == -1)
    img_solution_bgr0 = copy.deepcopy(data.img_blank_comp)
    img_solution_bgr1 = copy.deepcopy(data.img_blank_comp)
    img_solution_bgr = copy.deepcopy(data.img_blank_comp)
    solution_contours = []
    img_sequential_rgb = []
    for index in range(0, n_placed_pieces):
        piece = puzzle.placement_order_piece[index]
        [x, y] = puzzle.placement_order_space[index]
        if piece != -1:
            x_val = data.av_length + x * data.av_length
            y_val = data.av_length + y * data.av_length
            puzzle.centers_solution[piece] = [x_val, y_val]
            img_solution_bgr0, contour_new = move_bgr(
                data.contours_rotated[piece], data.img_processed_bgr, data.grid_centers[piece], img_solution_bgr0,
                puzzle.centers_solution[piece], data.img_blank_comp)
            angle = puzzle.rotation[y][x]
            if piece < (len(data.corners) + len(data.edges)):  # it's a border
                angle = -angle  # weird shit
            else:
                angle = -angle - 90  # even weirder shit
            img_solution_bgr1, contour_new = rotate_bgr(
                contour_new, img_solution_bgr0, puzzle.centers_solution[piece], img_solution_bgr1, angle, data.img_blank_comp)
            img_solution_bgr, contour_new = move_bgr(
                contour_new, img_solution_bgr1, puzzle.centers_solution[piece], img_solution_bgr,
                puzzle.centers_solution[piece], data.img_blank_comp)
            width = data.av_length * puzzle.x_limit + data.av_length
            height = data.av_length * puzzle.y_limit + data.av_length
            cropped = np.zeros([height, width, 3], dtype=np.uint8)
            cropped = img_solution_bgr[:height, :width]
            solution_contours.append(contour_new)
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            img_sequential_rgb.append(cropped_rgb)
            if index == (n_placed_pieces-1):
                for i in range(0, 21):
                    img_sequential_rgb.append(cropped_rgb)
    imageio.mimsave(filename, img_sequential_rgb)


def createGIFTransformation(data, puzzle, filename='tranformation_solution.gif'):
    """Creates a GIF showing all the pieces moving from their start positions to their end positions at the same time."""
    # create mapping between original piece numbering and grid piece numbering
    grid2orig = []
    count = 0
    for index in range(0, len(data.corners)):
        grid2orig.append(data.corners[index])
    for index in range(0, len(data.edges)):
        grid2orig.append(data.edges[index])
    for index in range(0, len(data.interior)):
        grid2orig.append(data.interior[index])

    # create image of original pieces showing only pieces that exist in the solution
    n_placed_pieces = np.size(puzzle.loc)-np.sum(puzzle.loc == -1)

    # generic GIF settings
    duration = 3
    fps = 30
    gif_fps = 1*fps
    print("GIF FPS", gif_fps)
    n_frames = duration*fps

    # centering
    h_orig, w_orig, ch_orig = data.img_masked.shape
    center_orig = [int(w_orig/2), int(h_orig/2)]
    w_final = data.av_length * puzzle.x_limit + data.av_length
    h_final = data.av_length * puzzle.y_limit + data.av_length
    center_final = [int(w_final/2), int(h_final/2)]
    center_offset = [center_orig[0] - center_final[0], center_orig[1] - center_final[1]]

    # create pathways
    paths = []
    for piece in range(0, puzzle.num_pieces):
        start_pos = data.centers[grid2orig[piece]]
        path = []
        if puzzle.placed_pieces[piece] == 1:
            end_pos = puzzle.centers_solution[piece]
            end_pos[0] = end_pos[0] + center_offset[0]
            end_pos[1] = end_pos[1] + center_offset[1]
            for inc in range(0, n_frames+1):
                dx = int((inc/n_frames)*(end_pos[0]-start_pos[0]))
                dy = int((inc/n_frames)*(end_pos[1]-start_pos[1]))
                point = [start_pos[0] + dx, start_pos[1] + dy]
                path.append(point)
        else:
            for inc in range(0, n_frames+1):
                path.append(start_pos)
        paths.append(path)

    # create angle trajectory
    transition_angles = []
    for piece in range(0, puzzle.num_pieces):
        count, edge1, edge2 = edgeOrientation(grid2orig[piece], data.edge_type)
        start_angle = data.angles[grid2orig[piece]]
        start_angle = start_angle - (90 * edge1)
        transition_angle = []
        if puzzle.placed_pieces[piece] == 1:
            end_angle = start_angle
            for y in range(puzzle.y_limit):
                for x in range(puzzle.x_limit):
                    if puzzle.loc[y, x] == piece:
                        end_angle = puzzle.rotation[y, x]
                        if piece < (len(data.corners) + len(data.edges)):  # it's a border
                            end_angle = -end_angle  # end_angle = -end_angle  # weird shit
                        else:
                            end_angle = -end_angle - 90  # end_angle = -end_angle - 90  # even weirder shit
            full_angle = end_angle + start_angle
            if full_angle < -180:
                full_angle = full_angle + 360
            if full_angle > 180:
                full_angle = full_angle - 360
            for inc in range(0, n_frames+1):
                da = (inc/n_frames)*(full_angle)
                angle = 0 + da
                transition_angle.append(angle)
        else:
            for inc in range(0, n_frames+1):
                transition_angle.append(0)
        transition_angles.append(transition_angle)

    # create image sequence
    img_trans_rgb = []
    for inc in range(0, n_frames + 1):
        img_solution_bgr = copy.deepcopy(data.img_blank_comp)
        for index in range(0, n_placed_pieces):
            img_solution_bgr0 = copy.deepcopy(data.img_blank_comp)
            img_solution_bgr1 = copy.deepcopy(data.img_blank_comp)
            piece = puzzle.placement_order_piece[index]
            [x, y] = puzzle.placement_order_space[index]
            if piece != -1:
                x_val = data.av_length + x * data.av_length
                y_val = data.av_length + y * data.av_length
                puzzle.centers_solution[piece] = [x_val, y_val]
                img_solution_bgr0, contour_new = move_bgr(
                    data.piece_contours[grid2orig[piece]], data.img_masked, data.centers[grid2orig[piece]], img_solution_bgr0,
                    paths[piece][inc], data.img_blank_comp)
                img_solution_bgr1, contour_new = rotate_bgr(
                    contour_new, img_solution_bgr0, paths[piece][inc], img_solution_bgr1,
                    transition_angles[piece][inc], data.img_blank_comp)
                img_solution_bgr, contour_new = move_bgr(
                    contour_new, img_solution_bgr1, paths[piece][inc], img_solution_bgr,
                    paths[piece][inc], data.img_blank_comp)
        img_downsampled = imageResize(img_solution_bgr, height=puzzle.settings.disp_height)
        cropped_rgb = cv2.cvtColor(img_downsampled, cv2.COLOR_BGR2RGB)
        img_trans_rgb.append(cropped_rgb)
        print("GIF Progress", inc+1, "/", n_frames+1)
        if inc == n_frames:
            for i in range(0, 3*fps):
                img_trans_rgb.append(cropped_rgb)
    imshow(imageResize(img_solution_bgr, height=puzzle.settings.disp_height))
    # save GIF
    print("Saving GIF")
    imageio.mimsave(filename, img_trans_rgb, fps=gif_fps)
    print("GIF creation finished")
