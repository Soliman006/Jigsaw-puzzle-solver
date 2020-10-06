import numpy as np
import random
import math
from graphics import imshow
from utils import imageResize, move_bgr, rotate_bgr
from maker.utils import move_bgr, rotate_bgr
import cv2
import copy


class PuzzleMaker:
    """Makes puzzles from a given image."""

    def __init__(self, img, n_rows, settings):
        """Initialisation."""
        self.img_input = img
        self.img_input_h, self.img_input_w, self.img_input_ch = self.img_input.shape
        print(self.img_input_h, self.img_input_w, self.img_input_ch)
        self.background_colour = np.array([0, 255, 0])
        self.img_input_blank = np.full((self.img_input_h, self.img_input_w), self.background_colour, dtype=np.uint8)
        #self.img_input_blank = np.zeros([self.img_input_h, self.img_input_w, self.img_input_ch], dtype=np.uint8)
        print(self.img_input_blank)
        self.n_rows = n_rows
        self.settings = settings
        self.contours = []
        self.ch = 3
        imshow(imageResize(self.img_input_blank, height=self.settings.disp_height), self.settings.env)
        imshow(imageResize(self.img_input, height=self.settings.disp_height), self.settings.env)
        self.run()

    def run(self):
        """Main code for creating a puzzle from an image."""
        self.img_input_h, self.img_input_w, self.img_input_ch = self.img_input.shape
        print("Original Image -", "width:", self.img_input_w, "height:", self.img_input_h, "channels:", self.img_input_ch)
        self.dimensions()
        print("The puzzle will have", self.n_pieces, "pieces,", self.n_rows, "rows and", self.n_cols, "columns")
        self.create_pieces()
        self.create_contours()
        self.random_placement()

    def dimensions(self):
        """Finds the optimal number of rows and columns, and the piece side length."""
        aspect_ratio = self.img_input_w / self.img_input_h
        self.n_cols = int(math.floor(self.n_rows * aspect_ratio))
        self.n_pieces = self.n_rows * self.n_cols
        self.mapping = list(range(0,self.n_pieces))
        self.side_len = int(math.floor(self.img_input_h / self.n_rows))
        self.img_puzzle_h = self.n_rows * self.side_len
        self.img_puzzle_w = self.n_cols * self.side_len
        self.img_result_h = int(2 * self.n_rows * self.side_len + 2 * self.side_len)
        self.img_result_w = int(2 * self.n_cols * self.side_len + 2 * self.side_len)
        self.result_grid = []
        for piece in range(self.n_pieces):
            for y in range(self.n_rows):
                for x in range(self.n_cols):
                    px_x = int(2 * self.side_len + 2 * x * self.side_len)
                    px_y = int(2 * self.side_len + 2 * y * self.side_len)
                    point = [px_x, px_y]
                    self.result_grid.append(point)
        self.img_result_blank = np.ones([self.img_result_h, self.img_result_w, self.ch], dtype=np.uint8)
        self.puzzle = [[False for i in range(self.n_cols)] for j in range(self.n_rows)]

    def create_pieces(self):
        for iy in range(self.n_rows):
            for ix in range(self.n_cols):
                if (ix == 0 or self.puzzle[iy][ix - 1] is False):
                    piece_left = False
                else:
                    piece_left = self.puzzle[iy][ix - 1]
                if (ix == self.n_cols - 1 or self.puzzle[iy][ix + 1] is False):
                    piece_right = False
                else:
                    piece_right = self.puzzle[iy][ix + 1]
                if (iy == 0 or self.puzzle[iy - 1][ix] is False):
                    piece_down = False
                else:
                    piece_down = self.puzzle[iy - 1][ix]
                if (iy == self.n_rows - 1 or self.puzzle[iy+1][ix] is False):
                    piece_up = False
                else:
                    piece_up = self.puzzle[iy][ix+1]
                self.puzzle[iy][ix] = Piece(self.side_len, self.n_cols, self.n_rows, ix, iy, piece_left=piece_left, piece_up=piece_up,
                                            piece_right=piece_right, piece_down=piece_down)

    def create_contours(self):
        self.contours = []
        self.centers_orig = []
        for y in range(len(self.puzzle)):
            for x in range(len(self.puzzle[y])):
                contour = []
                segments = self.puzzle[y][x].boundary
                center = self.find_center(segments)
                self.centers_orig.append(center)
                for side in range(len(segments)):
                    for index in range(len(segments[side])):
                        point = segments[side][index]
                        contour.append(np.asarray(point))
                self.contours.append(np.asarray(contour))

    def random_placement(self):
        random.shuffle(self.mapping)
        self.angles = []
        for piece in range(self.n_pieces):
            angle = random.uniform(0,360)
            self.angles.append(angle)


    def overlay(self):
        img_overlay = copy.deepcopy(self.img_input)
        cv2.drawContours(img_overlay, self.contours, -1, (255, 255, 255), thickness=2)
        imshow(imageResize(img_overlay, height=self.settings.disp_height), self.settings.env)

    def display_result(self):
        img_result = copy.deepcopy(self.img_result_blank)
        img_temp = copy.deepcopy(self.img_result_blank)
        for piece in range(len(self.contours)):
            random_map = self.mapping[piece]
            img_temp, contour_new = move_bgr(self.contours[piece], self.img_input, self.centers_orig[piece],
                     img_temp, self.result_grid[random_map], self.img_result_blank)
            img_result, contour_new = rotate_bgr(contour_new, img_temp, self.result_grid[random_map],
                                                        img_result, self.angles[piece], self.img_result_blank)
        imshow(img_result, self.settings.env)

    def find_center(self, segments):
        sum_x = 0
        sum_y = 0
        count = 0
        for side in range(len(segments)):
            start_x = segments[side][0][0]
            start_y = segments[side][0][1]
            end_x = segments[side][-1][0]
            end_y = segments[side][-1][1]
            sum_x = sum_x + start_x + end_x
            sum_y = sum_y + start_y + end_y
            count = count + 2
        av_x = int(sum_x / count)
        av_y = int(sum_y / count)
        center = [av_x, av_y]
        return center


class Piece:
    """Docstring."""

    def __init__(self, side_len, n_cols, n_rows, piece_nx, piece_ny, piece_left=False, piece_up=False, piece_right=False, piece_down=False):
        self.side_len = side_len
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.piece_nx = piece_nx
        self.piece_ny = piece_ny
        self.is_in_up = random.choice([True, False])
        self.is_in_left = random.choice([True, False])
        self.is_in_down = random.choice([True, False])
        if not self.is_in_up and not self.is_in_left and not self.is_in_down:
            self.is_in_right = True
        else:
            self.is_in_right = random.choice([True, False])
        self.piece_left = piece_left
        self.piece_up = piece_up
        self.piece_right = piece_right
        self.piece_down = piece_down
        self.n_points = 100
        self.run()

    def run(self):
        if self.piece_ny+1 == self.n_rows:
            self.boundary_up = np.array([[self.piece_nx, self.piece_ny+1], [self.piece_nx+1, self.piece_ny+1]])
        else:
            self.boundary_up = self.piece_edge(self.n_points, self.is_in_up, math.radians(0), self.piece_nx, self.piece_ny)
        if self.piece_ny == 0:
            self.boundary_down = np.array([[self.piece_nx+1, self.piece_ny], [self.piece_nx, self.piece_ny]])
        else:
            self.boundary_down = self.piece_edge(self.n_points, self.is_in_up, math.radians(180), self.piece_nx, self.piece_ny)
        if self.piece_nx == 0:
            self.boundary_left = np.array([[self.piece_nx, self.piece_ny], [self.piece_nx, self.piece_ny + 1]])
        else:
            self.boundary_left = self.piece_edge(self.n_points, self.is_in_up, math.radians(270), self.piece_nx, self.piece_ny)
        if self.piece_nx + 1 == self.n_cols:
            self.boundary_right = np.array([[self.piece_nx + 1, self.piece_ny + 1], [self.piece_nx + 1, self.piece_ny]])
        else:
            self.boundary_right = self.piece_edge(self.n_points, self.is_in_up, math.radians(90), self.piece_nx, self.piece_ny)
        if not(not(self.piece_up)):
            self.boundary_up = self.piece_up.boundary_down[::-1]
            self.is_in_up = not(self.piece_up.is_in_down)
        if not(not(self.piece_right)):
            self.boundary_right = self.piece_right.boundary_left[::-1]
            self.is_in_right = not(self.piece_right.is_in_left)
        if not(not(self.piece_down)):
            self.boundary_down = self.piece_down.boundary_up[::-1]
            self.is_in_down = not(self.piece_down.is_in_up)
        if not(not(self.piece_left)):
            self.boundary_left = self.piece_left.boundary_right[::-1]
            self.is_in_left = not(self.piece_left.is_in_right)
        self.boundary = self.get_boundary()

    def get_boundary(self):
        boundary = []
        boundary.append((self.side_len * self.boundary_left).astype(int))
        boundary.append((self.side_len * self.boundary_up).astype(int))
        boundary.append((self.side_len * self.boundary_right).astype(int))
        boundary.append((self.side_len * self.boundary_down).astype(int))
        return boundary

    def piece_edge(self, n_points, is_in, rot_form_top, x, y):
        a = 1.4
        h = 1.25
        w = 4
        rh = 0.6
        rw = 0.6
        ra = 0.1
        var = 0.4
        var2 = 0.2
        s = np.array([[0, 0], [+a+ra*random.uniform(-var, var), 0], [0.5-w+rw*random.uniform(-var, var), h+rh*random.uniform(-var, var)],
                      [0.5+w+rw*random.uniform(-var, var), h+rh*random.uniform(-var, var)], [1-a+ra*random.uniform(-var, var), 0], [1, 0]])
        # Make a random puzzle lock 0,0 to 1,0 up to 1 unit high
        L = self.bezier_curve(s, n_points)
        # Move it and scale it so it is 0.5 wide and flip if is_in
        L = L+[random.uniform(-var2, var2), 0]
        lock_scale = 3.5
        L = L / lock_scale + np.array([(1/lock_scale), 0])
        if is_in:
            L = L * np.array([1, -1])
        L[0][0] = 0
        L[0][1] = 0
        L[-1][0] = 1
        L[-1][1] = 0
        #  centered ofset and then rotate accordingly
        L = L + np.array([-0.5, 0.5])
        r = np.array([[math.cos(rot_form_top), -math.sin(rot_form_top)], [math.sin(rot_form_top), math.cos(rot_form_top)]])
        L = np.dot(L, r)
        L = L + np.array([0.5, 0.5])
        L = L + np.array([x, y])
        return L

    def bezier_curve(self, p, n):
        pn = len(p)
        L = np.zeros((n, 2))
        count = 0
        Lt = np.linspace(0, 1, num=n)
        for t in Lt:
            temp_p = p
            for i in range(1, pn):
                temp_p = temp_p[0:len(temp_p)-1]+(t)*(-temp_p[0:len(temp_p)-1]+temp_p[1:len(temp_p)])
            L[count] = temp_p
            count = count + 1
        return L
