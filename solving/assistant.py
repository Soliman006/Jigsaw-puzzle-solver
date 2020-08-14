"Assistant"
import numpy as np
import cv2
import copy
from utils import zoom
from graphics import imshow
from utils import imageResize, move_bgr, rotate_bgr


class Assistant:
    """Class that descripes an instance of the prompt that asks a user to pick the correct piece."""

    def __init__(self, point, matches, data, img_bg, x_limit, y_limit, settings):
        """Initialises the prompt with the information of the current instance."""
        self.selection = 0
        self.max_options = 8
        self.point = point
        if len(matches) > self.max_options:
            self.matches = matches[0:self.max_options]
        else:
            self.matches = matches
        self.data = data
        self.img_bg = img_bg
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.settings = settings

    def run(self):
        """Creates and executes the prompt asking the user to pick the correct match."""
        w = self.data.av_length * self.x_limit + self.data.av_length
        h = self.data.av_length * self.y_limit + self.data.av_length
        cropped = np.zeros([h, w, 3], dtype=np.uint8)
        cropped = self.img_bg[:h, :w]

        x = self.point[0]
        y = self.point[1]
        x_val = self.data.av_length + x * self.data.av_length
        y_val = self.data.av_length + y * self.data.av_length
        coordinate = [x_val, y_val]

        h = 2*self.data.av_length
        w = 2*self.data.av_length
        options = []
        scores = np.zeros(self.max_options)
        for i in range(len(self.matches)):
            scores[i] = self.matches[i][3]
            option = self.generateOption(i, coordinate, self.matches[i][1], self.matches[i][2], cropped, scores[i])
            options.append(option)
        if len(self.matches) < self.max_options:
            for i in range(len(self.matches), self.max_options):
                option = np.zeros([h, w, 3], dtype=np.uint8)
                option[0:2, :] = (0, 0, 255)
                option[h-2:h, :] = (0, 0, 255)
                option[:, 0:2] = (0, 0, 255)
                option[:, w-2:w] = (0, 0, 255)
                cv2.circle(img=option, center=tuple([int(0.85*w), int(0.85*h)]),
                           radius=int(0.1*h), color=[255, 255, 255], thickness=-1)
                cv2.putText(option, str(i + 1), (int(0.795*w), int(0.905*h)), self.settings.font1, 0.8, (0, 0, 200), 1, cv2.LINE_AA)
                options.append(option)
                scores[i] = None

        img_options = np.zeros([2*h, 4*w, 3], dtype=np.uint8)

        img_options[0:h, 0:w] = options[0]
        img_options[0:h, w:2*w] = options[1]
        img_options[0:h, 2*w:3*w] = options[2]
        img_options[0:h, 3*w:4*w] = options[3]
        img_options[h:2*h, 0:w] = options[4]
        img_options[h:2*h, w:2*w] = options[5]
        img_options[h:2*h, 2*w:3*w] = options[6]
        img_options[h:2*h, 3*w:4*w] = options[7]
        imshow(imageResize(img_options, height=int(self.settings.disp_height/2)), self.settings.env)

        # get input from user
        print("please enter the correct option (1-8 or none):")
        inputString = input()
        if inputString == '1' or\
                inputString == '2' or\
                inputString == '3' or\
                inputString == '4' or\
                inputString == '5' or\
                inputString == '6' or\
                inputString == '7' or\
                inputString == '8':
            self.selection = int(inputString)
            return self.selection
        else:
            if inputString == 'none':
                self.selection = inputString
                print('The user has indicated that no options are correct')
                return 0
            else:
                print('Invalid Entry')
                return 0

    def generateOption(self, index, coordinate, piece, rot, img, score):
        """Generates an image for a single 'option' that shows one of the potential matches."""
        x_val = coordinate[0]
        y_val = coordinate[1]
        img_solution_bgr0 = copy.deepcopy(img)
        img_solution_bgr = copy.deepcopy(img)
        img_solution_bgr0, contour_new = move_bgr(self.data.contours_rotated[piece], self.data.img_processed_bgr,
                                                  self.data.grid_centers[piece], img_solution_bgr0, coordinate, self.data.img_blank_comp)
        angle = 90 * rot
        if piece < (len(self.data.corners) + len(self.data.edges)):  # it's a border
            angle = -angle  # weird shit
        else:
            angle = -angle - 90  # even weirder shit
        img_solution_bgr, contour_new = rotate_bgr(contour_new, img_solution_bgr0, coordinate, img_solution_bgr, angle,
                                                   self.data.img_blank_comp)
        cropped = zoom(img_solution_bgr, [x_val, y_val], self.data.av_length)
        h, w, ch = cropped.shape
        cropped[0:2, :] = (0, 0, 200)
        cropped[h-2:h, :] = (0, 0, 200)
        cropped[:, 0:2] = (0, 0, 200)
        cropped[:, w-2:w] = (0, 0, 200)
        cv2.circle(img=cropped, center=tuple([int(0.85*w), int(0.85*h)]),
                   radius=int(0.1*h), color=[255, 255, 255], thickness=-1)
        cv2.putText(cropped, str(index + 1), (int(0.795*w), int(0.905*h)), self.settings.font1, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(cropped, "{:<10.5f}".format(score), (int(0.1*w), h-5), self.settings.font1, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return cropped
