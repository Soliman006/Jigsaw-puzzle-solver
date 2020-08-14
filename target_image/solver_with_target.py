# Main file for solving when the target/solution image is provided.
import cv2
import numpy as np
import copy

from solving.comparator import closestDist
from target_image.cropping import cropping
from graphics import imshow
from utils import imageResize


class SolverWithTarget:
    """Puzzle solver where the target image is provided."""
    MAX_FEATURES = 10000
    GOOD_MATCH_PERCENT = 0.5

    def __init__(self, data, img_target, settings):
        self.data = data
        self.img_target = img_target
        self.settings = settings
        self.reset()

    def reset(self):
        self.value = 0

    def solve(self):
        cropped_pieces, cropped_contours, cropped_masks, cropped_masks_scaled = cropping(self.data)
        # Read reference image
        im2 = copy.deepcopy(self.img_target)
        puzzle_loc = []
        for piece in range(len(cropped_pieces)):
            # Read image to be aligned
            im1 = cropped_pieces[piece]  # imageResize(piece, height=settings.disp_height)

            # Convert images to grayscale
            im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            # Detect ORB features and compute descriptors.
            orb = cv2.ORB_create(self.MAX_FEATURES)
            keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, cropped_masks_scaled[piece])
            keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

            # Match features.
            matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
            matches = matcher.match(descriptors1, descriptors2, None)

            # Sort matches by score
            matches.sort(key=lambda x: x.distance, reverse=False)

            # Remove not so good matches
            numGoodMatches = int(len(matches) * self.GOOD_MATCH_PERCENT)
            matches = matches[:numGoodMatches]

            # Draw top matches
            imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
            imshow(imageResize(imMatches, height=self.settings.disp_height))

            # Extract location of good matches
            points2 = np.zeros((len(matches), 2), dtype=np.float32)

            for i, match in enumerate(matches):
                points2[i, :] = keypoints2[match.trainIdx].pt

            grid_count = np.zeros(len(self.data.grid_centers))
            # target = copy.deepcopy(img_target)
            for point in points2:
                # cv2.circle(img=target,center=tuple(point),radius=settings.point_radius,color=[0,0,255],thickness=-1)
                grid_index, grid_point, min_e_dist, min_x_dist, min_y_dist = closestDist(point, self.data.grid_centers)
                grid_count[grid_index] = grid_count[grid_index] + 1
            loc_index = np.argmax(grid_count)
            puzzle_loc.append(loc_index)
            print(loc_index)
