# This module holds all the hard-coded values that the user may want to tweak:
import cv2
import numpy as np


class globalSettings:
    """Holds all the user-editable settings for the program."""

    def __init__(self):
        self.env = 'JUPYTER'
        # --------------------
        # Graphics Settings
        # --------------------

        self.compute_height = 1000  # image height in pixels for computations
        self.disp_height = 400  # image height in pixels for displaying on screen
        self.line_thickness = 2  # thickness of lines drawn on images
        self.point_radius = 6  # radius of circles drawn on images
        self.font1 = cv2.FONT_HERSHEY_SIMPLEX

        # --------------------
        # Display Settings
        # --------------------

        # extraction display settings
        self.show_extraction_headings = True
        self.show_extraction_text = True
        self.show_basic_extraction_graphics = True
        self.show_full_extraction_graphics = True
        self.show_extracted_colours = True
        self.show_colour_extraction_progress = True

        # solving display settings
        self.show_incremental_solution = True
        self.show_leg_BGR = True
        self.show_colour_comparison = True
        self.show_comparison_text = True
        self.show_selection_text = True
        self.show_current_space_text = True
        self.show_solver_progress_text = True
        self.show_error_text = True
        self.show_backtracker = False
        self.show_final_bgr = True

        # --------------------
        # Extraction Settings
        # --------------------

        # "green screen" thresholding
        # default background is pure black
        self.bg_thresh_low = self.hsv_to_cvhsv(0, 0, 0)
        self.bg_thresh_high = self.hsv_to_cvhsv(360, 1, 1)

        # epsilon values for thresholding
        self.e_contour_smoothing = 0
        self.approx_hull_contours_epsilon = 5
        self.convexity_epsilon = 2500

        # pixel grouping for creating colour contours
        self.inc = 6

        # --------------------
        # Solver Settings
        # --------------------

        # score weighting
        self.score_shape_scalar = 0.052
        self.score_colour_scalar = 441.673
        self.score_mult_colour = 5
        self.score_mult_shape = 1
        self.score_thresh = 1.5
        self.max_lock_peak_dist = 10
        self.score_colour_thresh = 0.8
        self.score_shape_frac = 0.8

        # helper
        self.helper = False
        self.helper_threshold = 1.5
        self.max_options = 6
        self.select_border = 0

        self.n_legs = 4
        self.start_short = True

        # contour interpolation
        self.interpolation_e = 1
        self.interpolate_ref = True
        self.interpolate_cand = True

    def hsv_to_cvhsv(self, h, s, v):
        """Converts typical HSV ranges of (0<H<360,0<S<100,0<V<100) to the ranges opencv expects of (0<H<179,0<S<255,0<V<255)"""
        cv_h = int(179 * h / 360)
        cv_s = int(255 * s / 100)
        cv_v = int(255 * v / 100)
        colour = np.array([cv_h, cv_s, cv_v])
        return colour
