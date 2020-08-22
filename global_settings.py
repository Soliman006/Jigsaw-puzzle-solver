# This module holds all the hard-coded values that the user may want to tweak:
from utils import hsv_to_cvhsv
import cv2


class globalSettings:
    """Holds all the user-editable settings for the program."""
    env = 'JUPYTER'
    # graphics
    compute_height = 1000  # image height in pixels for computations
    disp_height = 400  # image height in pixels for displaying on screen
    line_thickness = 2  # thickness of lines drawn on images
    point_radius = 6  # radius of circles drawn on images
    font1 = cv2.FONT_HERSHEY_SIMPLEX

    show_extraction_headings = True
    show_extraction_text = True
    show_basic_extraction_graphics = True
    show_full_extraction_graphics = True
    show_extracted_colours = True
    show_colour_extraction_progress = True

    show_incremental_solution = True
    show_colour_comparison = True
    show_comparison_text = True
    show_selection_text = True
    show_current_space_text = True
    show_solver_progress_text = True
    show_error_text = True
    show_backtracker = False

    # "green screen" thresholding
    lower_blue = hsv_to_cvhsv(180, 50, 45)
    upper_blue = hsv_to_cvhsv(220, 100, 100)

    # epsilon values for thresholding
    e_contour_smoothing = 0
    approx_hull_contours_epsilon = 5
    convexity_epsilon = 2500

    # pixel grouping for creating colour contours
    inc = 6

    # score weighting
    score_shape_scalar = 0.052
    score_mult_colour = 5
    score_mult_shape = 1
    score_thresh = 1.5

    helper = False
    helper_threshold = 1.5
    max_options = 6
    select_border = 0

    n_legs = 4
