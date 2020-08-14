""" Example of Puzzle Solver Implementation """

import cv2
from extraction.extractor import ExtractedData
from extraction.contour_finder import approxContours
from solving.solver import SolverData
from extraction.image_import import retrieveExample
from graphics import createSpacedSolution, displaySpacedSolution, createSolution, displaySolution, createBGRSolution, displayBGRSolution
from graphics import createGIFSequential, createGIFTransformation
from utils import hsv_to_cvhsv
from global_settings import globalSettings
settings = globalSettings()

""" Alter Settings as Desired """
# for a full list of settings see global_settings.py
settings.show_extraction_headings = True
settings.show_extraction_text = False
settings.show_basic_extraction_graphics = True
settings.show_full_extraction_graphics = True
settings.show_extracted_colours = False
settings.show_colour_extraction_progress = False

settings.show_incremental_solution = True
settings.show_colour_comparison = False
settings.show_comparison_text = False
settings.show_selection_text = True
settings.show_current_space_text = False
settings.show_solver_progress_text = True
settings.show_error_text = True

settings.helper = True
settings.helper_threshold = 1.5
settings.score_mult_colour = 5
settings.e_contour_smoothing = 1

""" dataset 2 """
# settings.lower_blue = hsv_to_cvhsv(0, 0, 0)
# settings.upper_blue = hsv_to_cvhsv(360, 20, 20)

""" dataset 3 """
# settings.lower_blue = hsv_to_cvhsv(0, 0, 85)
# settings.upper_blue = hsv_to_cvhsv(360, 100, 100)
# settings.compute_height = 993

""" Run Image Capture """

# img_orig = cv2.imread('PATH', cv2.IMREAD_UNCHANGED)
img_orig = retrieveExample('1_unsolved.jpg')
img_target = retrieveExample('1_target.jpg')


""" Run Data Extraction """
# initialise data extraction object with the input image and settings:
data = ExtractedData(img_orig, settings)

# execute the extraction:
data.extract()

# contour smoothing
data.processed_pieces = approxContours(data.processed_pieces, settings.e_contour_smoothing, state=False)

# optional inspection of all the pieces up close:
data.visualPiecesInspection()

""" Solve Puzzle """

puzzle = SolverData(data, settings)
# puzzle.manualCompare(piece1, side1, rotation1, piece2, side2, rotation2)
# puzzle.manualPlace([x,y],piece,rotation)
puzzle.reset()
puzzle.solve()

""" Display Solution """

displaySpacedSolution(createSpacedSolution(data, puzzle), data.radius_max, puzzle.x_limit, puzzle.y_limit, settings)
displaySolution(createSolution(data, puzzle), data.av_length, puzzle.x_limit, puzzle.y_limit, settings)
puzzle.solution_bgr, solution_contours = createBGRSolution(data, puzzle)
displayBGRSolution(puzzle.solution_bgr, data.av_length, puzzle.x_limit, puzzle.y_limit, settings)

""" GIF """
# GIF is located in the root directory of the repository
createGIFSequential(data, puzzle)
createGIFTransformation(data, puzzle)
