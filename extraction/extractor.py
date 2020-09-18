"""Extractor"""
import numpy as np
from graphics import imshow
from utils import imageResize, zoom
from extraction.image_import import imgCapture
from extraction.contour_finder import contourFinder, approxContours
from extraction.clearance_radius import clearanceRadius
from extraction.hull_creator import hullCreator
from extraction.corner_finder import cornerFinder
from extraction.convexity import convexity
from extraction.center_finder import centerFinder
from extraction.side_separator import sideSeparator
from extraction.piece_types import pieceType, puzzleSize
from extraction.locks import locksSearcher
from extraction.aligner import aligner
from extraction.processed_data import processedData
from extraction.bgr_data import bgrData
from extraction.colour_identification import colourIdentification


class ExtractedData:
    """Class that handles extracting piece information from an input image."""

    def __init__(self, img, settings):
        """Initialises the extractor with the input image and settings."""
        self.img_orig = img
        self.settings = settings

    def extract(self):
        """Executes the extraction process."""
        # image import
        if self.settings.show_extraction_headings:
            print("importing image")
        img_orig, img_comp, img_disp, img_blank_orig, img_blank_comp, img_blank_disp = imgCapture(self.img_orig, self.settings)
        if self.settings.show_basic_extraction_graphics:
            imshow(imageResize(img_comp, height=self.settings.disp_height), self.settings.env)

        # contour finder
        if self.settings.show_extraction_headings:
            print("finding contours")
        piece_contours, img_mask_bgr, img_mask, img_masked = contourFinder(img_comp, img_blank_comp, self.settings)
        if self.settings.show_full_extraction_graphics:
            imshow(imageResize(img_mask, height=self.settings.disp_height), self.settings.env)
            print("Number of Puzzle Pieces:", len(piece_contours))
            imshow(imageResize(img_masked, height=self.settings.disp_height), self.settings.env)

        # clearance radius
        if self.settings.show_extraction_headings:
            print("detecting clearance radii")
        img_circles, circle_centers, radius_max = clearanceRadius(piece_contours, img_mask_bgr, self.settings)
        if self.settings.show_extraction_text:
            print("Max Clearance Radius:", radius_max)
        if self.settings.show_full_extraction_graphics:
            imshow(imageResize(img_circles, height=self.settings.disp_height), self.settings.env)

        # hull creator
        if self.settings.show_extraction_headings:
            print("creating hulls")
        img_hull_mask_bgr, hull, hull_points = hullCreator(piece_contours, img_mask_bgr, self.settings)
        if self.settings.show_full_extraction_graphics:
            imshow(imageResize(img_hull_mask_bgr, height=self.settings.disp_height), self.settings.env)

        # convexity
        if self.settings.show_extraction_headings:
            print("detecting convexity defects")
        img_defects, defects_f = convexity(piece_contours, hull, img_mask_bgr, self.settings)
        if self.settings.show_full_extraction_graphics:
            imshow(imageResize(img_defects, height=self.settings.disp_height), self.settings.env)

        # corner finder
        if self.settings.show_extraction_headings:
            print("finding corners")
        best_rectangles_sorted, best_rectangles_index, av_length, img_corners = cornerFinder(hull_points, piece_contours,
                                                                                             img_mask_bgr, self.settings)
        if self.settings.show_full_extraction_graphics:
            imshow(imageResize(img_corners, height=self.settings.disp_height), self.settings.env)

        # center finder
        if self.settings.show_extraction_headings:
            print("finding piece centers")
        img_centers, centers = centerFinder(best_rectangles_sorted, img_corners, self.settings)
        if self.settings.show_full_extraction_graphics:
            imshow(imageResize(img_centers, height=self.settings.disp_height), self.settings.env)

        # side separator
        if self.settings.show_extraction_headings:
            print("splitting piece contours into sides")
        all_curves, side_lengths = sideSeparator(best_rectangles_index, piece_contours, img_blank_comp, self.settings)

        # corner & edge pieces
        if self.settings.show_extraction_headings:
            print("determining piece types")
        img_piece_type, piece_type, edge_type, defects_by_side, interior, edges, corners = pieceType(defects_f, best_rectangles_index,
                                                                                                     piece_contours, img_blank_comp)
        if self.settings.show_full_extraction_graphics:
            imshow(imageResize(img_piece_type, height=self.settings.disp_height), self.settings.env)

        puzzle_rows, puzzle_columns, corner_piece_count, edge_piece_count, standard_piece_count = puzzleSize(piece_type)

        h_spaced = 2 * radius_max * (puzzle_rows) + 2 * radius_max
        w_spaced = 2 * radius_max * (puzzle_columns) + 2 * radius_max

        img_blank_spaced = np.zeros([h_spaced, w_spaced, 3], dtype=np.uint8)

        corner_piece_count = len(corners)
        edge_piece_count = len(edges)
        standard_piece_count = len(interior)

        # locks searcher
        if self.settings.show_extraction_headings:
            print("detecting locks")
        img_locks, outer_count, inner_count, edge_count = locksSearcher(defects_by_side, corner_piece_count, edge_piece_count,
                                                                        standard_piece_count, img_mask_bgr, all_curves, self.settings)
        if self.settings.show_extraction_text:
            print("Outer Locks:", outer_count, "Inner Locks:", inner_count, "Edges:", edge_count)
        if self.settings.show_full_extraction_graphics:
            imshow(imageResize(img_locks, height=self.settings.disp_height), self.settings.env)

        # aligner
        if self.settings.show_extraction_headings:
            print("aligning pieces")
        grid_centers, angles, contours_rotated, all_segments_rotated, all_corners_rotated, processed_edge_types, img_align_segments\
            = aligner(radius_max, best_rectangles_sorted, corners, edges, interior, piece_contours, img_blank_spaced, edge_type, centers,
                      all_curves, puzzle_rows, puzzle_columns, self.settings)
        if self.settings.show_full_extraction_graphics:
            imshow(imageResize(img_align_segments, height=self.settings.disp_height), self.settings.env)

        # processed data
        if self.settings.show_extraction_headings:
            print("preparing data")
        processed_pieces, img_processed_segments = processedData(all_segments_rotated, img_blank_spaced, processed_edge_types,
                                                                 all_corners_rotated, self.settings)
        processed_pieces = approxContours(processed_pieces, self.settings.e_contour_smoothing, state=False)
        if self.settings.show_basic_extraction_graphics:
            imshow(imageResize(img_processed_segments, height=self.settings.disp_height), self.settings.env)

        # bgr data
        if self.settings.show_extraction_headings:
            print("performing BGR data manipulation")
        processed_bgr, img_processed_bgr = bgrData(img_blank_spaced, corners, edges, interior, angles, piece_contours, img_masked, centers,
                                                   grid_centers, edge_type)
        if self.settings.show_basic_extraction_graphics:
            imshow(imageResize(img_processed_bgr, height=self.settings.disp_height), self.settings.env)

        # colour identification
        if self.settings.show_extraction_headings:
            print("identifying piece edge colours")
        colour_contours, colour_contours_xy = colourIdentification(processed_pieces, img_processed_bgr, self.settings)

        self.piece_contours = piece_contours
        self.puzzle_rows = puzzle_rows
        self.puzzle_columns = puzzle_columns
        self.av_length = av_length
        self.piece_type = piece_type
        self.processed_edge_types = processed_edge_types
        self.processed_pieces = processed_pieces
        self.colour_contours = colour_contours
        self.colour_contours_xy = colour_contours_xy
        self.img_processed_bgr = img_processed_bgr
        self.grid_centers = grid_centers
        self.radius_max = radius_max
        self.contours_rotated = contours_rotated
        self.corners = corners
        self.edges = edges
        self.interior = interior
        self.img_blank_comp = img_blank_comp
        self.img_masked = img_masked
        self.centers = centers
        self.angles = angles
        self.edge_type = edge_type

        if (self.settings.show_extraction_headings):
            print("EXTRACTION COMPLETE")

    def visualPiecesInspection(self):
        """Displays zoomed in images of every piece extracted."""
        for piece in range(len(self.grid_centers)):
            cropped = zoom(self.img_processed_bgr, self.grid_centers[piece], self.radius_max)
            imshow(imageResize(cropped, height=self.settings.disp_height), self.settings.env)
