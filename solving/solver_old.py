import cv2
import numpy as np
import copy

from graphics import createSolution, displaySolution, imshow, createBGRSolution, displayBGRSolution
from solving.comparator import normaliseContours, compareContours, colourClosestDist
from solving.utils import loc_type_detail_to_rotation, setProcessedLists
from solving.updator import incrementPriorities, updateCurves
from utils import imageResize, zoom, takeFourth
from solving.assistant import Assistant


class SolverData:
    """Puzzle solver where no target image is provided."""

    def __init__(self, data, settings):
        """Initialises the puzzle solver with the extracted information."""
        self.data = data
        self.settings = settings
        self.x_limit = self.data.puzzle_columns
        self.y_limit = self.data.puzzle_rows
        self.num_pieces = len(self.data.piece_contours)
        self.n_exterior_pieces = len(self.data.corners) + len(self.data.edges)
        # Location type
        self.loc_type = np.full((self.y_limit, self.x_limit), 0)
        self.loc_type[0][0] = 2
        self.loc_type[0][np.size(self.loc_type, 1) - 1] = 2
        self.loc_type[np.size(self.loc_type, 0) - 1][0] = 2
        self.loc_type[np.size(self.loc_type, 0) - 1][np.size(self.loc_type, 1) - 1] = 2
        for x in range(1, np.size(self.loc_type, 1) - 1):
            self.loc_type[0][x] = 1
        for x in range(1, np.size(self.loc_type, 1) - 1):
            self.loc_type[np.size(self.loc_type, 0) - 1][x] = 1
        for y in range(1, np.size(self.loc_type, 0) - 1):
            self.loc_type[y][0] = 1
        for y in range(1, np.size(self.loc_type, 0) - 1):
            self.loc_type[y][np.size(self.loc_type, 1) - 1] = 1
        # Detailed location type
        self.loc_type_detail = np.full((self.y_limit, self.x_limit), 0)
        self.loc_type_detail[0][0] = 1
        self.loc_type_detail[0][np.size(self.loc_type_detail, 1) - 1] = 3
        self.loc_type_detail[np.size(self.loc_type_detail, 0) - 1][0] = 6
        self.loc_type_detail[np.size(self.loc_type_detail, 0) - 1][np.size(self.loc_type_detail, 1) - 1] = 8
        for x in range(1, np.size(self.loc_type_detail, 1) - 1):
            self.loc_type_detail[0][x] = 2
        for x in range(1, np.size(self.loc_type_detail, 1) - 1):
            self.loc_type_detail[np.size(self.loc_type_detail, 0) - 1][x] = 7
        for y in range(1, np.size(self.loc_type_detail, 0) - 1):
            self.loc_type_detail[y][0] = 4
        for y in range(1, np.size(self.loc_type_detail, 0) - 1):
            self.loc_type_detail[y][np.size(self.loc_type_detail, 1) - 1] = 5
        self.compare_points = [round(self.n_exterior_pieces/2), self.n_exterior_pieces, round((self.num_pieces-self.n_exterior_pieces)/2)]
        self.compare_step = 0
        self.hardReset()

    def hardReset(self):
        self.flags = Flags()
        self.trials_border = []
        self.trials_interior = []
        self.trials = []
        self.border_choice = 0
        self.reset()

    def reset(self):
        """Clears any existing steps taken in attempting to solve the puzzle."""
        self.memory = []
        # Create an array to store the piece number of a piece that is successfully placed in that cell:
        self.loc = np.full((self.y_limit, self.x_limit), -1)
        # Array to store rotation for solution piece
        self.rotation = np.full((self.y_limit, self.x_limit), -1)
        # Create an array to store the variable priority of a space.
        # The solver will use this priority to determine what space to solve next,
        # and update the values of surrounding pieces when a piece is successfully placed:
        self.priority = np.full((self.y_limit, self.x_limit), 1)
        self.priority[self.y_limit - 1][:] = 4
        for y in range(self.y_limit):
            self.priority[y][0] = 5
            self.priority[y][self.x_limit - 1] = 6
        self.priority[0][:] = 6
        # Create an array that can be used to flag whether a piece has been placed:
        self.placed_pieces = np.full((1, self.num_pieces), 0)
        self.placed_pieces = self.placed_pieces[0]
        # Keep track on the order in which pieces are placed
        self.placement_order_piece = np.full((1, self.num_pieces), -1)
        self.placement_order_piece = self.placement_order_piece[0]
        # Array for storing the coordinates of the center of a piece when it is placed in the puzzle:
        self.centers_solution = np.full((self.num_pieces, 2), 0)
        self.centers_solution[1] = [2 * self.data.av_length, 2 * self.data.av_length]
        # To store the contours of already placed pieces to compare new pieces against:
        self.space = np.full((self.y_limit, self.x_limit, 4, 2), -1)
        # Reset the lists of processed unplaced pieces:
        self.processed_corners, self.processed_edges, self.processed_interior = setProcessedLists(
            self.data.corners, self.data.edges, self.data.interior)
        self.placement_order_space = np.full((self.num_pieces, 2), -1)
        self.placement_num = 0
        # reset flags
        self.flags.solvable = 1
        self.flags.backtrack = 0

    def solve2(self):
        """Main function for initiating the iterative solving process with backtracking."""
        while self.placed_pieces.min() == 0:  # so long as there are piece to place
            self.solvePath()  # try to solve path
            # then solvePath will terminate either because:
            # the puzzle is solved, unsolvable, done all current compares, hit a compare count, or to backtrack.
            # if solved:
            if self.placed_pieces.min() != 0:
                print("Puzzle Solved!")
                self.solution_bgr, solution_contours = createBGRSolution(self.data, self)
                displayBGRSolution(self.solution_bgr, self.data.av_length, self.x_limit, self.y_limit, self.settings)
                break
            # if unsolvable:
            if self.flags.solvable == 0 and len(self.trials_border) == 0:
                print("Puzzle Not Solvable!")
                break
            # if generated a compare branch
            if len(self.memory) == self.compare_points[self.compare_step]:
                self.trials.append(self.memory)
            # if generated all current compares

            # if tried a compare branch
            # if tried all compare branches
            # if normal backtrack
            self.backtracker(final_step, final_option)

            final_step, final_option = self.backtrace(self.memory)  # where should we backtrack to?
            if self.flags.solvable == 0 and self.flags.all_borders_tried == 1:  # if unsolvable:
                print("Puzzle Not Solvable!")
                break
            if self.flags.solvable == 0 and self.flags.all_borders_tried == 0:
                print("Trying next best border")
                self.flag.solvable = 1
                self.memory = self.trials_border[self.border_choice]
                final_step = len(self.memory) - 1
                final_option = self.memory[final_step].choice

            if (len(self.processed_corners) + len(self.processed_edges)) != 0:  # border isn't complete
                final_step, final_option = self.backtrace(self.memory)  # where should we backtrack to?
                if self.flags.solvable == 0 and len(self.trials_border) == 0:  # if unsolvable:
                    print("Puzzle Not Solvable!")
                    break
                if self.flags.solvable == 0 and len(self.trials_border) != 0:  # if all borders created:
                    print("All possible borders created!")
                    self.rankBorders()  # rank how good each attempt at the border is
                    if self.settings.select_border:
                        self.border_choice = self.borderSelect()
                    self.memory = self.trials_border[self.border_choice]  # write the next best one to memory
                    final_step = len(self.memory) - 1
                    final_option = self.memory[final_step].choice
                    self.backtracker(final_step, final_option)
                    self.flags.solve_interior = 1
                    self.flags.solvable = 1
                if self.flags.solvable:
                    self.backtracker(final_step, final_option)
            else:  # the border is complete
                self.trials_border.append(self.memory)  # save trial
                final_step, final_option = self.backtrace(self.memory)  # where should we backtrack to?
                if self.flags.solvable == 0:  # all border combos have been created
                    print("All possible borders created!")
                    self.rankBorders()  # rank how good each attempt at the border is
                    if self.settings.select_border:
                        self.border_choice = self.borderSelect()
                    self.memory = self.trials_border[self.border_choice]  # write the next best one to memory
                    final_step = len(self.memory) - 1
                    final_option = self.memory[final_step].choice
                    self.backtracker(final_step, final_option)
                    self.flags.solve_interior = 1
                    self.flags.solvable = 1
                else:
                    self.backtracker(final_step, final_option)  # trigger backtracker then loop back

    def solve(self):
        """Main function for initiating the iterative solving process with backtracking."""
        while self.placed_pieces.min() == 0:
            if self.flags.solve_interior:  # if we are solving the interior
                self.solvePath()  # try to solve path
                # then solvePath will terminate either because the puzzle is solved, unsolvable, or to backtrack.
                if self.placed_pieces.min() != 0:  # if solved:
                    print("Puzzle Solved!")
                    self.solution_bgr, solution_contours = createBGRSolution(self.data, self)
                    displayBGRSolution(self.solution_bgr, self.data.av_length, self.x_limit, self.y_limit, self.settings)
                    break
                else:  # not solved
                    final_step, final_option = self.backtrace(self.memory)  # where should we backtrack to?
                    if self.flags.solvable == 0 and self.flags.all_borders_tried == 1:  # if unsolvable:
                        print("Puzzle Not Solvable!")
                        break
                    if self.flags.solvable == 0 and self.flags.all_borders_tried == 0:
                        print("Trying next best border")
                        self.flag.solvable = 1
                        self.memory = self.trials_border[self.border_choice]
                        final_step = len(self.memory) - 1
                        final_option = self.memory[final_step].choice
                    self.backtracker(final_step, final_option)  # trigger backtracker then loop back
            else:
                self.solvePath()
                if (len(self.processed_corners) + len(self.processed_edges)) != 0:  # border isn't complete
                    final_step, final_option = self.backtrace(self.memory)  # where should we backtrack to?
                    if self.flags.solvable == 0 and len(self.trials_border) == 0:  # if unsolvable:
                        print("Puzzle Not Solvable!")
                        break
                    if self.flags.solvable == 0 and len(self.trials_border) != 0:  # if all borders created:
                        print("All possible borders created!")
                        self.rankBorders()  # rank how good each attempt at the border is
                        if self.settings.select_border:
                            self.border_choice = self.borderSelect()
                        self.memory = self.trials_border[self.border_choice]  # write the next best one to memory
                        final_step = len(self.memory) - 1
                        final_option = self.memory[final_step].choice
                        self.backtracker(final_step, final_option)
                        self.flags.solve_interior = 1
                        self.flags.solvable = 1
                    if self.flags.solvable:
                        self.backtracker(final_step, final_option)
                else:  # the border is complete
                    self.trials_border.append(self.memory)  # save trial
                    final_step, final_option = self.backtrace(self.memory)  # where should we backtrack to?
                    if self.flags.solvable == 0:  # all border combos have been created
                        print("All possible borders created!")
                        self.rankBorders()  # rank how good each attempt at the border is
                        if self.settings.select_border:
                            self.border_choice = self.borderSelect()
                        self.memory = self.trials_border[self.border_choice]  # write the next best one to memory
                        final_step = len(self.memory) - 1
                        final_option = self.memory[final_step].choice
                        self.backtracker(final_step, final_option)
                        self.flags.solve_interior = 1
                        self.flags.solvable = 1
                    else:
                        self.backtracker(final_step, final_option)  # trigger backtracker then loop back

    def solve3(self):
        """Main function for initiating the iterative solving process with backtracking."""
        self.solveJourney()
        if self.flags.unsolvable:
            print("Puzzle Cannot Be Solved!")
        if self.flags.solved:
            print("Puzzle Solved")

    def solveJourney(self):
        """function for finding the most optimal Journey in which to complete the puzzle."""
        print("solveJourney")
        while self.placed_pieces.min() == 0:
            self.solveLeg()
            if self.flags.unsolvable:
                break
            if self.flags.solved:
                # still save current path
                break
            if self.flags.legs_exhausted:

                break
            if self.flags.leg_complete:
                self.legs.append(self.memory)  # save trial
            final_step, final_option = self.backtrace(self.memory)
            self.backtracker(final_step, final_option)  # trigger backtracker then loop back

    def solveLeg(self, leg_start, leg_end):
        """function for finding the most optimal path in which to complete a leg."""
        print("solveLeg")
        while self.placed_pieces.min() == 0:
            paths = []
            self.solvePath(leg_end)
            if self.flags.unsolvable:
                break
            if self.flags.solved:
                # still save current path
                break
            if self.flags.paths_exhausted:
                paths = self.rankPaths(paths)
                break
            final_step, final_option = self.backtrace(self.memory)
            if self.flags.path_complete:
                paths.append(self.memory)
                self.backtracker(final_step, final_option)
                self.flags.path_complete = False
            if self.flags.backtrack:
                self.backtracker(final_step, final_option)
                self.flags.backtrack = False

    def solvePath(self, path_end):
        """function for forward solving a pathway."""
        while self.placed_pieces.min() == 0:
            if len(self.memory) == path_end:
                self.flags.path_complete = True
                break
            space = self.nextSpace()
            x = space[0]
            y = space[1]
            if self.settings.show_current_space_text:
                print(" ")
                print("Now solving for space", space)
            if self.loc_type[y][x] == 2:  # corner
                if self.loc_type_detail[y][x] == 1:  # starting piece
                    corner_index = 0
                    piece = self.processed_corners[corner_index]
                    rotation = loc_type_detail_to_rotation(self.loc_type_detail[y][x])
                    score = 0
                    option = Option(piece, rotation, score)
                    options = []
                    options.append(option)
                    choice = 0
                    step = Step(space, options, choice)
                else:
                    step = self.solveStep(space, self.processed_corners)
            if self.loc_type[y][x] == 1:  # edge
                step = self.solveStep(space, self.processed_edges)
            if self.loc_type[y][x] == 0:  # interior
                step = self.solveStep(space, self.processed_interior)
            if self.flags.backtrack:
                break
            # Place in puzzle and update
            self.place(step)
            if self.settings.show_solver_progress_text:
                print("Progress:", self.placement_num, "/", self.num_pieces)
            if self.settings.show_incremental_solution:
                displaySolution(createSolution(self.data, self), self.data.av_length, self.x_limit, self.y_limit, self.settings)

    def solveStep(self, space, pieces):
        """When provided with a location in the puzzle and list of pieces, it will find the best piece to put in the space."""
        optimal_index, optimal_piece, optimal_rotation, optimal_piece_score,\
            n_sides_compared, matches = self.generate_options(space, pieces)
        if (optimal_piece_score > 99999):
            if self.settings.show_error_text:
                print("No legal matches!")
                print("Beginning Backtracking Protocol")
            self.flags.backtrack = 1
            options = []
            choice = 0
            step = Step(space, options, choice)
            return step
        else:
            if (optimal_piece_score > self.settings.score_thresh*n_sides_compared):
                if self.settings.show_error_text:
                    print("No good matches!")
                    print("Beginning Backtracking Protocol")
                self.flags.backtrack = 1
                options = []
                choice = 0
                step = Step(space, options, choice)
                return step
        # filter to keep only the matches with somewhat decent scores:
        good_matches = self.matchFilter(matches)
        good_matches = self.truncate(good_matches, self.settings.max_options)
        # restructure the data into a list of objects
        options = []
        for index in range(len(good_matches)):
            piece = good_matches[index][1]
            rotation = good_matches[index][2]
            score = good_matches[index][3]
            option = Option(piece, rotation, score)
            options.append(option)
        # save all the good options for this step
        choice = 0
        step = Step(space, options, choice)

        if self.settings.helper:
            # if there is only one decent match then it is correct:
            if len(good_matches) == 1:
                return step
            else:
                # create image of current partial solution
                img_partial_solve, solution_contours = createBGRSolution(self.data, self)
                helper = Assistant(space, good_matches, self.data, img_partial_solve, self.x_limit, self.y_limit, self.settings)
                result = helper.run()
                if result > 0:
                    choice = result - 1
                    step.choice = choice
                return step
        else:
            return step

    def generate_options(self, space, pieces):
        space_x = space[0]
        space_y = space[1]
        max_score = 100000000
        optimal_piece_score = max_score
        optimal_rotation = -1
        optimal_index = -1
        optimal_piece = -1
        n_sides_compared = 0
        matches = []

        for i in range(len(pieces)):
            piece = pieces[i]
            piece_score = max_score
            best_rotation = -1
            for rotation in range(0, 4):
                rotation_score_total = 0
                rotation_score_shape = 0
                rotation_score_colour = 0
                if self.validPieceComparison(space, piece, rotation) == 1:
                    n_sides_compared = 0
                    for side in range(0, 4):
                        space_piece_ref = self.space[space_y][space_x][side][0]
                        space_side_ref = self.space[space_y][space_x][side][1]
                        if space_piece_ref != -1:  # make sure there is a contour to compare to
                            contour1 = self.data.processed_pieces[piece][(side + rotation) % 4]
                            contour2 = self.data.processed_pieces[space_piece_ref][space_side_ref]
                            contour1, contour2, peak_point1, peak_point2 = normaliseContours(contour1, contour2, self.data.av_length)
                            colour_curve1 = self.data.colour_contours[piece][(side + rotation) % 4]
                            colour_curve2 = self.data.colour_contours[space_piece_ref][space_side_ref]
                            colour_contour_xy1 = self.data.colour_contours_xy[piece][(side + rotation) % 4]
                            colour_contour_xy2 = self.data.colour_contours_xy[space_piece_ref][space_side_ref]
                            colour_contour_xy1, colour_contour_xy2, colour_peak_point1, colour_peak_point2 = normaliseContours(
                                colour_contour_xy1, colour_contour_xy2, self.data.av_length)
                            side_score_shape, side_score_colour, side_score_total\
                                = compareContours(contour1, contour2, colour_curve1, colour_curve2, colour_contour_xy1,
                                                  colour_contour_xy2, self.data.av_length, self.settings)
                            rotation_score_shape = rotation_score_shape + side_score_shape
                            rotation_score_colour = rotation_score_colour + side_score_colour
                            rotation_score_total = rotation_score_total + side_score_total
                            n_sides_compared = n_sides_compared + 1
                    if self.settings.show_comparison_text:
                        print("Comparing piece", piece, "with rotation", rotation, "to space", space,
                              f'scores: shape {rotation_score_shape:.4f} colour {rotation_score_colour:.4f}'
                              f' total {rotation_score_total:.4f}')
                else:
                    rotation_score_total = 10000000

                if rotation_score_total < piece_score:
                    piece_score = rotation_score_total
                    best_rotation = rotation
            score_log = [i, piece, best_rotation, piece_score]
            matches.append(score_log)
            if piece_score < optimal_piece_score:
                optimal_piece_score = piece_score
                optimal_rotation = best_rotation
                optimal_index = i
                optimal_piece = piece
        return optimal_index, optimal_piece, optimal_rotation, optimal_piece_score, n_sides_compared, matches

    def place(self, step):  # space, piece, correct_rotation):
        """Command allowing the user to manually force a certain piece into a certain place in the puzzle."""
        self.memory.append(step)
        choice = step.choice
        piece = step.options[choice].piece
        rotation = 90 * step.options[choice].rotation
        space = step.space
        x = space[0]
        y = space[1]
        # delete placed piece from list of available pieces:
        if self.loc_type[y][x] == 2:
            # corner
            for index in range(len(self.processed_corners)):
                if self.processed_corners[index] == piece:
                    cat_index = index
            del self.processed_corners[cat_index]
        if self.loc_type[y][x] == 1:
            # edge
            for index in range(len(self.processed_edges)):
                if self.processed_edges[index] == piece:
                    cat_index = index
            del self.processed_edges[cat_index]
        if self.loc_type[y][x] == 0:
            # interior
            for index in range(len(self.processed_interior)):
                if self.processed_interior[index] == piece:
                    cat_index = index
            del self.processed_interior[cat_index]
        # update the puzzle:
        self.updatePuzzle(space, piece, rotation)
        self.placement_num = self.placement_num + 1
        if self.settings.show_selection_text:
            print("Piece", piece, "with rotation", rotation, "has been placed into space", space)

    def nextSpace(self):
        """Determines which space in the puzzle to attemp to solve next, based on the priority array."""
        for i in range(-7, -1):
            level = -i
            for y in range(0, self.y_limit):
                for x in range(0, self.x_limit):
                    if self.priority[y][x] == level:
                        return [x, y]

    def validPieceComparison(self, space, piece, rotation):
        """Checks if a space and a piece could possibly be a match based on basic criteria."""
        space_x = space[0]
        space_y = space[1]
        # for border pieces, check that the straight edge is on the outside
        border_rot = loc_type_detail_to_rotation(self.loc_type_detail[space_y][space_x])
        if ((rotation != border_rot) and (border_rot != -1)):
            # print("border_rot",border_rot,"rotation",rotation)
            # print("Bad Border")
            return 0

        # matching locks
        for side in range(0, 4):
            if self.space[space_y][space_x][side][0] != -1:  # check there is something to compare to
                piece1 = piece
                side1 = (side + rotation) % 4
                piece2 = self.space[space_y][space_x][side][0]
                side2 = self.space[space_y][space_x][side][1]
                edge_type1 = self.data.processed_edge_types[piece1][side1]
                edge_type2 = self.data.processed_edge_types[piece2][side2]
                if edge_type1 == 0:
                    if self.settings.show_current_space_text:
                        print("The piece has a border!")
                    return 0
                if edge_type2 == 0:
                    if self.settings.show_current_space_text:
                        print("The SPACE has a border!")
                    return 0
                if edge_type1 == edge_type2:
                    # print("Bad Locks")
                    return 0
                # print("Considering comparing space type",edge_type2,"with piece type",edge_type1)
            # else:
                # print("nothing to compare")
        # print("Good Comparison")
        return 1

    def updatePuzzle(self, space, piece, rotation):
        """Main command for updating the solution after a piece has been chosen to be placed in a space."""
        x = space[0]
        y = space[1]
        self.placed_pieces[piece] = 1
        self.loc[y][x] = piece
        self.rotation[y][x] = rotation
        self.priority = incrementPriorities(
            x, y, self.loc_type, self.priority, self.x_limit, self.y_limit)
        self.space = updateCurves(x, y, piece, rotation / 90, self.space,
                                  self.x_limit, self.y_limit)
        self.placement_order_space[self.placement_num] = space
        self.placement_order_piece[self.placement_num] = piece

    def manualCompare(self, piece1, side1, rotation1, piece2, side2, rotation2):
        """Command allowing the user to manually compare how well 2 pieces match."""
        img_processed_bgr = copy.deepcopy(self.data.img_processed_bgr)
        contour1 = self.data.processed_pieces[piece1][(side1 + rotation1) % 4]
        contour2 = self.data.processed_pieces[piece2][(side2 + rotation2) % 4]
        contour1, contour2, peak_point1, peak_point2 = normaliseContours(contour1, contour2, self.data.av_length)
        colour_contour_xy1 = self.data.colour_contours_xy[piece1][(side1 + rotation1) % 4]
        colour_contour_xy2 = self.data.colour_contours_xy[piece2][(side2 + rotation2) % 4]
        colour_contour_xy1, colour_contour_xy2, colour_peak_point1, colour_peak_point2 = normaliseContours(
            colour_contour_xy1, colour_contour_xy2, self.data.av_length)
        colour_curve1 = self.data.colour_contours[piece1][(side1 + rotation1) % 4]
        colour_curve2 = self.data.colour_contours[piece2][(side2 + rotation2) % 4]
        score_shape, score_colour, score_total = compareContours(
            contour1, contour2, colour_curve1, colour_curve2, colour_contour_xy1, colour_contour_xy2, self.data.av_length, self.settings)

        print("Comparing piece", piece1, "(side", side1, "rot", rotation1, ") with piece", piece2, "(side", side2,
              "rot", rotation2, "). ", f'Scores: shape {score_shape:.4f} colour {score_colour:.4f} total {score_total:.4f}')
        # bgr overlays
        cv2.polylines(img=img_processed_bgr, pts=[self.data.processed_pieces[piece1][(
            side1 + rotation1) % 4]], isClosed=0, color=(0, 0, 255), thickness=1)
        cv2.polylines(img=img_processed_bgr, pts=[self.data.processed_pieces[piece2][(
            side2 + rotation2) % 4]], isClosed=0, color=(0, 255, 0), thickness=1)

        cropped1 = zoom(img_processed_bgr, self.data.grid_centers[piece1], self.data.radius_max)
        h, w, ch = cropped1.shape
        cropped = np.zeros([h, 3 * w, 3], dtype=np.uint8)
        cropped[:, 0:w] = cropped1
        cropped2 = zoom(img_processed_bgr, self.data.grid_centers[piece2], self.data.radius_max)
        cropped[:, w:2 * w] = cropped2

        # overlapped contours
        img_norm_segments = np.zeros([h, w, 3], dtype=np.uint8)
        offset = [75, 50]
        cv2.polylines(img=img_norm_segments, pts=[contour2 - offset],
                      isClosed=0, color=(0, 255, 0), thickness=1)
        cv2.polylines(img=img_norm_segments, pts=[contour1 - offset],
                      isClosed=0, color=(0, 0, 255), thickness=1)
        cv2.circle(img=img_norm_segments, center=tuple(peak_point1 - offset),
                   radius=1, color=[0, 255, 255], thickness=-1)
        cv2.circle(img=img_norm_segments, center=tuple(peak_point2 - offset),
                   radius=1, color=[255, 0, 255], thickness=-1)
        cropped[:, 2 * w:] = img_norm_segments

        # colour bar
        col_w = int(self.settings.inc - 1)
        width = col_w * len(colour_contour_xy1)
        height = 20
        img_colour = np.zeros([height, width, 3], dtype=np.uint8)
        for index in range(len(colour_contour_xy1)):
            point1 = colour_contour_xy1[index]
            colour1 = colour_curve1[index]
            dist, colour2 = colourClosestDist(
                point1, colour_contour_xy1, colour1, colour_contour_xy2, colour_curve2)
            img_colour[0:10, col_w * index:col_w * index + col_w] = colour1
            img_colour[10:21, col_w * index:col_w * index + col_w] = colour2
        cropped[h - 40:h - 20, 2 * w:2 * w + width] = img_colour
        imshow(imageResize(cropped, height=self.settings.disp_height), self.settings.env)

    def manualPlace(self, space, piece, rotation):
        """Manually forces a piece into the solution."""
        score = 0
        option = Option(piece, rotation, score)
        options = []
        options.append(option)
        choice = 0
        step = Step(space, options, choice)
        self.place(step)

    def matchFilter(self, matches):
        """Filters piece matches, keeps only the good ones and ranking them from best to worst."""
        good_matches = []
        matches.sort(key=takeFourth)
        for k in range(len(matches)):
            if self.settings.helper_threshold*matches[0][3] > matches[k][3]:
                good_matches.append(matches[k])
        return good_matches

    def backtrace(self, memory):
        """Determines where the backtracker should move back to."""
        if self.flags.solve_interior:  # if solving the interior
            for i in range(len(memory)):
                step = len(memory) - 1 - i
                if step <= self.n_exterior_pieces - 1:
                    if self.border_choice != len(self.trials_border) - 1:
                        self.border_choice = self.border_choice + 1
                        return -1, -1
                    else:
                        self.flags.all_borders_tried = 1
                        self.flags.solvable = 0
                        return -1, -1
                else:
                    if memory[step].choice != len(memory[step].options) - 1:
                        step_number = step
                        option_number = memory[step].choice + 1
                        return step_number, option_number
            self.flags.solvable = 0
            return -1, -1
        else:  # if solving the border
            for i in range(len(memory)):
                step = len(memory) - 1 - i
                if memory[step].choice != len(memory[step].options) - 1:
                    step_number = step
                    option_number = memory[step].choice + 1
                    return step_number, option_number
            self.flags.solvable = 0
            return -1, -1

    def backtracker(self, final_step, final_option):
        """Undoes solver placements back to a specified option in a specified step."""
        # TODO don't backtrack into border, instead choose next best border
        memory = copy.deepcopy(self.memory)
        memory[final_step].choice = final_option
        self.reset()
        if self.settings.show_backtracker:
            count = 0
            for step in range(final_step + 1):
                count = count + 1
                choice = memory[step].choice
                print(choice, end=" ")
            if count < 54:
                for i in range(54 - count):
                    print("X", end=" ")
            print("")
        for step in range(final_step + 1):
            choice = memory[step].choice
            self.place(memory[step])
        if self.settings.show_incremental_solution:
            displaySolution(createSolution(self.data, self), self.data.av_length, self.x_limit, self.y_limit, self.settings)

    def truncate(self, list, length):
        if len(list) > length:
            list = list[0:length]
        return list

    def rankPaths(self, paths):
        """Ranks all the border attempts based on average match score."""
        ranked_paths = []
        for index in range(len(paths)):
            path = paths[index]
            score_sum = 0
            score_count = 0
            for step in range(len(path)):
                choice = path[step].choice
                score = path[step].options[choice].score
                score_sum = score_sum + score
                score_count = score_count + 1
            score_av = score_sum / score_count
            entry = [score_av, path]
            ranked_paths.append(entry)
        ranked_paths.sort()
        if self.settings.show_backtracker:
            for entry in range(len(ranked_paths)):
                memory = ranked_paths[entry][1]
                count = 0
                for step in range(len(memory)):
                    count = count + 1
                    choice = memory[step].choice
                    print(choice, end=" ")
                if count < 54:
                    for i in range(54 - count):
                        print("X", end=" ")
                score = ranked_paths[entry][0]
                print(round(score, 6), end=" ")
                print("")
        return ranked_paths

    def borderSelect(self):
        for entry in range(len(self.trials_border)):
            self.memory = self.trials_border[entry]
            final_step = len(self.memory) - 1
            final_option = self.memory[final_step].choice
            self.backtracker(final_step, final_option)
            self.solution_bgr, solution_contours = createBGRSolution(self.data, self)
            displayBGRSolution(self.solution_bgr, self.data.av_length, self.x_limit, self.y_limit, self.settings)
            print("Is this the correct border? (y/n)")
            inputString = input()
            if inputString == 'y' or\
                    inputString == 'yes' or\
                    inputString == 'Y' or\
                    inputString == 'Yes' or\
                    inputString == 'YES':
                return entry


class Step:
    """Stores one attempt to solve a puzzle space, with all the pieces that all possible matches."""

    def __init__(self, space, options, choice=0):
        """Initialisation"""
        self.space = space
        self.options = options
        self.choice = choice


class Option:
    """Stores one match option."""

    def __init__(self, piece, rotation, score):
        """Initialisation"""
        self.piece = piece
        self.rotation = rotation
        self.score = score


class Flags:
    """Stores flags."""

    def __init__(self):
        """Initialisation"""
        self.solvable = 1
        self.backtrack = 0
        self.all_borders_tried = 0
        self.solve_interior = 0
