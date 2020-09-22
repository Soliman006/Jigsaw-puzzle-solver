import numpy as np
import copy
import cv2

from graphics import createSolution, displaySolution
from solving.utils import loc_type_detail_to_rotation, setProcessedLists, locTypeInit, priorityInit, Option, Step, rankPaths

from graphics import createBGRSolution, displayBGRSolution, imshow
from solving.comparator import normaliseContours, compareContours, colourClosestDist
from solving.updator import incrementPriorities, updateCurves
from solving.assistant import Assistant
from utils import takeFourth, imageResize, zoom


class Flags:
    """Stores flags."""

    def __init__(self):
        """Initialisation"""
        self.backtrack = False
        self.path_complete = False


class Solver:
    """Puzzle solver where no target image is provided."""

    def __init__(self, data, settings):
        """Initialises the puzzle solver with the extracted information."""
        self.data = data
        self.settings = settings
        self.n_pieces = len(self.data.piece_contours)
        self.n_exterior_pieces = len(self.data.corners) + len(self.data.edges)
        if self.data.puzzle_columns < self.data.puzzle_rows:
            self.short = self.data.puzzle_columns
            self.long = self.data.puzzle_rows
        else:
            self.short = self.data.puzzle_rows
            self.long = self.data.puzzle_columns
        self.leg_points = []
        self.leg_points.append(0)
        self.leg_points.append(self.short)
        self.leg_points.append(self.short+self.long-1)
        self.leg_points.append((2*self.short)+self.long-2)
        self.leg_points.append(self.n_exterior_pieces)
        for i in range(1, self.long-3):
            self.leg_points.append(self.n_exterior_pieces + (i*(self.short-2)))
        self.leg_points.append(self.n_pieces)
        self.n_legs = len(self.leg_points)-1
        self.hardReset()

    def hardReset(self):
        self.flags = Flags()
        self.reset()

    def reset(self):
        """Clears any existing steps taken in attempting to solve the puzzle."""
        self.x_limit = self.long
        self.y_limit = self.long
        self.memory = []
        self.loc_type, self.loc_type_detail = locTypeInit(self.x_limit, self.y_limit, self.short)
        # Create an array to store the piece number of a piece that is successfully placed in that cell:
        self.loc = np.full((self.y_limit, self.x_limit), -1)
        # Array to store rotation for solution piece
        self.rotation = np.full((self.y_limit, self.x_limit), -1)
        # Create an array to store the variable priority of a space.
        # The solver will use this priority to determine what space to solve next,
        # and update the values of surrounding pieces when a piece is successfully placed:
        self.priority = priorityInit(self.x_limit, self.y_limit)
        # Create an array that can be used to flag whether a piece has been placed:
        self.placed_pieces = np.full((1, self.n_pieces), 0)
        self.placed_pieces = self.placed_pieces[0]
        # Keep track on the order in which pieces are placed
        self.placement_order_piece = np.full((1, self.n_pieces), -1)
        self.placement_order_piece = self.placement_order_piece[0]
        # Array for storing the coordinates of the center of a piece when it is placed in the puzzle:
        self.centers_solution = np.full((self.n_pieces, 2), 0)
        self.centers_solution[1] = [2 * self.data.av_length, 2 * self.data.av_length]
        # To store the contours of already placed pieces to compare new pieces against:
        self.space = np.full((self.y_limit, self.x_limit, 4, 2), -1)
        # Reset the lists of processed unplaced pieces:
        self.processed_corners, self.processed_edges, self.processed_interior = setProcessedLists(
            self.data.corners, self.data.edges, self.data.interior)
        self.placement_order_space = np.full((self.n_pieces, 2), -1)
        self.placement_num = 0
        # reset flags

    def solve(self):
        """Main function for initiating the iterative solving process with backtracking."""
        self.journey = self.solveJourney()
        if len(self.memory) == self.n_pieces:
            print("Puzzle Solved")
        else:
            print("Puzzle Cannot Be Solved!")

    def solveJourney(self):
        """function for finding the most optimal Journey in which to complete the puzzle."""
        print("solveJourney")
        self.legs = []
        self.path_choice = []
        for i_leg in range(self.n_legs):
            self.path_choice.append(0)
        while True:
            i = len(self.legs)
            leg_start = self.leg_points[i]
            leg_end = self.leg_points[i + 1]
            ranked_paths = self.solveLeg(leg_start, leg_end)
            if self.flags.backtrack:
                self.flags.backtrack = False
                i_leg = self.legBacktrace()
                if i_leg == -1:
                    break
                leg = self.legs[i_leg][self.path_choice[i_leg]][1]
                for j in range(i_leg+1, self.n_legs):
                    self.path_choice[j] = 0
                temp_legs = self.legs
                self.legs = []
                for k in range(i_leg+1):
                    self.legs.append(temp_legs[k])
                print("backtracking to leg", i_leg, "path", self.path_choice[i_leg])
            else:
                self.legs.append(ranked_paths)
                leg = ranked_paths[0][1]
            final_step = len(leg)-1
            final_option = leg[final_step].choice
            self.memory = leg
            self.backtracker(final_step, final_option)
            if len(self.memory) == self.n_pieces:
                while True:
                    print("Puzzle Solved")
                    self.solution_bgr, solution_contours = createBGRSolution(self.data, self)
                    displayBGRSolution(self.solution_bgr, self.data.av_length, self.x_limit, self.y_limit, self.settings)
                    print("Is this the correct solution? (y/n)")
                    # get input from user
                    # inputString = input()
                    inputString = 'y'
                    if inputString == 'y':
                        return self.legs
                    else:
                        if inputString == 'n':
                            self.selection = inputString
                            print('trying again')
                        else:
                            print('Invalid entry, assuming no')
                        i_leg = self.legBacktrace()
                        if i_leg == -1:
                            return self.legs
                        leg = self.legs[i_leg][self.path_choice[i_leg]][1]
                        for j in range(i_leg+1, self.n_legs):
                            self.path_choice[j] = 0
                        temp_legs = self.legs
                        self.legs = []
                        for k in range(i_leg+1):
                            self.legs.append(temp_legs[k])
                        print("backtracking to leg", i_leg, "path", self.path_choice[i_leg])
                        final_step = len(leg)-1
                        final_option = leg[final_step].choice
                        self.memory = leg
                        self.backtracker(final_step, final_option)
                        if i_leg != self.n_legs-1:
                            break
            else:
                if self.settings.show_leg_BGR:
                    self.solution_bgr, solution_contours = createBGRSolution(self.data, self)
                    displayBGRSolution(self.solution_bgr, self.data.av_length, self.x_limit, self.y_limit, self.settings)
        return self.legs

    def solveLeg(self, leg_start, leg_end):
        """function for finding the most optimal path in which to complete a leg."""
        # print("solveLeg")
        self.paths = []
        while True:
            path = self.solvePath(leg_end)
            final_step, final_option = self.backtrace(self.memory)
            if self.flags.path_complete:
                self.paths.append(path)
            if final_step < leg_start:
                # print("Number of paths tried", len(self.paths))
                self.flags.path_complete = False
                if len(self.paths) == 0:
                    self.flags.backtrack = True
                    return -1
                else:
                    ranked_paths = rankPaths(self.paths, self.settings)
                    # leg = ranked_paths[0][1]
                    self.flags.backtrack = False
                    return ranked_paths
            if self.flags.path_complete:
                self.backtracker(final_step, final_option)
                self.flags.path_complete = False
                self.flags.backtrack = False
            if self.flags.backtrack:
                self.backtracker(final_step, final_option)
                self.flags.backtrack = False

    def solvePath(self, path_end):
        """function for forward solving a pathway."""
        while True:
            if len(self.memory) >= path_end:
                self.flags.path_complete = True
                path = self.memory
                return path
            space = self.nextSpace()
            x = space[0]
            y = space[1]
            if self.settings.show_current_space_text:
                print(" ")
                print("Now solving for space", space)
            if self.loc_type[y][x] == 3:  # corner or edge
                processed_exterior = self.processed_corners + self.processed_edges
                step = self.solveStep(space, processed_exterior)
            if self.loc_type[y][x] == 2:  # corner
                if self.loc_type_detail[y][x] == 1:  # starting piece
                    options = []
                    rotation = loc_type_detail_to_rotation(self.loc_type_detail[y][x])
                    for index in range(len(self.processed_corners)):
                        piece = self.processed_corners[index]
                        score = 1 + (0.01*index)
                        option = Option(piece, rotation, score)
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
                path = self.memory
                return path
            # Place in puzzle and update
            self.place(step)
            if self.settings.show_solver_progress_text:
                print("Progress:", self.placement_num, "/", self.n_pieces)
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
                            # reference data
                            contour_ref = self.data.processed_pieces[space_piece_ref][space_side_ref]
                            colour_curve_ref = self.data.colour_contours[space_piece_ref][space_side_ref]
                            colour_contour_xy_ref = self.data.colour_contours_xy[space_piece_ref][space_side_ref]
                            # candidate data
                            contour_cand = self.data.processed_pieces[piece][(side + rotation) % 4]
                            colour_curve_cand = self.data.colour_contours[piece][(side + rotation) % 4]
                            colour_contour_xy_cand = self.data.colour_contours_xy[piece][(side + rotation) % 4]
                            # normalisation
                            contour_ref, contour_cand, peak_point_ref, peak_point_cand\
                                = normaliseContours(contour_ref, contour_cand, self.data.av_length)
                            colour_contour_xy_ref, colour_contour_xy_cand, colour_peak_point_ref, colour_peak_point_cand\
                                = normaliseContours(colour_contour_xy_ref, colour_contour_xy_cand, self.data.av_length)
                            # comparison
                            side_score_shape, side_score_colour, side_score_total\
                                = compareContours(contour_ref, contour_cand, colour_curve_ref, colour_curve_cand, colour_contour_xy_ref,
                                                  colour_contour_xy_cand, self.data.av_length, self.settings)
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

    def validPieceComparison(self, space, piece, rotation):
        """Checks if a space and a piece could possibly be a match based on basic criteria."""
        space_x = space[0]
        space_y = space[1]
        # for border pieces, check that the straight edge is on the outside
        border_rot = loc_type_detail_to_rotation(self.loc_type_detail[space_y][space_x])
        if self.loc_type[space_y][space_x] == 3:
            if piece in self.processed_corners:
                border_rot = 1
            else:
                border_rot = 0
        if ((rotation != border_rot) and (border_rot != -1)):
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
                    return 0
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
            if self.settings.helper_threshold*matches[0][3] >= matches[k][3]:
                good_matches.append(matches[k])
        return good_matches

    def truncate(self, list, length):
        if len(list) > length:
            list = list[0:length]
        return list

    def nextSpace(self):
        """Determines which space in the puzzle to attemp to solve next, based on the priority array."""
        for i in range(-20, -1):
            level = -i
            if self.y_limit >= self.x_limit:  # puzzle is tall
                for y in range(0, self.y_limit):
                    for x in range(0, self.x_limit):
                        if self.priority[y][x] == level:
                            return [x, y]
            else:  # puzzle is wide
                for x in range(0, self.x_limit):
                    for y in range(0, self.y_limit):
                        if self.priority[y][x] == level:
                            return [x, y]

    def place(self, step):
        """Command allowing the user to manually force a certain piece into a certain place in the puzzle."""
        self.memory.append(step)
        choice = step.choice
        piece = step.options[choice].piece
        rotation = 90 * step.options[choice].rotation
        space = step.space
        x = space[0]
        y = space[1]
        # delete placed piece from list of available pieces:
        if self.loc_type[y][x] == 3:  # corner or edge
            if piece in self.processed_corners:
                self.processed_corners.remove(piece)
                self.x_limit = self.short
                self.y_limit = self.long
            else:  # is edge
                self.processed_edges.remove(piece)
                self.x_limit = self.long
                self.y_limit = self.short
            self.loc_type, self.loc_type_detail = locTypeInit(self.x_limit, self.y_limit, self.short)
            self.loc = self.loc[:self.y_limit, :self.x_limit]
            self.rotation = self.rotation[:self.y_limit, :self.x_limit]
            temp_priority = self.priority
            self.priority = priorityInit(self.x_limit, self.y_limit)
            self.priority[0:2, 0:self.short-1] = temp_priority[0:2, 0:self.short-1]
            self.space = self.space[:self.y_limit, :self.x_limit]

        if self.loc_type[y][x] == 2:
            self.processed_corners.remove(piece)
        if self.loc_type[y][x] == 1:
            self.processed_edges.remove(piece)
        if self.loc_type[y][x] == 0:
            self.processed_interior.remove(piece)
        # update the puzzle:
        self.updatePuzzle(space, piece, rotation)
        self.placement_num = self.placement_num + 1
        if self.settings.show_selection_text:
            print("Piece", piece, "with rotation", rotation, "has been placed into space", space)

    def backtrace(self, memory):
        """Determines where the backtracker should move back to."""
        for i in range(len(memory)):
            step = len(memory) - 1 - i
            if memory[step].choice != len(memory[step].options) - 1:
                step_number = step
                option_number = memory[step].choice + 1
                return step_number, option_number
        return -1, -1

    def legBacktrace(self):
        for i in range(len(self.legs)):
            i_leg = len(self.legs) - 1 - i
            if self.path_choice[i_leg] != len(self.legs[i_leg]) - 1:
                self.path_choice[i_leg] = self.path_choice[i_leg] + 1
                return i_leg
        return -1

    def backtracker(self, final_step, final_option):
        """Undoes solver placements back to a specified option in a specified step."""
        # TODO don't backtrack into border, instead choose next best border
        memory = copy.deepcopy(self.memory)
        memory[final_step].choice = final_option
        self.reset()
        for step in range(final_step + 1):
            self.place(memory[step])
        # Display Output:
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
        if self.settings.show_incremental_solution:
            displaySolution(createSolution(self.data, self), self.data.av_length, self.x_limit, self.y_limit, self.settings)

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
