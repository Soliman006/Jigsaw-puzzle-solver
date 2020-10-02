import cv2
import numpy as np
import copy

from graphics import createSolution, displaySolution, imshow, createBGRSolution, displayBGRSolution
from solving.comparator import normaliseContour, normaliseContours, compareColourContours, colourClosestDist, findeuclid_dist, process_contour, remove_duplicates, edist
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
        #self.data.puzzle_row = 9
        #self.data.puzzle_columns = 6
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
        self.solution = []
        self.hardReset()

    def hardReset(self):
        self.flags = Flags()
        self.trials_border = []
        self.trials_interior = []
        self.trials = []
        self.forward_border = []
        self.backwards_border = []
        self.possible_edges = []
        self.possible_rows = []
        self.possible_columns = []
        self.column_scores = []
        self.row_scores = []
        self.edge_scores = []
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
        self.flags.cornerfound = False
        
    def fullsolver_columns(self):
        self.solveborder2()
        for i in range(1,self.data.puzzle_columns-1):
            self.hardReset()
            self.placecurrentsolution()
            self.flags.solvingborder = False
            best_row, best_row_score = self.solvebestcolumn(i)
            for j in best_row:
                self.solution.append(j)
    
    def solveborder2(self):
        best_scores = []
        best_edges = []
        self.setloctype(self.data.puzzle_rows,self.data.puzzle_columns)
        corners = self.processed_corners.copy()
        for i in corners:
            best_edge, best_score = self.solvebestfulledge(i)
            sum_best_edge = 0
            count = 0
            for j in best_edge:
                count = count + 1
                if count == len(best_edge):
                    edge_average = sum_best_edge/(len(best_edge)-2)
                    if j.options[j.choice].score > self.settings.corner_thresh * edge_average:
                        best_score = 10
                else:
                    sum_best_edge = sum_best_edge + j.options[j.choice].score
            best_scores.append(best_score)
            best_edges.append(best_edge)
            self.hardReset()
        for i in best_scores:
            print(i)
        for i in range(len(best_edges)):
            for j in best_edges[i]:
                print(j.options[j.choice].piece, j.options[j.choice].score)
                print()
        best_score_index = np.argmin(best_scores)
        best_overall_edge = best_edges[best_score_index]
        for i in best_overall_edge:
            self.solution.append(i)
        
        best_scores = []
        best_edges = []
        self.removepieces()
        corners = self.processed_corners.copy()
        for i in corners:
            best_edge, best_score = self.solvebestfulledge(i)
            sum_best_edge = 0
            count = 0
            for j in best_edge:
                count = count + 1
                if count == len(best_edge):
                    edge_average = sum_best_edge/(len(best_edge)-2)
                    if j.options[j.choice].score > self.settings.corner_thresh * edge_average:
                        best_score = 10
                else:
                    sum_best_edge = sum_best_edge + j.options[j.choice].score
            best_scores.append(best_score)
            best_edges.append(best_edge)
            self.hardReset()
            self.removepieces()
        for i in best_scores:
            print(i)
        for i in range(len(best_edges)):
            for j in best_edges[i]:
                print(j.options[j.choice].piece, j.options[j.choice].score)
                print()
        best_score_index = np.argmin(best_scores)
        best_overall_edge = best_edges[best_score_index]
        for i in best_overall_edge:
            self.solution.append(i)
        
        self.setloctype(self.data.puzzle_columns,self.data.puzzle_rows)
        
        for i in range(self.data.puzzle_rows):
            self.solution[i].space = [0,self.data.puzzle_rows-1-i]
            if i == self.data.puzzle_rows-1:
                self.solution[i].options[self.solution[i].choice].rotation = 0
            else:
                self.solution[i].options[self.solution[i].choice].rotation = 3
            
            index = i+self.data.puzzle_rows
            self.solution[index].space = [self.data.puzzle_columns-1,i]
            if index == len(self.solution)-1:
                self.solution[index].options[self.solution[index].choice].rotation = 2
            else:
                self.solution[index].options[self.solution[index].choice].rotation = 1
        
        self.hardReset()
        best_scores = []
        best_edges = []
        self.placecurrentsolution()
        self.flags.solvingborder = False
        best_edge, best_score = self.solvebestpartialedge(0)
        best_scores.append(best_score)
        best_edges.append(best_edge)
        self.hardReset()
        self.placecurrentsolution()
        self.flags.solvingborder = False
        best_edge, best_score = self.solvebestpartialedge(self.data.puzzle_rows-1)
        best_scores.append(best_score)
        best_edges.append(best_edge)
        best_score_index = np.argmin(best_scores)
        best_overall_edge = best_edges[best_score_index]
        for i in best_overall_edge:
            self.solution.append(i)
            
        self.hardReset()
        self.placecurrentsolution()
        self.flags.solvingborder = False
        if best_score_index == 0:
            best_edge, best_score = self.solvebestpartialedge(self.data.puzzle_rows-1)
        else:
            best_edge, best_score = self.solvebestpartialedge(0)
        for i in best_edge:
            self.solution.append(i)
        
        self.hardReset()
        self.placecurrentsolution()
        displaySolution(createSolution(self.data, self), self.data.av_length, self.x_limit, self.y_limit, self.settings)

    
    def solvebestfulledge(self,corner):
        sum_score = 0
        potential_edge = []
        while self.flags.solvable != 0:
            while len(self.memory) < self.x_limit:
                space = [len(self.memory),0]
                x = space[0]
                y = space[1]
                if self.settings.show_current_space_text:
                    print(" ")
                    print("Now solving for space", space)
                if self.loc_type[y][x] == 2:  # corner
                    if len(self.memory) == 0:  # starting piece
                        piece = corner
                        rotation = loc_type_detail_to_rotation(self.loc_type_detail[y][x])
                        score = 0
                        option = Option(piece, rotation, score)
                        options = []
                        options.append(option)
                        choice = 0
                        step = Step(space, options, choice)
                    else:
                        step = self.solveSpace(space, self.processed_corners)
                        print('Corner is found')
                if self.loc_type[y][x] == 1:  # edge
                    step = self.solveSpace(space, self.processed_edges)
                if self.flags.backtrack:
                    self.flags.backtrack = 0    
                    final_step, final_option = self.backtrace(self.memory)
                    if self.flags.solvable == 0:
                        break 
                    else:
                        self.backtracker(final_step, final_option)
                        continue
                # Place in puzzle and update
                self.place(step)
                if self.settings.show_solver_progress_text:
                    print("Progress:", self.placement_num, "/", self.num_pieces)
                if self.settings.show_incremental_solution:
                    displaySolution(createSolution(self.data, self), self.data.av_length, self.x_limit, self.y_limit, self.settings)
            
            if self.flags.solvable == 0:
                break
            else:
                potential_edge = list(self.memory)
                self.possible_edges.append(potential_edge)
                for i in range(len(self.memory)):
                    sum_score = sum_score + self.memory[i].options[self.memory[i].choice].score
                self.edge_scores.append(sum_score)
                sum_score = 0
                final_step, final_option = self.backtrace(self.memory)
                if self.flags.solvable == 0:
                    break
                self.backtracker(final_step, final_option)
                self.flags.cornerfound = False
        
        if len(self.edge_scores) == 0:
            best_edge = []
            best_score = 10
        else:
            best_score = min(self.edge_scores)
            best_score_index = np.argmin(self.edge_scores)
            best_edge = self.possible_edges[best_score_index]
        return best_edge, best_score
    
    def solvebestpartialedge(self,row):
        sum_score = 0
        potential_edge = []
        while self.flags.solvable != 0:
            while len(self.memory) < self.x_limit-2:
                space = [len(self.memory)+1,row]
                x = space[0]
                y = space[1]
                if self.settings.show_current_space_text:
                    print(" ")
                    print("Now solving for space", space)
                if self.loc_type[y][x] == 1:  # edge
                    step = self.solveSpace(space, self.processed_edges)
                if self.flags.backtrack:
                    self.flags.backtrack = 0    
                    final_step, final_option = self.backtrace(self.memory)
                    if self.flags.solvable == 0:
                        break 
                    else:
                        self.backtracker(final_step, final_option)
                        continue
                # Place in puzzle and update
                self.place(step)
                if self.settings.show_solver_progress_text:
                    print("Progress:", self.placement_num, "/", self.num_pieces)
                if self.settings.show_incremental_solution:
                    displaySolution(createSolution(self.data, self), self.data.av_length, self.x_limit, self.y_limit, self.settings)
            
            if self.flags.solvable == 0:
                break
            else:
                potential_edge = list(self.memory)
                self.possible_edges.append(potential_edge)
                for i in range(len(self.memory)):
                    sum_score = sum_score + self.memory[i].options[self.memory[i].choice].score
                self.edge_scores.append(sum_score)
                sum_score = 0
                final_step, final_option = self.backtrace(self.memory)
                if self.flags.solvable == 0:
                    break
                self.backtracker(final_step, final_option)
                self.flags.cornerfound = False
        
        best_score = min(self.edge_scores)
        best_score_index = np.argmin(self.edge_scores)
        best_edge = self.possible_edges[best_score_index]
        '''print('best border is number',best_score_index+1,'and the pieces are:')
        for i in best_edge:
            print(i.options[i.choice].piece)'''
        return best_edge, best_score
    
    def solvebestcolumn(self,column):
        sum_score = 0
        potential_column = []
        while self.flags.solvable != 0:
            while len(self.memory) != self.data.puzzle_rows-2:
                space = [column,len(self.memory)+1]
                x = space[0]
                y = space[1]
                if self.settings.show_current_space_text:
                    print(" ")
                    print("Now solving for space", space)

                step = self.solveSpace(space, self.processed_interior)
                if self.flags.backtrack:
                    self.flags.backtrack = 0    
                    final_step, final_option = self.backtrace(self.memory)
                    if self.flags.solvable == 0:
                        break 
                    else:
                        self.backtracker(final_step, final_option)
                        continue
                # Place in puzzle and update
                self.place(step)
                if self.settings.show_solver_progress_text:
                    print("Progress:", self.placement_num, "/", self.num_pieces)
                if self.settings.show_incremental_solution:
                    displaySolution(createSolution(self.data, self), self.data.av_length, self.x_limit, self.y_limit, self.settings)
            
            if self.flags.solvable == 0:
                break
            else:
                potential_column = list(self.memory)
                self.possible_columns.append(potential_column)
                for i in range(len(self.memory)):
                    sum_score = sum_score + self.memory[i].options[self.memory[i].choice].score
                self.column_scores.append(sum_score)
                sum_score = 0
                final_step, final_option = self.backtrace(self.memory)
                if self.flags.solvable == 0:
                    break
                self.backtracker(final_step, final_option)
                self.flags.cornerfound = False
        
        best_score = min(self.column_scores)
        best_score_index = np.argmin(self.column_scores)
        best_column = self.possible_columns[best_score_index]
        '''print('best border is number',best_score_index+1,'and the pieces are:')
        for i in best_edge:
            print(i.options[i.choice].piece)'''
        return best_column, best_score
    
    def solveinterior_column1(self):
        self.hardReset()
        self.placecurrentsolution()
        self.flags.solvingborder = False
        best_column, best_column_score = self.solvebestcolumn(1)
        for i in best_column:
            self.solution.append(i)

    def solveSpace(self, space, pieces):
        """When provided with a location in the puzzle and list of pieces, it will find the best piece to put in the space."""
        optimal_index, optimal_piece, optimal_rotation, optimal_piece_score,\
            n_sides_compared, matches = self.generate_options(space, pieces)
        print(optimal_piece_score,n_sides_compared)
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
    '''
    def generate_options(self, space, pieces):
        space_x = space[0]
        space_y = space[1]
        max_score = 100000000
        optimal_piece_score = max_score
        optimal_rotation = -1
        optimal_index = -1
        optimal_piece = -1
        n_sides = 0
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
                            contour1 = self.data.processed_pieces[space_piece_ref][space_side_ref]
                            contour2 = self.data.processed_pieces[piece][(side + rotation) % 4]
                            contour1, contour2, peak_point1, peak_point2 = normaliseContours(contour1, contour2, self.data.av_length)
                            peak_dist = edist(peak_point1, peak_point2)
                            if peak_dist > self.settings.peak_dist_thresh:
                                rotation_score_total = 10000000
                                break
                            
                            reference = process_contour(contour1)
                            candidate = process_contour(contour2)
                            colour_curve1 = self.data.colour_contours[space_piece_ref][space_side_ref]
                            colour_curve2 = self.data.colour_contours[piece][(side + rotation) % 4]
                            colour_contour_xy1 = self.data.colour_contours_xy[space_piece_ref][space_side_ref]
                            colour_contour_xy2 = self.data.colour_contours_xy[piece][(side + rotation) % 4]
                            colour_contour_xy1, colour_contour_xy2, colour_peak_point1, colour_peak_point2 = normaliseContours(
                                colour_contour_xy1, colour_contour_xy2, self.data.av_length)
                            
                            side_score_colour = compareColourContours(colour_contour_xy1, colour_contour_xy2, colour_curve1, colour_curve2, self.settings)
                            if side_score_colour > self.settings.score_colour_thresh:
                                rotation_score_total = 10000000
                                break
                                
                            side_score_shape = findeuclid_dist(reference,candidate)
                            side_score_total = self.settings.shape_score_frac*side_score_shape + (1-self.settings.shape_score_frac)*side_score_colour
                            
                            rotation_score_shape = rotation_score_shape + side_score_shape
                            rotation_score_colour = rotation_score_colour + side_score_colour
                            rotation_score_total = rotation_score_total + side_score_total
                            n_sides_compared = n_sides_compared + 1
                    if n_sides < n_sides_compared:
                        n_sides = n_sides_compared
                    if self.settings.show_comparison_text:
                        print("Comparing piece", piece, "with rotation", rotation, "to space", space,
                              f'scores: shape {rotation_score_shape:.4f} colour {rotation_score_colour:.4f}'
                              f' total {rotation_score_total:.4f}')
                else:
                    rotation_score_total = 10000000

                if rotation_score_total < piece_score:
                    piece_score = rotation_score_total
                    best_rotation = rotation
            if piece_score < 10000000:
                score_log = [i, piece, best_rotation, piece_score]
                matches.append(score_log)
                if piece_score < optimal_piece_score:
                    optimal_piece_score = piece_score
                    optimal_rotation = best_rotation
                    optimal_index = i
                    optimal_piece = piece
        return optimal_index, optimal_piece, optimal_rotation, optimal_piece_score, n_sides, matches'''
    
    def generate_options(self, space, pieces):
        space_x = space[0]
        space_y = space[1]
        max_score = 100000000
        optimal_piece_score = max_score
        optimal_rotation = -1
        optimal_index = -1
        optimal_piece = -1
        n_sides = 0
        matches = []
        ref_sides = []
        
        for side in range(0,4):
            space_piece_ref = self.space[space_y][space_x][side][0]
            space_side_ref = self.space[space_y][space_x][side][1]
            if space_piece_ref != -1:
                reference = self.data.processed_pieces[space_piece_ref][space_side_ref]
                reference, peak_point = normaliseContour(reference,0,self.data.av_length)
                reference_inter = process_contour(reference)
                ref_side = Ref_side(reference, peak_point, reference_inter)
                ref_sides.append(ref_side)
                n_sides = n_sides + 1
            else:
                ref_sides.append(space_piece_ref)

        for i in range(len(pieces)):
            piece = pieces[i]
            piece_score = max_score
            best_rotation = -1
            for rotation in range(0, 4):
                rotation_score_total = 0
                rotation_score_shape = 0
                rotation_score_colour = 0
                invalid_rotation = False
                if self.validPieceComparison(space, piece, rotation) == 1:
                    n_sides_compared = 0
                    for side in range(0, 4):
                        if ref_sides[side] != -1:  # make sure there is a contour to compare to
                            candidate = self.data.processed_pieces[piece][(side + rotation) % 4]
                            candidate, candidate_peak = normaliseContour(candidate, 1, self.data.av_length)
                            peak_dist = edist(ref_sides[side].peak, candidate_peak)
                            if peak_dist > self.settings.peak_dist_thresh:
                                rotation_score_total = 10000000
                                invalid_rotation = True
                                break
                    if invalid_rotation is True:
                        continue
                    
                    for side in range(0, 4):
                        if ref_sides[side] != -1:
                            space_piece_ref = self.space[space_y][space_x][side][0]
                            space_side_ref = self.space[space_y][space_x][side][1]
                            colour_curve1 = self.data.colour_contours[space_piece_ref][space_side_ref]
                            colour_curve2 = self.data.colour_contours[piece][(side + rotation) % 4]
                            colour_contour_xy1 = self.data.colour_contours_xy[space_piece_ref][space_side_ref]
                            colour_contour_xy2 = self.data.colour_contours_xy[piece][(side + rotation) % 4]
                            colour_contour_xy1, colour_contour_xy2, colour_peak_point1, colour_peak_point2 = normaliseContours(
                                colour_contour_xy1, colour_contour_xy2, self.data.av_length)
                            
                            side_score_colour = compareColourContours(colour_contour_xy1, colour_contour_xy2, colour_curve1, colour_curve2, self.settings)
                            if side_score_colour > self.settings.score_colour_thresh:
                                rotation_score_total = 10000000
                                invalid_rotation = True
                                break
                            rotation_score_colour = rotation_score_colour + side_score_colour
                    
                    if invalid_rotation is True:
                        continue
                    
                    for side in range(0, 4):
                        if ref_sides[side] != -1:
                            candidate = process_contour(candidate)    
                            side_score_shape = findeuclid_dist(ref_sides[side].interpolation,candidate)
                            rotation_score_shape = rotation_score_shape + side_score_shape
                            
                    rotation_score_total = self.settings.shape_score_frac*rotation_score_shape + (1-self.settings.shape_score_frac)*rotation_score_colour
                    if self.settings.show_comparison_text:
                        print("Comparing piece", piece, "with rotation", rotation, "to space", space,
                              f'scores: shape {rotation_score_shape:.4f} colour {rotation_score_colour:.4f}'
                              f' total {rotation_score_total:.4f}')
                else:
                    rotation_score_total = 10000000

                if rotation_score_total < piece_score:
                    piece_score = rotation_score_total
                    best_rotation = rotation
            if piece_score < 10000000:
                score_log = [i, piece, best_rotation, piece_score]
                matches.append(score_log)
                if piece_score < optimal_piece_score:
                    optimal_piece_score = piece_score
                    optimal_rotation = best_rotation
                    optimal_index = i
                    optimal_piece = piece
        return optimal_index, optimal_piece, optimal_rotation, optimal_piece_score, n_sides, matches
    
    def place(self, step):  # space, piece, correct_rotation):
        """Command allowing the user to manually force a certain piece into a certain place in the puzzle."""
        if self.flags.putinmemory:
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
            self.processed_corners.remove(piece)

        if self.loc_type[y][x] == 1:
            # edge
            self.processed_edges.remove(piece)
            
        if self.loc_type[y][x] == 0:
            # interior
            self.processed_interior.remove(piece)
            
        # update the puzzle:
        self.updatePuzzle(space, piece, rotation)
        self.placement_num = self.placement_num + 1
        if self.settings.show_selection_text:
            print("Piece", piece, "with rotation", rotation, "has been placed into space", space)

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
        print(edist(peak_point1,peak_point2))
        colour_contour_xy1 = self.data.colour_contours_xy[piece1][(side1 + rotation1) % 4]
        colour_contour_xy2 = self.data.colour_contours_xy[piece2][(side2 + rotation2) % 4]
        colour_contour_xy1, colour_contour_xy2, colour_peak_point1, colour_peak_point2 = normaliseContours(
            colour_contour_xy1, colour_contour_xy2, self.data.av_length)
        colour_curve1 = self.data.colour_contours[piece1][(side1 + rotation1) % 4]
        colour_curve2 = self.data.colour_contours[piece2][(side2 + rotation2) % 4]

        reference = process_contour(contour1)
        candidate = process_contour(contour2)
        print('number of points in contour1 = ',len(contour1),' and contour2 = ',len(contour2))
        print('number of points in reference = ',len(reference),' and candidate = ',len(candidate))

        '''score_shape, score_colour, score_total = compareContours(
            contour1, contour2, colour_curve1, colour_curve2, colour_contour_xy1, colour_contour_xy2, self.data.av_length, self.settings)'''
        score_colour = compareColourContours(colour_contour_xy1, colour_contour_xy2, colour_curve1, colour_curve2, self.settings)
        score_shape = findeuclid_dist(reference,candidate)
        score_total = self.settings.shape_score_frac*score_shape + (1-self.settings.shape_score_frac)*score_colour
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
        memory = copy.deepcopy(self.memory)
        memory[final_step].choice = final_option
        self.reset()
        if self.flags.solvingborder:
            self.removepieces()
        else:
            self.placecurrentsolution()
            
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
            
    def removepieces(self):
        for i in self.solution:
            x = i.space[0]
            y = i.space[1]
            if self.loc_type[y][x] == 2:
                self.processed_corners.remove(i.options[i.choice].piece)
            if self.loc_type[y][x] == 1:
                self.processed_edges.remove(i.options[i.choice].piece)
            if self.loc_type[y][x] == 0:
                self.processed_interior.remove(i.options[i.choice].piece)
    
    def removecertainpieces(self,start,end):
        for i in self.solution[start:end]:
            x = i.space[0]
            y = i.space[1]
            if self.loc_type[y][x] == 2:
                self.processed_corners.remove(i.options[i.choice].piece)
            if self.loc_type[y][x] == 1:
                self.processed_edges.remove(i.options[i.choice].piece)
            if self.loc_type[y][x] == 0:
                self.processed_interior.remove(i.options[i.choice].piece)
    
    def placecurrentsolution(self):
        self.flags.putinmemory = False
        for i in self.solution:
            self.place(i)
        self.flags.putinmemory = True
        
    def placespecificsolution(self,index):
        self.flags.putinmemory = False
        self.place(self.solution[index])
        self.flags.putinmemory = True
                
    def setloctype(self,xlimit,ylimit):
        self.y_limit = ylimit
        self.x_limit = xlimit
        self.reset()
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
                
    def truncate(self, list, length):
        if len(list) > length:
            list = list[0:length]
        return list

    def rankBorders(self):
        """Ranks all the border attempts based on average match score."""
        rank_list = []
        for trial in range(len(self.trials_border)):
            border = self.trials_border[trial]
            score_sum = 0
            score_count = 0
            for step in range(len(border)):
                choice = border[step].choice
                score = border[step].options[choice].score
                score_sum = score_sum + score
                score_count = score_count + 1
            score_av = score_sum / score_count
            entry = [score_av, border]
            rank_list.append(entry)
        rank_list.sort()
        rank_borders = []
        for entry in range(len(rank_list)):
            border = rank_list[entry][1]
            rank_borders.append(border)
        self.trials_border = rank_borders
        if self.settings.show_backtracker:
            for entry in range(len(self.trials_border)):
                memory = self.trials_border[entry]
                count = 0
                for step in range(len(memory)):
                    count = count + 1
                    choice = memory[step].choice
                    print(choice, end=" ")
                if count < 54:
                    for i in range(54 - count):
                        print("X", end=" ")
                score = rank_list[entry][0]
                print(round(score, 6), end=" ")
                print("")

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

class Ref_side:
    "Stores the original contour, lock_peak and processed contour for a space side"
    
    def __init__(self, contour, peak, interpolation):
        """Initialisation"""
        self.contour = contour
        self.peak = peak
        self.interpolation = interpolation

class Flags:
    """Stores flags."""

    def __init__(self):
        """Initialisation"""
        self.solvable = 1
        self.backtrack = 0
        self.all_borders_tried = 0
        self.solve_interior = 0
        self.cornerfound = False
        self.putinmemory = True
        self.solvingborder = True
