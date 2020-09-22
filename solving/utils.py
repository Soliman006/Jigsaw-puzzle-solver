import copy
import numpy as np


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


def loc_type_detail_to_rotation(loc):
    """For border and corner pieces, this function returns the correct rotation to have flat edges on the outside."""
    if loc == 1:
        return 0
    if loc == 2:
        return 0
    if loc == 3:
        return 1
    if loc == 4:
        return 3
    if loc == 5:
        return 1
    if loc == 6:
        return 3
    if loc == 7:
        return 2
    if loc == 8:
        return 2
    if loc == 0:
        return -1


def setProcessedLists(corners, edges, interior):
    """Creates lists of pieces that can be edited during the solving process without altering the source data."""
    processed_corners = copy.deepcopy(corners)
    processed_edges = copy.deepcopy(edges)
    processed_interior = copy.deepcopy(interior)
    for index in range(len(corners)):
        processed_corners[index] = index
    for index in range(len(edges)):
        processed_edges[index] = index + len(corners)
    for index in range(len(interior)):
        processed_interior[index] = index + len(corners) + len(edges)
    return processed_corners, processed_edges, processed_interior


def locTypeInit(x_limit, y_limit, short):
    # Location type
    loc_type = np.full((y_limit, x_limit), 0)
    loc_type[0][0] = 2
    loc_type[0][np.size(loc_type, 1) - 1] = 2
    loc_type[np.size(loc_type, 0) - 1][0] = 2
    loc_type[np.size(loc_type, 0) - 1][np.size(loc_type, 1) - 1] = 2
    for x in range(1, np.size(loc_type, 1) - 1):
        loc_type[0][x] = 1
    for x in range(1, np.size(loc_type, 1) - 1):
        loc_type[np.size(loc_type, 0) - 1][x] = 1
    for y in range(1, np.size(loc_type, 0) - 1):
        loc_type[y][0] = 1
    for y in range(1, np.size(loc_type, 0) - 1):
        loc_type[y][np.size(loc_type, 1) - 1] = 1
    loc_type[0][short - 1] = 3
    # Detailed location type
    loc_type_detail = np.full((y_limit, x_limit), 0)
    loc_type_detail[0][0] = 1
    loc_type_detail[0][np.size(loc_type_detail, 1) - 1] = 3
    loc_type_detail[np.size(loc_type_detail, 0) - 1][0] = 6
    loc_type_detail[np.size(loc_type_detail, 0) - 1][np.size(loc_type_detail, 1) - 1] = 8
    for x in range(1, np.size(loc_type_detail, 1) - 1):
        loc_type_detail[0][x] = 2
    for x in range(1, np.size(loc_type_detail, 1) - 1):
        loc_type_detail[np.size(loc_type_detail, 0) - 1][x] = 7
    for y in range(1, np.size(loc_type_detail, 0) - 1):
        loc_type_detail[y][0] = 4
    for y in range(1, np.size(loc_type_detail, 0) - 1):
        loc_type_detail[y][np.size(loc_type_detail, 1) - 1] = 5
    return loc_type, loc_type_detail


def priorityInit(x_limit, y_limit):
    priority = np.full((y_limit, x_limit), 1)
    for y in range(y_limit):
        priority[y][0] = 8  # left
        priority[y][x_limit - 1] = 4  # right
    priority[y_limit - 1][:] = 6  # bottom
    priority[0][:] = 10  # top
    return priority


def rankPaths(paths, settings):
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
    if settings.show_backtracker:
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


def interpolate_curve(curve, settings):
    n_del = 0
    n_add = 0
    for count in range(1, len(curve)):
        i = count - n_del + n_add
        d = curve[i] - curve[i-1]
        if max(abs(d)) == 0:
            curve = np.delete(curve, i, axis=0)
            n_del = n_del + 1
        elif max(abs(d)) > settings.interpolation_e:
            dx = d[0]
            dy = d[1]
            x_sign = int(dx > 0) - int(dx < 0)
            y_sign = int(dy > 0) - int(dy < 0)
            new_points = [[0, 0]]
            if abs(dx) >= abs(dy):
                ratio = abs(dy)/abs(dx)
                for k in range(1, abs(dx)):
                    point = [[curve[i-1][0] + x_sign*k, round(curve[i-1][1] + y_sign*ratio*k)]]
                    new_points = np.append(new_points, point, axis=0)
            else:
                ratio = abs(dx)/abs(dy)
                for k in range(1, abs(dy)):
                    point = [[round(curve[i-1][0] + x_sign*ratio*k), curve[i-1][1] + y_sign*k]]
                    new_points = np.append(new_points, point, axis=0)
            new_points = np.delete(new_points, 0, axis=0)
            n_add = n_add + len(new_points)
            curve = np.insert(curve, i, new_points, axis=0)
    return curve
