def incrementPriorities(center_x, center_y, loc_type, puzzle_priority, x_limit, y_limit):
    """Updates a location in the puzzle as solved and increases the priority of the locations around it that have yet to be solved."""
    puzzle_priority[center_y][center_x] = -1
    left = [center_x - 1, center_y]
    right = [center_x + 1, center_y]
    above = [center_x, center_y - 1]
    below = [center_x, center_y + 1]
    puzzle_priority = incrementPriority(left, puzzle_priority, x_limit, y_limit)
    puzzle_priority = incrementPriority(right, puzzle_priority, x_limit, y_limit)
    puzzle_priority = incrementPriority(above, puzzle_priority, x_limit, y_limit)
    puzzle_priority = incrementPriority(below, puzzle_priority, x_limit, y_limit)
    return puzzle_priority


def incrementPriority(point, puzzle_priority, x_limit, y_limit):
    """Increases the priority of a spot in the puzzle, indicates to the solver that it is now a better location to attempt to solve."""
    x = point[0]
    y = point[1]
    if ((x != -1) & (x != x_limit) & (y != -1) & (y != y_limit)):
        if (puzzle_priority[y][x] != -1):
            puzzle_priority[y][x] = puzzle_priority[y][x] + 1
    return puzzle_priority


def updateCurves(center_x, center_y, piece, rot, puzzle_space, x_limit, y_limit):
    """Updates the memory of pieces placed in the puzzle, so that the correct edges can be recalled when trying to place future pieces."""
    y = center_y
    x = center_x
    center = [x, y]
    for space_side in range(0, 4):
        updateCurve(center, space_side, piece, space_side + rot, puzzle_space, x_limit, y_limit)
    left = [x - 1, y]
    right = [x + 1, y]
    above = [x, y - 1]
    below = [x, y + 1]
    puzzle_space = updateCurve(above, 2, piece, (0 + rot) % 4, puzzle_space, x_limit, y_limit)
    puzzle_space = updateCurve(left, 3, piece, (1 + rot) % 4, puzzle_space, x_limit, y_limit)
    puzzle_space = updateCurve(below, 0, piece, (2 + rot) % 4, puzzle_space, x_limit, y_limit)
    puzzle_space = updateCurve(right, 1, piece, (3 + rot) % 4, puzzle_space, x_limit, y_limit)
    return puzzle_space


def updateCurve(space, space_side, piece, piece_side, puzzle_space, x_limit, y_limit):
    """Updates one side of one space in the puzzle solution."""
    x = space[0]
    y = space[1]
    if ((x != -1) & (x != x_limit) & (y != -1) & (y != y_limit)):
        puzzle_space[y][x][space_side][0] = piece
        puzzle_space[y][x][space_side][1] = piece_side
    return puzzle_space
