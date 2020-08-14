import copy


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
