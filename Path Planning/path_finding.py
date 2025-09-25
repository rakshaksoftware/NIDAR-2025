def find_shortest_path(targets, start_pos):
    """Simple nearest neighbor path algorithm"""
    unvisited = targets[:]
    path = []
    current = start_pos

    while unvisited:
        nearest = min(unvisited, key=lambda p: distance(current, p))
        path.append(nearest)
        current = nearest
        unvisited.remove(nearest)

    return path
