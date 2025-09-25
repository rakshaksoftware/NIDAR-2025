# S-shaped path through grid centers
def generate_s_path_centers():
    centers = []
    for row in range(rows):
        if row % 2 == 0:  # Left to right (even rows)
            for col in range(cols):
                x = grid_start_x + col * GRID_SIZE + GRID_SIZE // 2
                y = grid_start_y + row * GRID_SIZE + GRID_SIZE // 2
                centers.append((x, y))
        else:  # Right to left (odd rows)
            for col in range(cols-1, -1, -1):
                x = grid_start_x + col * GRID_SIZE + GRID_SIZE // 2
                y = grid_start_y + row * GRID_SIZE + GRID_SIZE // 2
                centers.append((x, y))
    return centers
