import random

def generate_points_blue_noise(grid_x, grid_y, n_points, radius = 1.5, num_attempts = 8):
    # Setting radius < sqrt(2) will break this algorithm
    # The grid size is assumed to be 1 on each side.
    # Adapted from https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf

    def frisbee_sample():
        x, y = 2 * radius, 2 * radius
        rsq = radius * radius
        distsq = x * x + y * y
        while (distsq > 4 * rsq or distsq < rsq):
            x = random.uniform(-2 * radius, 2 * radius)
            y = random.uniform(-2 * radius, 2 * radius)
            distsq = x * x + y * y
        return x, y

    def frisbee_inside_grid(x, y):
        out_x, out_y = -1, -1
        while(out_x < 0 or out_y < 0 or out_x >= grid_x or out_y >= grid_y):
            rand_x, rand_y = frisbee_sample()
            out_x = x + rand_x
            out_y = y + rand_y
        return out_x, out_y

    grid = [[-1 for i in range(grid_y)] for j in range(grid_x)]
    cells = []
    active = []

    init_sample = (random.uniform(0, grid_x - 1e-5), random.uniform(0, grid_y - 1e-5))
    cells.append(init_sample)
    grid[int(init_sample[0])][int(init_sample[1])] = 0
    active.append(0)

    while len(cells) < n_points and len(active) > 0:
        active_index = random.randint(0, len(active) - 1)
        index = active[active_index]
        cur_point = cells[index]
        added_point = False
        for attempt_i in range(num_attempts):
            try_x, try_y = frisbee_inside_grid(*cur_point)
            valid = True
            for check_x in range(int(try_x - radius), int(try_x + radius) + 1):
                if check_x < 0 or check_x >= grid_x:
                    continue
                if not valid:
                    break
                for check_y in range(int(try_y - radius), int(try_y + radius) + 1):
                    if check_y < 0 or check_y >= grid_y:
                        continue
                    if grid[check_x][check_y] == -1:
                        continue # not occupied
                    neighbor = cells[grid[check_x][check_y]]
                    dx = neighbor[0] - try_x
                    dy = neighbor[1] - try_y
                    if dx * dx + dy * dy < radius * radius:
                        valid = False # too close
                        break
            if valid:
                cells.append((try_x, try_y))
                grid[int(try_x)][int(try_y)] = len(cells) - 1
                active.append(len(cells) - 1)
                added_point = True
                break
        if not added_point:
            del active[active_index]

    return cells