from discrete_maze.grid_maze import GridExplore
from discrete_maze.maze import ExploreTask
import dill

for size in [4, 8, 16, 32]:
    grid = GridExplore(size, size, remember = False)
    with open('saved_mazes/grid_%d.dill' % size, 'wb') as f:
        dill.dump(grid, f)
    grid_rem = GridExplore(size, size, remember = True)
    with open('saved_mazes/grid_rem_%d.dill' % size, 'wb') as f:
        dill.dump(grid_rem, f)

    for is_tree, name in [(False, 'delaunay'), (True, 'tree')]:
        task = ExploreTask(size, reward_type = 'penalty+finished', is_tree = is_tree)
        with open('saved_mazes/%s_%d.dill' % (name, size), 'wb') as f:
            dill.dump(task, f)
        task_rem = ExploreTask(size, reward_type = 'remember', is_tree = is_tree)
        with open('saved_mazes/%s_rem_%d.dill' % (name, size), 'wb') as f:
            dill.dump(task_rem, f)
