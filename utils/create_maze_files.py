from discrete_maze.grid_maze import GridExplore
from discrete_maze.maze import ExploreTask
import dill

gerem8 = GridExplore(8, 8)
with open('saved_mazes/gerem8.dill', 'wb') as f:
    dill.dump(gerem8, f)

gerem16 = GridExplore(16, 16)
with open('saved_mazes/gerem16.dill', 'wb') as f:
    dill.dump(gerem16, f)
