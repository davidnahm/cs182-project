import matplotlib.pyplot as plt
import numpy as np
from blue_noise import generate_points_blue_noise

def visualize(grid_x, grid_y, points):
    points = np.array(points)
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.show()

visualize(5, 5, generate_points_blue_noise(15, 15, 50))
