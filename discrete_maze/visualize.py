import matplotlib.pyplot as plt
import numpy as np
from blue_noise import generate_points_blue_noise
from scipy.spatial import Delaunay

def visualize(points, edges):
    points = np.array(points)

    edge_coords = []
    for node_a, node_b in edges:
        edge_coords.append([points[node_a][0], points[node_b][0]])
        edge_coords.append([points[node_a][1], points[node_b][1]])
    plt.plot(*edge_coords)
    plt.plot(points[:, 0], points[:, 1], 'ro')
    plt.show()

    tri = Delaunay(points)
    # plt.triplot(points[:,0], points[:,1], tri.simplices.copy())

    edge_coords = []
    indices, indptr = tri.vertex_neighbor_vertices
    for index_i in range(indices.shape[0] - 1):
        for index in range(indices[index_i], indices[index_i + 1]):
            edge_coords.append([points[index_i][0], points[indptr[index]][0]])
            edge_coords.append([points[index_i][1], points[indptr[index]][1]])
    plt.plot(*edge_coords)
    plt.plot(points[:,0], points[:,1], 'ro')
    plt.show()

visualize(*generate_points_blue_noise(50, 50, 100))
