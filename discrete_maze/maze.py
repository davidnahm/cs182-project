import matplotlib.pyplot as plt
import numpy as np
from .blue_noise import generate_points_blue_noise
from scipy.spatial import Delaunay
import random
import math

class MazeChoice:
    size = 16
    def sample(self):
        return random.randint(0, self.size - 1)


class ExploreTask:
    """
    In general this matches the OpenAI Gym API, no guarantees though...
    Mostly don't follow it so that we can initialize mazes of different
    sizes and types.

    This task puts an agent on a random node in a randomly generated
    planar graph, and rewards the agent for quickly finding an "exit"
    node.
    """
    action_space = MazeChoice()
    point_dist = 1.5
    id_size = 5

    def __init__(self, n_points, is_tree = True):
        map_size = int(math.sqrt(n_points) * self.point_dist * 2)
        self.points, edges = generate_points_blue_noise(map_size,
                                map_size,
                                n_points,
                                radius = self.point_dist,
                                provide_edges = is_tree)
        self.edge_list = [[] for p in self.points]
        self.point_ids = np.random.randint(-1, 2, size = (n_points, self.id_size))

        # this is so that points are always distinguishable from empty parts
        # of the observation.
        self.point_ids[:, 0] = 1.0
        
        if not is_tree:
            tri = Delaunay(self.points)
            indices, indptr = tri.vertex_neighbor_vertices
            for index_i in range(indices.shape[0] - 1):
                for index in range(indices[index_i], indices[index_i + 1]):
                    pa, pb = self.points[index_i], self.points[indptr[index]]
                    # Faster to do this manually than with numpy operations
                    dx, dy = pa[0] - pb[0], pa[1] - pb[1]

                    # This makes the edges a bit sparser, so that we don't get
                    # edges that form small angles.
                    if dx * dx + dy * dy < self.point_dist * self.point_dist * 4:
                        self.edge_list[index_i].append(indptr[index])
        else:
            for node_a, node_b in edges:
                self.edge_list[node_a].append(node_b)            
                self.edge_list[node_b].append(node_a)
        
        # First coordinate is 1 + action_space.size because
        # we include the id of the node the agent is on in the
        # observation
        self.obs_size = [self.action_space.size + 1, self.id_size]
        self.end_node = random.randint(0, len(self.points) - 1)

        # Figure for rendering plots
        self.fig = None

        # We allow the agent to go to each node approximately 5
        # times before terminating the session.
        self.max_allowed_steps = 5 * self.points.shape[0]

    def _angle_index(self, node_i):
        point = self.points[node_i]
        dx, dy = point[0] - self.points[self.agent, 0], point[1] - self.points[self.agent, 1]
        angle_index = int((np.arctan2(dy, dx) / np.pi + 1) * self.action_space.size)

        # Correcting for possible floating point error
        return max(0, min(angle_index, self.action_space.size - 1))

    def _observation(self):
        obs = np.zeros(self.obs_size)
        obs[0] = self.point_ids[self.agent]
        agent_p = self.points[self.agent]
        for node_i in self.edge_list[self.agent]:
            angle_index = self._angle_index(node_i)
            # Should not be able to overwrite observations, but no guarantees...
            # If it does, it happens rarely enough as not to matter.
            obs[angle_index + 1] = self.point_ids[node_i]
        return obs


    def step(self, action):
        self.n_steps += 1
        info = {}

        # searching if one of the neighbors is in the direction
        # of the action, and moving to that location if it is.
        matched = False
        for node_i in self.edge_list[self.agent]:
            if self._angle_index(node_i) == action:
                self.agent = node_i
                matched = True
                break
        info['correct_direction'] = matched

        done = self.agent == self.end_node
        rew = 0.0 if done else -1.0

        if self.n_steps > self.max_allowed_steps:
            done = True
            info['truncated'] = True
        else:
            info['truncated'] = False

        return self._observation(), rew, done, info

    def reset(self):
        # Will never create a new maze, but will just place
        # agent on random node.
        self.n_steps = 0
        self.agent = random.randint(0, len(self.points) - 1)
        while self.agent == self.end_node:
            self.agent = random.randint(0, len(self.points) - 1)
        return self._observation()

    def render(self):
        if not self.fig:
            plt.ion()
            self.fig = plt.figure()
            ax = self.fig.add_subplot(111)

            edge_coords = []
            for node_a, neighbors in enumerate(self.edge_list):
                for node_b in neighbors:
                    if node_b < node_a:
                        pass
                    edge_coords.append([self.points[node_a][0], self.points[node_b][0]])
                    edge_coords.append([self.points[node_a][1], self.points[node_b][1]])
            ax.plot(*edge_coords)

            ax.plot(self.points[:,0], self.points[:,1], 'ro')
            self.agent_display, = ax.plot(*self.points[self.agent], 'bo', markersize = 20)
            ax.plot(*self.points[self.end_node], 'yo', markersize = 20) # destination
        else:
            self.agent_display.set_xdata(self.points[self.agent, 0])
            self.agent_display.set_ydata(self.points[self.agent, 1])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        if self.fig:
            self.fig.close()
