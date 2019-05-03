import matplotlib.pyplot as plt
import numpy as np
from .blue_noise import generate_points_blue_noise
from scipy.spatial import Delaunay
import random
import math
import time
from collections import deque

class MazeChoice:
    def __init__(self, angle_divisions):
        self.n = angle_divisions

    def sample(self):
        return random.randint(0, self.n - 1)

class MazeObservation:
    def __init__(self, history_size, angle_divisions, id_size):
        # Second coordinate is angle_divisions + 1 because
        # we include the id of the node the agent is on in the
        # observation
        self.shape = [history_size, angle_divisions + 1, id_size]
        self.dtype = np.int8

class ExploreTask:
    """
    In general this matches the OpenAI Gym API, no guarantees though...
    Mostly don't follow it so that we can initialize mazes of different
    sizes and types.

    This task puts an agent on a random node in a randomly generated
    planar graph, and rewards the agent for quickly finding an "exit"
    node.

    Observations are numpy arrays of dtype int8.
    """
    point_dist = 1.5
    reward_types = ['distance', 'only_finished', 'penalties', 'penalty+finished']

    def __init__(self, n_points, is_tree = True, max_allowed_step_ratio = 2.5,
                angle_divisions = 16, id_size = 5,
                history_size = 1,
                place_agent_far_from_dest = True,
                agent_placement_prop = 0.8,
                reward_type = 'penalty+finished',
                scale_reward_by_difficulty = True,
                grid_maze = False):
        """
        It's not strictly guaranteed that you will get n_points in the graph, 
        as potentially there might be fewer due to the generating grid not being large enough.
        However, I have not seen this happen.

        max_allowed_step_ratio determines how quickly an episode will end. We multiply
        it by the number of nodes in the graph to get how many steps we allow in the episode.

        angle_divisions is the number of angles along which we sample for edges in an observation,
        as well as the size of the action space. Making this finer will help give more positioning
        information, but will be more expensive. If grid_maze is set to True this is overridden
        to be 4.

        id_size is the length of each "identifier" for a node, each feature of which is uniformly
        randomly selected from [-1, 0, 1]

        history_size determines how many previous frames we give the agent. Frames which don't exist
        are just set to 0, which uniquely identifies a non-existent frame as each valid frames must
        have frame[0][0] == 1

        place_agent_far_from_dest will use BFS to find the distances for each node from the end node,
        treating all edges as the same length as the actual distance does not matter. It will then take a
        random node more than MAX_DISTANCE * agent_placement_prop away from the end node and place the
        agent there. Turning this option on helps make random actions less likely to finish the task,
        and helps make the gradients more meaningful as otherwise some environments may place the agent
        too close to the destination node, falsely giving the sense that that particular agent's behavior
        was somehow better. This also doesn't seem to slow down initialization too much.

        reward_type alters the way this environment doles out rewards.
            * only_finished: will just give a reward of 1.0 when the destination is reached, and will not give
              negative rewards.
            * penalities: will output -1.0 every time step the goal has not been reached
            * distance: will output 1.0 if agent gets closer to the goal, and -1.0 if agent remains same distance
              or gets further away
            * penalty+finished: will output 1.0 when agent reaches goal, -.01 every time step, and an extra -.01
              every time it performs an invalid action. This is based on the SNAIL paper's maze task reward shaping.

        scale_reward_by_difficulty will scale the rewards by
            (average length of random path) / (probability random path will finish)
        This does increase the time to run .reset() significantly, doubling the time necessary to initialize
        at around maze size 40.

        grid_maze will make a random maze on integer coordinates, with edges only going up, down, left, or right.
            This reduces angle_divisions to 4.
        """

        self.grid_maze = grid_maze
        if grid_maze:
            angle_divisions = 4

        self.action_space = MazeChoice(angle_divisions)
        self.observation_space = MazeObservation(history_size, angle_divisions, id_size)
        self.id_size = id_size
        self.history_size = history_size
        self.place_agent_far_from_dest = place_agent_far_from_dest
        self.agent_placement_prop = agent_placement_prop

        assert reward_type in self.reward_types, "Reward type passed in to ExploreTask not valid."
        self.reward_type = reward_type

        self.scale_reward_by_difficulty = scale_reward_by_difficulty

        map_size = int(math.sqrt(n_points) * self.point_dist * 2)
        self.points, edges = generate_points_blue_noise(map_size,
                                map_size,
                                n_points,
                                radius = self.point_dist,
                                provide_edges = is_tree)

        # would be annoying if this actually activated.
        # assert self.points.shape[0] == n_points

        self.edge_list = [[] for p in self.points]
        self.point_ids = np.random.randint(-1, 2, size = (n_points, self.id_size), dtype = np.int8)

        # this is so that points are always distinguishable from empty parts
        # of the observation.
        self.point_ids[:, 0] = 1
        
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
        
        self.end_node = random.randint(0, len(self.points) - 1)

        # Figure for rendering plots
        self.fig = None

        # We allow the agent to go to each node approximately max_allowed_step_ratio
        # times before terminating the session.
        self.max_allowed_steps = int(max_allowed_step_ratio * self.points.shape[0])

        self.distances = None

    def _angle_index(self, node_i):
        point = self.points[node_i]
        dx, dy = point[0] - self.points[self.agent, 0], point[1] - self.points[self.agent, 1]
        angle_index = int((np.arctan2(dy, dx) / np.pi + 1) * self.action_space.n)

        # Correcting for possible floating point error
        return max(0, min(angle_index, self.action_space.n - 1))

    def _observation(self):
        # obs has time dimension 1
        obs = np.zeros([1] + self.observation_space.shape[1:], dtype = np.int8)
        obs[0, 0] = self.point_ids[self.agent]
        agent_p = self.points[self.agent]
        for node_i in self.edge_list[self.agent]:
            angle_index = self._angle_index(node_i)
            # Should not be able to overwrite observations, but no guarantees...
            # If it does, it happens rarely enough as not to matter.
            obs[0, angle_index + 1] = self.point_ids[node_i]

        self.observation_history = np.delete(self.observation_history, 0, 0)
        self.observation_history = np.append(self.observation_history, obs, axis = 0)
        return self.observation_history


    def step(self, action):
        self.n_steps += 1
        info = {}
        if self.reward_type == 'distance':
            old_distance = self.distances[self.agent]

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

        if self.reward_type == 'penalties':
            rew = 0.0 if done else -1 / self.difficulty
        elif self.reward_type == 'only_finished':
            rew = self.difficulty if done else 0.0
        elif self.reward_type == 'penalty+finished':
            rew = self.difficulty if done else -.01 / self.difficulty
            if not matched:
                rew -= .01 / self.difficulty
        else:
            ddist = self.distances[self.agent] - old_distance
            rew = (1 / self.difficulty) if ddist < 0 else (-1 / self.difficulty)

        if self.n_steps > self.max_allowed_steps:
            done = True
            info['truncated'] = True
        else:
            info['truncated'] = False

        return self._observation(), rew, done, info

    def reset(self):
        """
        Will never create a new maze, but will just place agent on random node.
        See note for __init__ to see effect of self.place_agent_far_from_dest
        """
        self.n_steps = 0
        self.observation_history = np.zeros(self.observation_space.shape, dtype = np.int8)

        if self.place_agent_far_from_dest or self.reward_type == 'distance' and not self.distances:
            q = deque([self.end_node])
            self.distances = [None for _ in range(self.points.shape[0])]
            visited = [False for _ in range(self.points.shape[0])]
            self.distances[self.end_node] = 0
            visited[self.end_node] = True
            max_dist = 0
            while len(q) > 0:
                node_a = q.popleft()
                for node_b in self.edge_list[node_a]:
                    if not visited[node_b]:
                        self.distances[node_b] = 1 + self.distances[node_a]
                        max_dist = max(max_dist, self.distances[node_b])
                        visited[node_b] = True
                        q.append(node_b)

        if not self.place_agent_far_from_dest:
            self.agent = random.randint(0, len(self.points) - 1)
            while self.agent == self.end_node:
                self.agent = random.randint(0, len(self.points) - 1)
        else:
            cutoff_dist = int(max_dist * self.agent_placement_prop)
            # If cutoff_dist == 1 we have to check it's not the end node
            self.placement_candidates = [i for i, d in enumerate(self.distances) if d >= cutoff_dist and i != self.end_node]
            self.agent = random.choice(self.placement_candidates)
        
        if self.scale_reward_by_difficulty:
            n_points = self.points.shape[0]

            end_ls = []
            end_ps = []
            p_acc = 1.0

            probabilities = [0.0 for i in range(n_points)]
            probabilities[self.agent] = 1.0
            for step_i in range(self.max_allowed_steps):
                new_probabilities = [0.0 for i in range(n_points)]
                for node_a, prob in enumerate(probabilities):
                    accumulate = prob / len(self.edge_list[node_a])
                    if accumulate == 0.0:
                        continue
                    for node_b in self.edge_list[node_a]:
                        new_probabilities[node_b] += accumulate
                probabilities = new_probabilities

                cur_p = probabilities[self.end_node]
                if cur_p != 0.0:
                    end_ls.append(step_i + 1)
                    end_ps.append(p_acc * cur_p)
                    p_acc *= 1 - cur_p
            weighted_l = sum([l * p for l,p in zip(end_ls, end_ps)])
            p_inv = 1 / sum(end_ps)
            self.difficulty = weighted_l * p_inv * p_inv
        else:
            self.difficulty = 1.0

        return self._observation()

    def render(self, debug = False):
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
            if debug:
                placement_points = self.points[self.placement_candidates]
                ax.plot(placement_points[:, 0], placement_points[:, 1], 'co', markersize = 22)
            # we draw the destination larger so that we can overlay the agent
            ax.plot(*self.points[self.end_node], 'yo', markersize = 22)
            self.agent_display, = ax.plot(*self.points[self.agent], 'bo', markersize = 15)
        else:
            self.agent_display.set_xdata(self.points[self.agent, 0])
            self.agent_display.set_ydata(self.points[self.agent, 1])
        
        self.fig.suptitle('Step #%d' % self.n_steps)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        if self.fig:
            plt.close(fig = self.fig)
