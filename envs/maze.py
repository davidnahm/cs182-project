import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, ImageFrame, TextFrame
from ..params import DEFAULT_PARAMS

class Maze(MiniWorldEnv):
    """
    Maze environment in which the agent has to reach a red box
    """

    def __init__(
        self,
        num_rows=8,
        num_cols=8,
        room_size=3,
        max_episode_steps=None,
        reuse_maze=False,
        **kwargs
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.room_size = room_size
        self.gap_size = 0.25
        self.reuse_maze = reuse_maze
        self.episode_num = 1

        super().__init__(
            max_episode_steps = max_episode_steps or num_rows * num_cols * 24,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        rows = []
        
        # For each row
        for j in range(self.num_rows):
            row = []

            # For each column
            for i in range(self.num_cols):

                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex='brick_wall',
                    #floor_tex='asphalt'
                )
                row.append(room)

            rows.append(row)

        visited = set()

        def visit(i, j):
            """
            Recursive backtracking maze construction algorithm
            https://stackoverflow.com/questions/38502
            """

            room = rows[j][i]

            visited.add(room)

            # Reorder the neighbors to visit in a random order
            neighbors = self.rand.subset([(0,1), (0,-1), (-1,0), (1,0)], 4)

            # For each possible neighbor
            for dj, di in neighbors:
                ni = i + di
                nj = j + dj

                if nj < 0 or nj >= self.num_rows:
                    continue
                if ni < 0 or ni >= self.num_cols:
                    continue

                neighbor = rows[nj][ni]

                if neighbor in visited:
                    continue

                if di == 0:
                    self.connect_rooms(room, neighbor, min_x=room.min_x, max_x=room.max_x)
                elif dj == 0:
                    self.connect_rooms(room, neighbor, min_z=room.min_z, max_z=room.max_z)

                visit(ni, nj)
        
        def get_coords(wall_num, room):
            room_width = 3
            space_between = 0.25

            if wall_num == 0:
                x = room.max_x 
                z = room.mid_z 
                dir = math.pi
            elif wall_num == 1:
                x = room.mid_x
                z = room.min_z
                dir = 3 * math.pi / 2
            elif wall_num == 2:
                x = room.min_x 
                z = room.mid_z 
                dir = 0
            elif wall_num == 3:
                x = room.mid_x 
                z = room.max_z 
                dir = math.pi / 2
            return ([x, 1.35, z], dir)
        # Generate the maze starting from the top-left corner
        visit(0, 0)
        for i in range(len(rows)):
            for j in range(len(rows[0])):
                #self.place_entity(TextFrame(None, 0, 'abc', height=1, depth=3), room=rows[i][j])
                room = rows[i][j]
                min_x, max_x = room.min_x, room.max_x
                min_z, max_z = room.min_z, room.max_z
                print(min_x, max_x)
                print(min_z, max_z)
    
                open_walls = []
                for k in range(len(room.portals)):
                    if len(room.portals[k]) == 0:
                        open_walls.append(k)
                for a in range(len(open_walls)):
                    pos, dir = get_coords(open_walls[a], room)
                    print(open_walls[a], pos)
                    self.entities.append(TextFrame(pos=pos, dir=dir, str=str(a), height=2, depth=3))
                
        #self.place_entity(TextFrame((0, 0, 0), math.pi/2, 'abc'))
        self.box = self.place_entity(Box(color='red'))

        self.place_agent()
        self.initial_agent_pos = self.agent.pos
        self.initial_agent_dir = self.agent.dir


    def get_coords(self, wall_num, room):
        room_width = 3
        space_between = 0.25
        
        if wall_num == 0:
            x = room.max_x * (room_width + space_between)
            z = room.mid_z * (room_width + space_between)
                    
        elif wall_num == 1:
            x = room.mid_x * (room_width + space_between)
            z = room.min_z * (room_width + space_between) 
        elif wall_num == 2:
            x = room.min_x * (room_width * space_between)
            z = room.mid_z * (room_width * space_between)
        elif wall_num == 3:
            x = room.mid_x * (room_width * space_between)
            z = room.max_z * (room_width * space_between)
        return (x, 1, z)

    def step(self, action):
        self.agent_last_pos = self.agent.pos
        self.agent_last_dir = self.agent.dir
        obs, _, done, info = super().step(action)
        reward = 0
        
        if self.episode_num == 2:
            reward += -.01
            if self.agent_last_dir == self.agent.dir and (self.agent.pos == self.agent_last_pos).all():
                reward += -.001     # Agent hits a wall

        if done and self.episode_num == 1:
            self.step_count = 0
            #self.episode1_success = False
            self.episode_num += 1
            self.agent.pos = self.initial_agent_pos
            self.agent.dir = self.initial_agent_dir
            obs = self.render_obs()
            done = False
        if self.near(self.box):
            if self.episode_num == 1:
                self.step_count = 0
                self.agent.pos = self.initial_agent_pos
                self.agent.dir = self.initial_agent_dir
                #self.episode1_success = True
                obs = self.render_obs()
            else:
                reward += 1    # Agent reaches box in second episode
                done = True
            self.episode_num += 1
        print(reward)
        return obs, reward, done, info

    def reset(self):
        self.episode_num = 1
        return super().reset()



class MazeS2(Maze):
    def __init__(self):
        super().__init__(num_rows=2, num_cols=2)

class MazeS3(Maze):
    def __init__(self):
        super().__init__(num_rows=3, num_cols=3)

class MazeS3Fast(Maze):
    def __init__(self, forward_step=0.7, turn_step=45):

        # Parameters for larger movement steps, fast stepping
        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', forward_step)
        params.set('turn_step', turn_step)

        max_steps = 300

        super().__init__(
            num_rows=3,
            num_cols=3,
            params=params,
            max_episode_steps=max_steps,
            domain_rand=False
        )
