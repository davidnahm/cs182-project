from discrete_maze.maze import ExploreTask

et = ExploreTask(10)
et.reset()
observation, reward, done, info = et.step(et.action_space.sample())
print(observation)

class ExploreCreatorSchedule:
    """
    This class will take in values from environment.step
    and determine how large the next generated environment should be.
    Will scale up environments when enough runs are finished without
    timing out.
    """
    def __init__(self,
                ratio_completed_trigger = 0.75,
                jump_ratio = 1.3,
                gamma = 0.975,
                start_size = 4,
                is_tree = False):
        """
        Make sure int(jump_ratio * start_size) > start_size or nothing will happen.
        """
        self.current_prop_estimate = 0.0
        self.ratio_completed_trigger = ratio_completed_trigger
        self.jump_ratio = jump_ratio
        self.gamma = gamma
        self.current_size = start_size
        self.is_tree = is_tree

    def update(self, done, info):
        """
        Pass the "done" and "info" values returned from
        environment.step. 
        """
        if done:
            self.current_prop_estimate *= self.gamma
            if not info['truncated']:
                # we finished the run without truncating; this
                # counts as a success
                self.current_prop_estimate += 1 - self.gamma
            if self.current_prop_estimate > self.ratio_completed_trigger:
                self.current_prop_estimate = 0.0
                self.current_size = int(self.current_size * self.jump_ratio)

    def new_env(self):
        return ExploreTask(self.current_size, self.is_tree)

class VanillaPolicy:
    def __init__(model, env_creator):
        # create operations for
        # [ ] sampling action
        # [ ] calculate loss
        # [ ] optimize
        # create function to create
        # create function to get advantage given trajectories

        pass