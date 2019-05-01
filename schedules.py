from discrete_maze.maze import ExploreTask
import gym

class CartPoleDummySchedule:
    """
    This class is just meant to be substituted in for ExploreCreatorSchedule
    so that you can check that Policy Gradients works.
    """

    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.env_type_will_change = False

    def update(self, done, info):
        pass

    def new_env(self):
        return self.env

    def add_logging_data(self, data):
        pass

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
                **explore_args):
        """
        Make sure int(jump_ratio * start_size) > start_size or nothing will happen.
        """
        self.current_prop_estimate = 0.0
        self.ratio_completed_trigger = ratio_completed_trigger
        self.jump_ratio = jump_ratio
        self.gamma = gamma
        self.current_size = start_size
        self.explore_args = explore_args
        self.env_type_will_change = False

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
                self.env_type_will_change = True

    def new_env(self):
        self.env_type_will_change = False
        return ExploreTask(self.current_size, **self.explore_args)

    def add_logging_data(self, data):
        data['current maze size'] = self.current_size
        data['current environment confidence'] = self.current_prop_estimate
