from discrete_maze.maze import ExploreTask
from loggy import Log
from schedules import ExploreCreatorSchedule as ECS
import random
import numpy as np
from tqdm import tqdm

# Run this from project root as:
# python -m utils.maze_terminate_test

class TotallyRandomAgent:
    def __init__(self, env):
        self.env = env

    def action(self, obs):
        return self.env.action_space.sample()

class RandomValidAgent:
    def __init__(self, env):
        pass

    def action(self, obs):
        indices = [i - 1 for i in range(1, obs.shape[1]) if obs[-1, i, 0]]
        return random.choice(indices)

class RandomNotVisitedAgent:
    # Just tries to avoid observations it has seen
    def __init__(self, env):
        self.seen = []
        self.fallback = RandomValidAgent(env)

    def action(self, obs):
        nonempty = [i for i in range(1, obs.shape[1]) if obs[-1, i, 0]]
        candidates = [i for i in nonempty if all([not np.array_equal(obs[-1, i, :], vec) for vec in self.seen])]
        if len(candidates) == 0:
            return self.fallback.action(obs)
        choice = random.choice(candidates) - 1
        self.seen.append(obs[-1, choice + 1, :])
        return choice

def test_random_actions(creation_schedule, Chooser, samples, log = None):
    print("Starting schedule at size", creation_schedule.current_size)
    n_truncated = 0
    sample_i = 0
    progress = tqdm(total = samples)
    while True:
        env = creation_schedule.new_env()
        obs = env.reset()
        chooser = Chooser(env)
        done = False
        rewards = []
        while not done:
            sample_i += 1
            progress.update(1)
            if sample_i == samples:
                progress.close()
                return
            choice = chooser.action(obs)
            obs, reward, done, info = env.step(choice)
            rewards.append(reward)
            creation_schedule.update(done, info)
        if log:
            logging_data = {
                'simulation steps': sample_i,
                'average reward': sum(rewards) # just the sum since we are only doing a batch of one
            }
            creation_schedule.add_logging_data(logging_data)
            log.step(logging_data)
        if creation_schedule.env_type_will_change:
            print("Changing schedule to size %d at sample=%d" % (creation_schedule.current_size, sample_i))

# print("\nTesting for totally random actions:")
# test_random_actions(ECS(is_tree = False, place_agent_far_from_dest = False), TotallyRandomAgent, 200000)

# print("\nTesting for random valid actions:")
# test_random_actions(ECS(is_tree = False, place_agent_far_from_dest = False), RandomValidAgent, 200000)

print("Testing for choosing random unseen nodes + placing far from destination:")
log = Log("random-unseen-actions")
test_random_actions(ECS(is_tree = False, place_agent_far_from_dest = True), RandomNotVisitedAgent, 200000, log)
log.close()

print("Testing for random valid actions + placing far from destination:")
log = Log("valid-random-actions")
test_random_actions(ECS(is_tree = False, place_agent_far_from_dest = True), RandomValidAgent, 200000, log)
log.close()

print("Testing for totally random actions + placing far from destination:")
log = Log("totally-random-actions")
test_random_actions(ECS(is_tree = False, place_agent_far_from_dest = True), TotallyRandomAgent, 200000, log)
log.close()

