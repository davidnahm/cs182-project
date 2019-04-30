from discrete_maze.maze import ExploreTask
from schedules import ExploreCreatorSchedule as ECS
import random
from tqdm import tqdm

# Run this from project root as:
# python -m utils.maze_terminate_test

def random_action(obs, env):
    return env.action_space.sample()

def random_valid_action(obs, env):
    indices = [i - 1 for i in range(1, obs.shape[1]) if obs[-1, i, 0]]
    return random.choice(indices)

def test_random_actions(creation_schedule, choice_func, samples):
    print("Starting schedule at size", creation_schedule.current_size)
    n_truncated = 0
    sample_i = 0
    progress = tqdm(total = samples)
    while True:
        env = creation_schedule.new_env()
        obs = env.reset()
        done = False
        while not done:
            sample_i += 1
            progress.update(1)
            if sample_i == samples:
                progress.close()
                return
            choice = choice_func(obs, env)
            obs, _, done, info = env.step(choice)
            creation_schedule.update(done, info)
        if creation_schedule.env_type_will_change:
            print("Changing schedule to size %d at sample=%d" % (creation_schedule.current_size, sample_i))

print("\nTesting for totally random actions:")
test_random_actions(ECS(is_tree = False, place_agent_far_from_dest = False), random_action, 1000000)

print("\nTesting for random valid actions:")
test_random_actions(ECS(is_tree = False, place_agent_far_from_dest = False), random_valid_action, 1000000)

print("Testing for totally random actions + placing far from destination:")
test_random_actions(ECS(is_tree = False, place_agent_far_from_dest = True), random_valid_action, 1000000)
