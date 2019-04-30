import time
import gym
from discrete_maze.maze import ExploreTask

for size in [3, 5, 10, 20, 50, 100]:
    start = time.time()
    env = ExploreTask(size, is_tree = True)
    env.reset()
    for i in range(10000):
        env.step(env.action_space.sample())
    print("time for 10000 steps for tree maze of size %d: %f" % (size, time.time() - start))
    start = time.time()
    for i in range(100):
        env = ExploreTask(size, is_tree = True)
        env.reset()
    print("time to initialize 100x tree mazes of size %d: %f" % (size, time.time() - start))

    start = time.time()
    env = ExploreTask(size, is_tree = False)
    env.reset()
    for i in range(10000):
        env.step(env.action_space.sample())
    print("time for 10000 steps for delaunay maze of size %d: %f" % (size, time.time() - start))
    start = time.time()
    for i in range(100):
        env = ExploreTask(size, is_tree = False)
        env.reset()
    print("time to initialize 100x delaunay mazes of size %d: %f" % (size, time.time() - start))

    start = time.time()
    for i in range(100):
        env = ExploreTask(size, is_tree = False, place_agent_far_from_dest = False)
        env.reset()
    print("time to initialize 100x delaunay mazes (place_agent_far_from_dest=False) of size %d: %f" % (size, time.time() - start))

for history_size in [1, 2, 4, 8, 16]:
    start = time.time()
    env = ExploreTask(30, is_tree = False, history_size = history_size)
    env.reset()
    for i in range(10000):
        env.step(env.action_space.sample())
    print("time for 10000 steps for delaunay maze with history size %d: %f" % (history_size, time.time() - start))

# Testing OpenAI's implementations

for env_name in ['CartPole-v0', 'Pong-v0']:
    env = gym.make('Pong-v0')
    start = time.time()
    env.reset()
    for _ in range(10000):
        k = env.step(env.action_space.sample()) # take a random action
    print("time taken for 10000 %s steps:" % env_name, time.time() - start)

    start = time.time()
    for i in range(100):
        env.reset()
    print("time taken to reset %s 100x:" % env_name, time.time() - start)