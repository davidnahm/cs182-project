from discrete_maze.maze import ExploreTask
from schedules import ExploreCreatorSchedule as ECS

jump_ratio = ECS().jump_ratio
size = ECS().current_size
while size < 300:
    difficulties = []
    for _ in range(100):
        env = ExploreTask(size, is_tree = False, id_size = 1, scale_reward_by_difficulty = True)
        env.reset()
        difficulties.append(env.difficulty)
    print("Average difficulty for n=%d is %f" % (size, sum(difficulties) / len(difficulties)))
    size = int(jump_ratio * size)
