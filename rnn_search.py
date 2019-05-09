import multiprocessing

def run_with_random_hyperparameters(_):
    # Loading within the process to save time
    from rnn_ppo import RNN_PPO
    import tensorflow as tf
    import loggy
    import random
    import schedules
    import models
    import discrete_maze.maze

    tf.reset_default_graph()
    log = loggy.Log("maze-hyperparam-search", autosave_freq = 15.0, autosave_vars_freq = 180.0, continuing = False)

    lr = 10 ** random.uniform(-5.5, -2.5)
    value_prop = 10 ** random.uniform(-1.5, 1.5)

    if random.random() > 0.8:
        separate_value_network = (lambda *args, **varargs:
            tf.squeeze(models.mlp(scope = "value_network", out_size = 1, hiddens = [64, 64],
                                  flatten = False, *args, **varargs), axis = 2))
    else:
        separate_value_network = None

    history_size = 1 if random.random() > 0.15 else random.randint(2, 5)
    id_size = 1 if random.random() > 0.15 else random.randint(2, 8)
    reward_type = random.choice(discrete_maze.maze.ExploreTask.reward_types)
    scale_reward_by_difficulty = random.random() > 0.5
    place_agent_far_from_dest = random.random() > 0.2
    agent_placement_prop = random.uniform(0.2, 0.9)
    time_penalty = 10 ** random.uniform(-2.3, -.8)
    invalid_move_penalty = 10 ** random.uniform(-1, 0.5)


    def dense_concat_net(*args, **varargs):
        return models.mlp(out_size = 16, output_activation = tf.tanh, scope = "concat_net",
                          flatten = False,
                          *args, **varargs)
    concat_net = dense_concat_net if random.random() > 0.5 else None

    params = {
        # 'env_creator': schedules.GridMazeSchedule(),
        'env_creator': schedules.ExploreCreatorSchedule(is_tree = False, history_size = history_size,
                                        id_size = id_size, reward_type = reward_type,
                                        scale_reward_by_difficulty = scale_reward_by_difficulty,
                                        place_agent_far_from_dest = place_agent_far_from_dest,
                                        agent_placement_prop = agent_placement_prop,
                                        time_penalty = time_penalty,
                                        invalid_move_penalty = invalid_move_penalty),
        'clip_ratio': random.uniform(0.18, 0.22), # this seems to be set well
        'max_policy_steps': random.randint(50, 100),
        'max_kl': random.uniform(0.01, 0.02),
        'lambda_gae': random.uniform(0.95, 1.0),
        'lr_schedule': (lambda t: lr),
        'value_prop_schedule': (lambda t: value_prop),
        'log': log,
        'gamma': random.uniform(0.95, 1.0),
        'min_observations_per_step': 4000,
        'render': False,
        'rnn_stacks': random.randint(1, 3),
        'hidden_units': 2 ** random.randint(4, 8),
        'separate_value_network': separate_value_network,
        'concat_net': concat_net
    }

    params = log.process_params(params)
    log.add_hyperparams(params)

    print("Running with parameters:", params)

    ppo = RNN_PPO(**params)
    ppo.initialize_variables()

    def early_stop(policy):
        maze_size = policy.log.get_last('current maze size', 4)
        steps = policy.log.get_last('simulation steps', 0)
        return maze_size == 4 and steps >= 50000

    ppo.optimize(500000, early_stop = early_stop)
    log.close()

if __name__ == '__main__':
    with multiprocessing.Pool(processes = 5) as pool:
        pool.map(run_with_random_hyperparameters, range(30))
