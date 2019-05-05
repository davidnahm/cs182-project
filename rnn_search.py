import multiprocessing

def run_with_random_hyperparameters(_):
    # Loading within the process to save time
    from rnn_ppo import RNN_PPO
    import tensorflow as tf
    import loggy
    import random
    import schedules
    import models

    log = loggy.Log("maze-hyperparam-search", autosave_freq = 15.0, autosave_vars_freq = 180.0, continuing = False)

    lr = 10 ** random.uniform(-5.5, -2.5)
    value_prop = 10 ** random.uniform(-1.5, 1.5)

    if random.random() > 0.8:
        separate_value_network = (lambda *args, **varargs:
            tf.squeeze(models.mlp(scope = "value_network", out_size = 1, hiddens = [64, 64],
                                  flatten = False, *args, **varargs), axis = 2))
    else:
        separate_value_network = None

    params = {
        # 'env_creator': schedules.GridMazeSchedule(),
        'env_creator': schedules.ExploreCreatorSchedule(is_tree = False, history_size = 1,
                                        id_size = 1, reward_type = 'penalty+finished', scale_reward_by_difficulty = False),
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
        'separate_value_network': separate_value_network
    }

    params = log.process_params(params)
    log.add_hyperparams(params)

    print("Running with parameters:", params)

    ppo = RNN_PPO(**params)
    ppo.initialize_variables()
    ppo.optimize(500000)
    log.close(ppo.session)

if __name__ == '__main__':
    with multiprocessing.Pool(processes = 5) as pool:
        pool.map(run_with_random_hyperparameters, range(30))
