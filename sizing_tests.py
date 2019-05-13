import loggy
import schedules
from ppo import PPO_GAE
from rnn_ppo import RNN_PPO
from snail_ppo import SNAIL_PPO
import models
import tensorflow as tf
import random
import math

def run(Policy, log, params, total_steps):
    tf.reset_default_graph()
    params = log.process_params(params)
    log.add_hyperparams(params)

    policy = Policy(**params)
    policy.initialize_variables()
    policy.optimize(total_steps)
    log.close()

def test_env(env_creator, name, obs_per_step = 250, total_steps = 10000, times = 3):
    for _ in range(times):
        lr = 3 * (10 ** random.uniform(-4, -3))
        base_params = {
            'env_creator': env_creator,
            'min_observations_per_step': obs_per_step,
            'lr_schedule': (lambda t: lr)
        }
        log_dense = loggy.Log("dense-" + name)
        params_dense = {
            'model': (lambda *args, **varargs: models.mlp(*args, **varargs)),
            'value_model': (lambda *args, **varargs: tf.squeeze(models.mlp(out_size = 1,
                                                                    *args, **varargs), axis = 1)),
            'log': log_dense,
            **base_params
        }
        run(PPO_GAE, log_dense, params_dense, total_steps)
        
        log_rnn = loggy.Log("rnn-" + name)
        params_rnn = {
            'log': log_rnn,
            'rnn_stacks': random.choice([1, 2, 3]),
            'hidden_units': random.choice([32, 64, 128]),
            **base_params
        }
        run(RNN_PPO, log_rnn, params_rnn, total_steps)

        log_snail = loggy.Log("snail-" + name)
        params_snail = {
            'log': log_snail,
            'representation_size': random.choice([32, 64]),
            'temporal_span': random.choice([16, 32, 64]),
            **base_params
        }
        run(SNAIL_PPO, log_snail, params_snail, total_steps)

    grapher = loggy.Grapher()
    grapher.add_last("dense-" + name, times)
    grapher.add_last("rnn-" + name, times)
    grapher.add_last("snail-" + name, times)
    grapher.plot('average reward', title = "Average Reward for %s" % name, save = 'figures/%s.png' % name)

def test_perm(k, obs_per_step, total_steps, times = 3):
    env_creator = schedules.ConstantPermSchedule(k, 3 * k * k)
    test_env(env_creator, 'perm-%d' % k, obs_per_step, total_steps, times)

def test_grid_maze(k, obs_per_step, total_steps, times = 3):
    env_creator = schedules.ConstantMazeSchedule('saved_mazes/grid_%s.dill' % k)
    test_env(env_creator, 'grid-%d' % k, obs_per_step, total_steps, times)


if __name__ == '__main__':
    for k in [4, 8, 16, 32]:
        obs_per_step = int(math.sqrt(k) * 300)
        test_grid_maze(k, obs_per_step, obs_per_step * 30)
