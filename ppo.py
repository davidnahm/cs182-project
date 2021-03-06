import tensorflow as tf
import numpy as np
from policy_gradients import VanillaPolicy
from policy_gradients_gae import VanillaPolicyGAE
import models
import schedules
import loggy

class PPO(VanillaPolicy):
    def __init__(self, clip_ratio, max_policy_steps, max_kl,
                 model, env_creator, lr_schedule, min_observations_per_step,
                 log, gamma, render=False, render_mod=16):
        super().__init__(model, env_creator, lr_schedule, min_observations_per_step,
                         log, gamma, render=render, render_mod=render_mod)

        self.clip_ratio = clip_ratio
        self.max_policy_steps = max_policy_steps
        self.max_kl = max_kl

    # From the spinningup implementation
    def _create_objective(self):
        ratio = tf.exp(self.logprob_op - self.logprob_sample_placeholder)
        min_adv = tf.where(self.adv_placeholder > 0,
                           (1 + self.clip_ratio) * self.adv_placeholder,
                           (1 - self.clip_ratio) * self.adv_placeholder)
        return tf.reduce_mean(tf.minimum(ratio * self.adv_placeholder, min_adv))

    def optimize(self, total_steps):
        steps = self.log.get_last('simulation steps', 0)
        while steps < total_steps:
            path = self.sample_trajectories()
            feed = {
                self.obs_placeholder: path['observations'],
                self.action_placeholder: path['actions'],
                self.adv_placeholder: path['advantages'],
                self.logprob_sample_placeholder: path['logprobs'],
                self.lr_placeholder: self.lr_schedule(steps),
            }
            for inner_step in range(self.max_policy_steps):
                _, approximate_entropy, approximate_kl = self.session.run(
                    [self.update_op, self.approx_entropy_op, self.approx_kl_divergence_op],
                    feed_dict = feed
                )

                if approximate_kl > self.max_kl:
                    break
            n_policy_steps = inner_step + 1

            steps += path['observations'].shape[0]

            log_data = {
                'average reward': np.mean(path['reward_totals']),
                'std of reward': np.std(path['reward_totals']),
                'approximate action entropy': approximate_entropy,
                'simulation steps': steps,
                'kl divergence': approximate_kl,
                'inner policy training steps': n_policy_steps,
            }
            self.env_creator.add_logging_data(log_data)

            self.log.step(log_data)
            self.log.print_step()

class PPO_GAE(VanillaPolicyGAE):
    def __init__(self,
                 log,
                 model,
                 value_model,
                 env_creator,
                 clip_ratio = 0.2,
                 max_policy_steps = 80,
                 max_val_steps = 80,
                 max_kl = 0.015,
                 value_lr_schedule = (lambda t: 0.0005),
                 lambda_gae = 0.97,
                 lr_schedule = (lambda t: 3e-4),
                 min_observations_per_step = 1000,
                 gamma = 0.999,
                 render = False,
                 render_mod = 16):
        self.clip_ratio = clip_ratio
        self.max_policy_steps = max_policy_steps
        self.max_val_steps = max_val_steps
        self.max_kl = max_kl
        
        super().__init__(value_model, value_lr_schedule, lambda_gae,
                         model, env_creator, lr_schedule,
                         min_observations_per_step,
                         log, gamma, render=render, render_mod=render_mod)

    # From the spinningup implementation
    def _create_objective(self):
        ratio = tf.exp(self.logprob_op - self.logprob_sample_placeholder)
        min_adv = tf.where(self.adv_placeholder > 0,
                            (1 + self.clip_ratio) * self.adv_placeholder,
                            (1 - self.clip_ratio) * self.adv_placeholder)
        return tf.reduce_mean(tf.minimum(ratio * self.adv_placeholder, min_adv))

    def optimize(self, total_steps):
        steps = self.log.get_last('simulation steps', 0)
        while steps < total_steps:
            path = self.sample_trajectories()
            feed = {
                self.obs_placeholder: path['observations'],
                self.action_placeholder: path['actions'],
                self.adv_placeholder: path['advantages'],
                self.logprob_sample_placeholder: path['logprobs'],
                self.lr_placeholder: self.lr_schedule(steps),
            }
            for inner_step in range(self.max_policy_steps):
                _, approximate_entropy, approximate_kl = self.session.run(
                    [self.update_op, self.approx_entropy_op, self.approx_kl_divergence_op],
                    feed_dict = feed
                )

                if approximate_kl > self.max_kl:
                    break
            n_policy_steps = inner_step + 1

            feed = {
                self.obs_placeholder: path['observations'],
                self.return_placeholder: path['returns'],
                self.value_lr_placeholder: self.value_lr_schedule(steps)
            }
            for inner_step in range(self.max_policy_steps):
                _, value_loss = self.session.run(
                    [self.update_value_op, self.value_loss],
                    feed_dict = feed
                )

            steps += path['observations'].shape[0]

            log_data = {
                'average reward': np.mean(path['reward_totals']),
                'std of reward': np.std(path['reward_totals']),
                'approximate action entropy': approximate_entropy,
                'simulation steps': steps,
                'kl divergence': approximate_kl,
                'inner policy training steps': n_policy_steps,
                'value loss': value_loss,
                'number of episodes': path['number of episodes']
            }
            if 'n_useless_actions' in path['info'][0]:
                total_useless_actions = sum([info['n_useless_actions'] for info in path['info']])
                log_data['proportion useless actions'] = total_useless_actions / float(path['observations'].shape[0])
            self.env_creator.add_logging_data(log_data)

            self.log.step(log_data)
            self.log.print_step()

if __name__ == '__main__':
    # import gym_miniworld
    # log = loggy.Log("miniworld-ppo", autosave_freq = 30.0)
    log = loggy.Log("dense-gerem8", autosave_freq = 15.0, autosave_vars_freq = 60.0, continuing = False)

    params = {
        'clip_ratio': 0.2,
        'max_policy_steps': 80,
        'max_val_steps': 80,
        'max_kl': 0.015,
        'model': (lambda *args, **varargs: models.mlp(*args, **varargs)),
        # 'model': (lambda *args, **varargs: models.mlp(models.miniworld_preprocess(*args, time = False), **varargs)),
        'value_model': (lambda *args, **varargs: tf.squeeze(models.mlp(out_size = 1,
                                                                *args, **varargs), axis = 1)),
        # 'env_creator': schedules.GridMazeSchedule(),
        # 'env_creator': schedules.ExploreCreatorSchedule(is_tree = False, history_size = 1,
        #                                 id_size = 1, reward_type = 'penalty+finished', scale_reward_by_difficulty = False),
        # 'env_creator': schedules.DummyGymSchedule('LunarLander-v2'),
        # 'env_creator': schedules.DummyGymSchedule('MiniWorld-Hallway-v0'),
        'env_creator': schedules.ConstantMazeSchedule('saved_mazes/gerem8.dill'),
        'lr_schedule': (lambda t: 3e-4),
        'min_observations_per_step': 5000,
        'log': log,
        'gamma': 0.999,
        'lambda_gae': .97,
        'value_lr_schedule': (lambda t: 0.0005),
        'render': False,
        'render_mod': 1
    }

    params = log.process_params(params)
    log.add_hyperparams(params)

    vpgae = PPO_GAE(**params)
    vpgae.initialize_variables()
    vpgae.optimize(1000000)
    log.close()
