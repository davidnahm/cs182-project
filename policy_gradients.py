import models
import loggy
import tensorflow as tf
import gym
import numpy as np
import time
import schedules
import argparse

class VanillaPolicy:
    def __init__(self, model,
                 env_creator, lr_schedule,
                 min_observations_per_step,
                 log, gamma, render = False, render_mod = 16):
        """
        gamma = our discount factor
        """
        self.env_creator = env_creator
        self.lr_schedule = lr_schedule
        self.min_observations_per_step = min_observations_per_step
        self.log = log
        self.gamma = gamma

        self.render = render
        self.render_mod = render_mod
        self.n_episodes = 0 # how many episodes have been simulated to completion

        # We create a throwaway environment for the observation / action shape.
        # This shouldn't be too slow.
        dummy_env = env_creator.new_env()

        fp_observations = not np.issubdtype(dummy_env.observation_space.dtype, np.floating)
        obs_dtype = tf.float32 if fp_observations else tf.int8
        self.obs_placeholder = tf.placeholder(obs_dtype,
                                              shape = [None] + list(dummy_env.observation_space.shape),
                                              name = "observation")
        if fp_observations:
            self.obs_input = self.obs_placeholder
        else:
            self.obs_input = tf.cast(self.obs_placeholder, tf.float32)
        self.net_op = model(self.obs_input,
                            out_size = dummy_env.action_space.n,
                            scope = "policy_net")

        self.action_placeholder = tf.placeholder(tf.int32, shape = [None],
                                              name = "action")
        self.adv_placeholder = tf.placeholder(tf.float32, shape = [None], name = "advantage")
        self.lr_placeholder = tf.placeholder(tf.float32, shape = [], name = "learning_rate")
        self.distribution = tf.distributions.Categorical(logits = self.net_op, name = "action_distribution")
        self.sample_op = self.distribution.sample(name = "sample_action")
        self.logprob_op = self.distribution.log_prob(self.action_placeholder, name = "logprob_for_action")

        # Idea for logging approximate KL divergence and Entropy comes from spinningup RL
        # In Vanilla Policy Gradients, KL divergence will always be 0 unless we iterate multiple times
        # on the same actions, which would not make much sense.
        self.logprob_sample_op = self.distribution.log_prob(self.sample_op, name = "logprob_sample")
        self.logprob_sample_placeholder = tf.placeholder(tf.float32, shape = [None], name = "logprob_sample_placeholder")
        self.approx_entropy_op = -tf.reduce_mean(self.logprob_op, name = "approximate_entropy")
        self.approx_kl_divergence_op = 0.5 * tf.reduce_mean(tf.square(self.logprob_sample_placeholder - self.logprob_op))

        self.policy_value_op = self._create_objective()
        self.update_op = tf.train.AdamOptimizer(self.lr_placeholder).minimize(-self.policy_value_op)

    def _create_objective(self):
        return tf.reduce_mean(self.logprob_op * self.adv_placeholder, name = "policy_value")

    def initialize_variables(self):
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.session = tf.Session(config = tf_config)
        tf.global_variables_initializer().run(session = self.session)


    def _initialize_sample_path_dict(self):
        return {
            'observations': [],
            'rewards': [],
            'actions': [],
            'logprobs': [],
            'info': {}
        }

    def _step_once(self, path, env, obs):
        path['observations'].append(obs)
        action, logprob = self.session.run(
            [self.sample_op, self.logprob_sample_op],
            feed_dict = {
                self.obs_placeholder: obs[None]
            }
        )
        action = action[0]
        path['actions'].append(action)
        path['logprobs'].append(logprob[0])

        obs, reward, done, info = env.step(action)
        self.env_creator.update(done, info)
        path['rewards'].append(reward)

        return obs, reward, done, info

    def sample_trajectory(self):
        env = self.env_creator.new_env()
        path = self._initialize_sample_path_dict()
        obs = env.reset()

        # This counts how many actions were in directions without nodes
        n_useless_actions = 0
        
        while True:
            if self.render and self.n_episodes % self.render_mod == 0:
                env.render()

            obs, reward, done, info = self._step_once(path, env, obs)

            # if we are doing CartPole-v0, we just ignore this
            if 'correct_direction' in info and not info['correct_direction']:
                n_useless_actions += 1

            if done:
                break
        
        if self.render and self.n_episodes % self.render_mod == 0:
            env.render()
            time.sleep(1) # so that we have time to see the end
            env.close()

        path['info']['n_useless_actions'] = n_useless_actions,
        path['info']['n_steps'] =  len(path['observations'])

        self.n_episodes += 1
        return path

    def _initialize_path_dict(self):
        return {
            'observations': [],
            'advantages': [],
            'actions': [],
            'reward_totals': [],
            'logprobs': [],
            'info': []
        }

    def _calculate_discounted_to_go(self, values, coef):
        accumulator = 0.0
        discounted = []
        for value in values[::-1]:
            accumulator *= coef
            accumulator += value
            discounted.append(accumulator)
        return discounted[::-1]

    def _calculate_advantages(self, path, path_i):
        path['advantages'].extend(self._calculate_discounted_to_go(path_i['rewards'], self.gamma))

    def sample_trajectories(self):
        path = self._initialize_path_dict()
        while len(path['observations']) < self.min_observations_per_step:
            path_i = self.sample_trajectory()

            path['observations'].extend(path_i['observations'])
            path['actions'].extend(path_i['actions'])
            path['logprobs'].extend(path_i['logprobs'])

            path['reward_totals'].append(sum(path_i['rewards']))
            path['info'].append(path_i['info'])

            # cummulative rewards
            self._calculate_advantages(path, path_i)
        
        # now the environment creator can move on
        if self.env_creator.allow_change():
            # will trigger whenever actually changes
            print("Environment type changing!")

        path['observations'] = np.array(path['observations'])
        path['advantages'] = np.array(path['advantages'])
        
        # normalizing advantages
        path['advantages'] -= np.mean(path['advantages'])
        path['advantages'] /= np.std(path['advantages'])

        path['reward_totals'] = np.array(path['reward_totals'])
        path['actions'] = np.array(path['actions'])

        return path
        

    def optimize(self, total_steps):
        steps = 0

        while steps < total_steps:
            path = self.sample_trajectories()
            _, approximate_entropy = self.session.run(
                [self.update_op, self.approx_entropy_op],
                feed_dict = {
                    self.obs_placeholder: path['observations'],
                    self.action_placeholder: path['actions'],
                    self.adv_placeholder: path['advantages'],
                    self.lr_placeholder: self.lr_schedule(steps)
                }
            )
            steps += path['observations'].shape[0]

            log_data = {
                'average reward': np.mean(path['reward_totals']),
                'std of reward': np.std(path['reward_totals']),
                'approximate action entropy': approximate_entropy,
                'simulation steps': steps
            }
            self.env_creator.add_logging_data(log_data)

            self.log.step(log_data)
            self.log.print_step()


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description = "Train a network to navigate a discrete maze.")
    argument_parser.add_argument("--history-size", default = 1, type = int,
                help = "Number of previous frames to give the network.")
    options = argument_parser.parse_args()


    log = loggy.Log("acrobot-vanilla")
    vp = VanillaPolicy(
        model = (lambda *args, **varargs: models.mlp(*args, **varargs)),
        # env_creator = schedules.ExploreCreatorSchedule(is_tree = False, history_size = options.history_size),
        env_creator = schedules.DummyGymSchedule('Acrobot-v1'),
        lr_schedule = lambda t: 1e-4,
        min_observations_per_step = 5000,
        log = log,
        gamma = 0.99,
        render = False,
        render_mod = 16
    )
    vp.initialize_variables()
    vp.optimize(500000)
    log.close()
