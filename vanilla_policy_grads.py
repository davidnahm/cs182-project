from discrete_maze.maze import ExploreTask
import models
import loggy
import tensorflow as tf
import gym
import numpy as np
import time

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

class VanillaPolicy:
    def __init__(self, model,
                 env_creator, lr_schedule,
                 min_observations_per_step,
                 log, gamma, fp_observations, render = False, render_mod = 16):
        """
        gamma = our discount factor
        fp_observations = whether observations come as floating point (fp32). If they don't, we cast from int8.
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

        obs_dtype = tf.float32 if fp_observations else tf.int8
        self.obs_placeholder = tf.placeholder(obs_dtype,
                                              shape = [None] + list(dummy_env.observation_space.shape),
                                              name = "observation")
        if fp_observations:
            obs_input = self.obs_placeholder
        else:
            obs_input = tf.cast(self.obs_placeholder, tf.float32)
        self.net_op = model(obs_input,
                            out_size = dummy_env.action_space.n,
                            scope = "mlp")

        self.action_placeholder = tf.placeholder(tf.int32, shape = [None],
                                              name = "action")
        self.adv_placeholder = tf.placeholder(tf.float32, shape = [None], name = "advantage")
        self.lr_placeholder = tf.placeholder(tf.float32, shape = [], name = "learning_rate")
        self.distribution = tf.distributions.Categorical(logits = self.net_op)
        self.sample_op = self.distribution.sample()
        self.logprob_op = self.distribution.log_prob(self.action_placeholder)
        self.policy_value_op = tf.reduce_sum(self.logprob_op * self.adv_placeholder)
        self.update_op = tf.train.AdamOptimizer(self.lr_placeholder).minimize(-self.policy_value_op)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.session = tf.Session(config = tf_config)
        tf.global_variables_initializer().run(session = self.session)

    def sample_trajectory(self):
        env = self.env_creator.new_env()
        observations = []
        rewards = []
        actions = []
        obs = env.reset()

        # This counts how many actions were in directions without nodes
        n_useless_actions = 0
        
        while True:
            if self.render and self.n_episodes % self.render_mod == 0:
                env.render()

            observations.append(obs)
            action = self.sample_op.eval(
                session = self.session,
                feed_dict = {
                    self.obs_placeholder: obs[None]
                }
            )[0]
            actions.append(action)
            obs, reward, done, info = env.step(action)
            self.env_creator.update(done, info)
            rewards.append(reward)

            # if we are doing CartPole-v0, we just ignore this
            if 'correct_direction' in info and not info['correct_direction']:
                n_useless_actions += 1

            if done:
                break
        
        if self.render and self.n_episodes % self.render_mod == 0:
            env.render()
            time.sleep(1) # so that we have time to see the end
            env.close()

        info = {
            'n_useless_actions': n_useless_actions,
            'n_steps': len(observations)
        }

        self.n_episodes += 1
        return observations, rewards, actions, info


    def sample_trajectories(self):
        observations = []
        advantages = []
        actions = []
        reward_totals = []
        info = []
        while len(observations) < self.min_observations_per_step:
            observations_i, rewards_i, actions_i, info_i = self.sample_trajectory()

            # We don't want to mix episodes where the environment is of varying difficulties.
            # Since environments will not frequently increase in difficulty, I think it's 
            # worth the performance hit to just restart with the new environment type.
            if self.env_creator.env_type_will_change:
                print("Environment type changing!")
                return self.sample_trajectories()

            observations.extend(observations_i)
            reward_totals.append(sum(rewards_i))
            actions.extend(actions_i)
            info.append(info_i)

            # cummulative rewards
            cum_reward = 0.0
            advantages_i = []
            for reward in rewards_i[::-1]:
                cum_reward *= self.gamma
                cum_reward += reward
                advantages_i.append(cum_reward)
            advantages.extend(advantages_i[::-1])
        
        observations = np.array(observations)
        advantages = np.array(advantages)
        
        # normalizing advantages
        advantages -= np.mean(advantages)
        advantages /= np.std(advantages)

        actions = np.array(actions)

        return observations, advantages, actions, reward_totals, info
        

    def optimize(self, total_steps):
        steps = 0

        while steps < total_steps:
            observations, advantages, actions, reward_totals, info = self.sample_trajectories()
            self.update_op.run(
                session = self.session,
                feed_dict = {
                    self.obs_placeholder: observations,
                    self.action_placeholder: actions,
                    self.adv_placeholder: advantages,
                    self.lr_placeholder: self.lr_schedule(steps)
                }
            )
            print('average reward:', sum(reward_totals) / len(reward_totals))
            steps += observations.shape[0]

if __name__ == '__main__':
    log = loggy.Log("test")
    vp = VanillaPolicy(
        model = (lambda *args, **varargs: models.mlp(n_layers = 2,
                                                     hidden_size = 64,
                                                     *args, **varargs)),
        env_creator = ExploreCreatorSchedule(is_tree = False, history_size = 2),
        lr_schedule = lambda t: 5e-3,
        min_observations_per_step = 1000,
        log = log,
        gamma = 1.0,
        fp_observations = False,
        render = True,
        render_mod = 128
    )
    vp.optimize(2000000)
    log.close()
