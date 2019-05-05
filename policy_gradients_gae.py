from policy_gradients import VanillaPolicy
import models
import tensorflow as tf
import numpy as np
import loggy
import schedules

# Implements GAE-Lambda with value estimator. This is based off of
# spinningup's implementation
class VanillaPolicyGAE(VanillaPolicy):
    def __init__(self, value_model, value_lr_schedule, lambda_gae,
                 model, env_creator, lr_schedule,
                 min_observations_per_step,
                 log, gamma, render = False, render_mod = 16):
        super().__init__(model, env_creator, lr_schedule,
                         min_observations_per_step,
                         log, gamma, render, render_mod)
        self.value_lr_schedule = value_lr_schedule
        self.lambda_gae = lambda_gae

        self.value_net_op = value_model(self.obs_input, scope = "value_net")
        self.return_placeholder = tf.placeholder(tf.float32, shape = [None], name = "returns")
        self.value_lr_placeholder = tf.placeholder(tf.float32, shape = [], name = "value_learning_rate")

        # TODO: replace square difference with Huber loss?
        self.value_loss = tf.reduce_mean((self.value_net_op - self.return_placeholder) ** 2)
        self.update_value_op = tf.train.AdamOptimizer(self.value_lr_placeholder).minimize(self.value_loss)

    def _initialize_sample_path_dict(self):
        path = super()._initialize_sample_path_dict()
        path['values'] = []
        return path

    def _step_once(self, path, env, obs):
        path['observations'].append(obs)
        action, logprob, value = self.session.run(
            [self.sample_op, self.logprob_sample_op, self.value_net_op],
            feed_dict = {
                self.obs_placeholder: obs[None]
            }
        )
        action = action[0]
        path['actions'].append(action)
        path['logprobs'].append(logprob[0])
        path['values'].append(value[0])

        obs, reward, done, info = env.step(action)
        self.env_creator.update(done, info)
        path['rewards'].append(reward)

        return obs, reward, done, info

    def _initialize_path_dict(self):
        path = super()._initialize_path_dict()
        path['returns'] = []
        return path

    def _calculate_advantages(self, path, path_i):
        path['returns'].extend(self._calculate_discounted_to_go(path_i['rewards'], self.gamma))

        # This is the GAE calculation, slightly modified from spinningup
        shifted_values = np.append(path_i['values'], 0)[1:]
        deltas = path_i['rewards'] + self.gamma * shifted_values - path_i['values']
        path['advantages'].extend(self._calculate_discounted_to_go(deltas, self.gamma * self.lambda_gae))

    def optimize(self, total_steps):
        steps = self.log.get_last('simulation steps', 0)

        while steps < total_steps:
            path = self.sample_trajectories()
            _, _, value_loss, approximate_entropy = self.session.run(
                [self.update_op, self.update_value_op, self.value_loss, self.approx_entropy_op],
                feed_dict = {
                    self.obs_placeholder: path['observations'],
                    self.action_placeholder: path['actions'],
                    self.adv_placeholder: path['advantages'],
                    self.return_placeholder: path['returns'],
                    self.lr_placeholder: self.lr_schedule(steps),
                    self.value_lr_placeholder: self.value_lr_schedule(steps)
                }
            )
            steps += path['observations'].shape[0]

            log_data = {
                'average reward': np.mean(path['reward_totals']),
                'std of reward': np.std(path['reward_totals']),
                'approximate action entropy': approximate_entropy,
                'simulation steps': steps,
                'value loss': value_loss
            }
            self.env_creator.add_logging_data(log_data)

            self.log.step(log_data, self.session)
            self.log.print_step()


if __name__ == '__main__':
    log = loggy.Log("maze-h1-pggae")
    vpgae = VanillaPolicyGAE(
        model = (lambda *args, **varargs: models.mlp(*args, **varargs)),
        value_model = (lambda *args, **varargs: tf.squeeze(models.mlp(out_size = 1,
                                                                *args, **varargs), axis = 1)),
        env_creator = schedules.ExploreCreatorSchedule(is_tree = False, history_size = 1,
                                        id_size = 1, reward_type = 'penalty+finished', scale_reward_by_difficulty = False),
        # env_creator = schedules.DummyGymSchedule('Acrobot-v1'),
        lr_schedule = (lambda t: 2e-4),
        value_lr_schedule = (lambda t: 2.4e-3),
        lambda_gae = .97,
        min_observations_per_step = 4000,
        log = log,
        gamma = 0.999,
        render = False,
        render_mod = 128
    )
    vpgae.initialize_variables()
    vpgae.optimize(200000)
    log.close(vpgae.session)
