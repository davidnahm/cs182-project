from ppo import PPO_GAE
import tensorflow as tf
import numpy as np
import loggy
import schedules

class RNN_PPO(PPO_GAE):
    def _create_objective(self):
        ratio = tf.exp(self.logprob_op - self.logprob_sample_placeholder)
        min_adv = tf.where(self.adv_placeholder > 0,
                            (1 + self.clip_ratio) * self.adv_placeholder,
                            (1 - self.clip_ratio) * self.adv_placeholder)
        policy_objective = tf.reduce_sum(self.mask_placeholder * tf.minimum(ratio * self.adv_placeholder, min_adv))
        policy_objective /= self.total_timesteps
        # TODO: replace square difference with Huber loss?
        self.value_loss = tf.reduce_sum(self.mask_placeholder * ((self.value_net_op - self.return_placeholder) ** 2))
        self.value_loss /= self.total_timesteps
        return policy_objective - self.value_loss * self.value_prop_placeholder

    def _create_rnn_variables(self, dummy_env):
        with tf.variable_scope("scope"):
            rnn_cells = [tf.nn.rnn_cell.LSTMCell(self.hidden_units, state_is_tuple = True)
                            for _ in range(self.rnn_stacks)]
            rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_cells, state_is_tuple = True)

            # we need to create separate operators for batch size of 1 (when actually running)
            # and for the actual batch size (when running gradient descent)
            # This came in handy:
            # https://stackoverflow.com/questions/39112622/how-do-i-set-tensorflow-rnn-state-when-state-is-tuple-true

            # Batch size 1

            self.state_placeholder = tf.placeholder(tf.float32, [self.rnn_stacks, 2, 1, self.hidden_units])
            self.zero_state_array = np.zeros((self.rnn_stacks, 2, 1, self.hidden_units))
            sp = tf.unstack(self.state_placeholder, axis = 0)
            tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(sp[idx][0], sp[idx][1]) for idx in range(self.rnn_stacks)])

            outputs, self.hidden_state_out = tf.nn.dynamic_rnn(rnn_cell, self.obs_input,
                                                        initial_state = tuple_state,
                                                        sequence_length = self.seqlen_placeholder,
                                                        dtype = tf.float32)
            with tf.variable_scope("out_layers"):
                x = tf.layers.dense(outputs, dummy_env.action_space.n, activation = tf.tanh)
                self.policy_1 = tf.nn.log_softmax(x)
                value = tf.layers.dense(outputs, 1)
                self.value_1 = tf.squeeze(value, axis = 2)
                self.distribution_1 = tf.distributions.Categorical(logits = self.policy_1)
                self.sample_op = self.distribution.sample()
                self.logprob_sample_1_op = self.distribution.log_prob(self.sample_1_op)
        
        # Batch size full
        zero_train_batch = rnn_cell.zero_state(self.batch_size, dtype = tf.float32)

        outputs, out_state = tf.nn.dynamic_rnn(rnn_cell, self.obs_input,
                                               initial_state = tuple_state,
                                               sequence_length = self.seqlen_placeholder,
                                               dtype = tf.float32)
        with tf.variable_scope("out_layers", reuse = True):
            x = tf.layers.dense(outputs, dummy_env.action_space.n, activation = tf.tanh)
            self.policy_batch = tf.nn.log_softmax(x)
            value = tf.layers.dense(outputs, 1)
            self.value_batch = tf.squeeze(value, axis = 2)
            self.distribution_batch = tf.distributions.Categorical(logits = self.policy_batch)
            self.logprob_op = self.distribution_batch.log_prob(self.action_placeholder, name = "logprob_for_action")

    def __init__(self, clip_ratio, max_policy_steps, max_kl,
                 lambda_gae, env_creator, lr_schedule, value_prop_schedule,
                 log, gamma, batch_size=256, render=False, render_mod=16,
                 preprocess_op = (lambda x: x), rnn_stacks = 1, hidden_units = 32):
        self.clip_ratio = clip_ratio
        self.max_policy_steps = max_policy_steps
        self.max_kl = max_kl
        self.value_prop_schedule = value_prop_schedule
        self.lambda_gae = lambda_gae
        self.env_creator = env_creator
        self.lr_schedule = lr_schedule
        self.log = log
        self.gamma = gamma
        self.batch_size = batch_size
        self.render = render
        self.render_mod = render_mod
        self.n_episodes = 0 # how many episodes have been simulated to completion
        self.preprocess_op = preprocess_op
        self.rnn_stacks = rnn_stacks
        self.hidden_units = hidden_units

        # We create a throwaway environment for the observation / action shape.
        # This shouldn't be too slow.
        dummy_env = env_creator.new_env()

        fp_observations = not np.issubdtype(dummy_env.observation_space.dtype, np.floating)
        obs_dtype = tf.float32 if fp_observations else tf.int8
        self.obs_placeholder = tf.placeholder(obs_dtype,
                                              shape = [None, None] + list(dummy_env.observation_space.shape),
                                              name = "observation")
        self.seqlen_placeholder = tf.placeholder(tf.int32, shape = [None])
        if fp_observations:
            self.obs_input = self.obs_placeholder
        else:
            self.obs_input = tf.cast(self.obs_placeholder, tf.float32)
        self.action_placeholder = tf.placeholder(tf.int32, shape = [None, None],
                                              name = "action")
        self.adv_placeholder = tf.placeholder(tf.float32, shape = [None, None], name = "advantage")
        self.lr_placeholder = tf.placeholder(tf.float32, shape = [], name = "learning_rate")
        self.value_prop_placeholder = tf.placeholder(tf.float32, shape = [], name = "value_prop")
        self.logprob_sample_placeholder = tf.placeholder(tf.float32, shape = [None, None], name = "logprob_sample_placeholder")
        self.return_placeholder = tf.placeholder(tf.float32, shape = [None], name = "returns")

        # Helps us tell when sequences end
        self.mask_placeholder = tf.placeholder(tf.float32, shape = [None, None], name = "mask")
        self.total_timesteps = tf.reduce_sum(self.mask_placeholder)

        self._create_rnn_variables(dummy_env)
        with tf.variable_scope("approximate_entropy"):
            self.approx_entropy_op = -tf.reduce_sum(self.logprob_op * self.mask_placeholder, name = "approximate_entropy")
            self.approx_entropy_op /= self.total_timesteps

        with tf.variable_scope("approximate_kl"):
            sqdif = tf.square(self.logprob_sample_placeholder - self.logprob_op)
            self.approx_kl_divergence_op = 0.5 * tf.reduce_sum(sqdif * self.mask_placeholder)
            self.approx_kl_divergence_op /= self.total_timesteps

        self.value_op = self._create_objective()
        self.update_op = tf.train.AdamOptimizer(self.lr_placeholder).minimize(-self.value_op)

    def _initialize_sample_path_dict(self):
        path = super()._initialize_sample_path_dict()
        path['last_hidden'] = np.zeros([self.rnn_stacks, 2, self.batch_size, self.hidden_units])
        return path

    def _step_once(self, path, env, obs):
        path['observations'].append(obs)
        action, logprob, value, hidden_state = self.session.run(
            [self.sample_op, self.logprob_sample_1_op, self.value_1, self.hidden_state_out],
            feed_dict = {
                self.obs_placeholder: obs[None, None],
                self.state_placeholder: path['last_hidden']
            }
        )
        action = action[0][0]
        path['actions'].append(action)
        path['logprobs'].append(logprob[0][0])
        path['values'].append(value[0][0])

        obs, reward, done, info = env.step(action)
        self.env_creator.update(done, info)
        path['rewards'].append(reward)

        return obs, reward, done, info

    # only difference from parent class is that we append to the path lists
    def _calculate_advantages(self, path, path_i):
        path['returns'].append(self._calculate_discounted_to_go(path_i['rewards'], self.gamma))

        # This is the GAE calculation, slightly modified from spinningup
        shifted_values = np.append(path_i['values'], 0)[1:]
        deltas = path_i['rewards'] + self.gamma * shifted_values - path_i['values']
        path['advantages'].append(self._calculate_discounted_to_go(deltas, self.gamma * self.lambda_gae))

    # Again the main change is that we don't create one long appended array for everything. Instead
    # we have to factor by batch and time since this is how dynamic_rnn takes its input.
    def sample_trajectories(self):
        path = self._initialize_path_dict()
        while len(path['observations']) < self.batch_size:
            path_i = self.sample_trajectory()

            path['observations'].append(path_i['observations'])
            path['actions'].append(path_i['actions'])
            path['logprobs'].append(path_i['logprobs'])

            path['reward_totals'].append(sum(path_i['rewards']))
            path['number of episodes'] += 1
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
            self.env_creator.add_logging_data(log_data)

            self.log.step(log_data)
            self.log.print_step()

if __name__ == '__main__':
    log = loggy.Log("rnn_ppo", autosave_freq = 10.0)
    vpgae = RNN_PPO(
        clip_ratio = 0.2,
        max_policy_steps = 80,
        max_kl = 0.015,
        env_creator = schedules.GridMazeSchedule(),
        # env_creator = schedules.ExploreCreatorSchedule(is_tree = False, history_size = 1,
        #                                 id_size = 1, reward_type = 'penalty+finished', scale_reward_by_difficulty = False),
        # env_creator = schedules.DummyGymSchedule('LunarLander-v2'),
        lr_schedule = (lambda t: 2e-4),
        log = log,
        gamma = 0.999,
        lambda_gae = .97,
        value_prop_schedule = (lambda t: 1e-2),
        render = False,
        render_mod = 256
    )
    vpgae.initialize_variables()
    vpgae.optimize(200000)
    log.close()