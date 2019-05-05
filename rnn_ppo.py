from ppo import PPO_GAE
import tensorflow as tf
import numpy as np
import loggy
import schedules
import models

class RNN_PPO(PPO_GAE):
    def _create_objective(self):
        ratio = tf.exp(self.logprob_op - self.logprob_sample_placeholder)
        min_adv = tf.where(self.adv_placeholder > 0,
                            (1 + self.clip_ratio) * self.adv_placeholder,
                            (1 - self.clip_ratio) * self.adv_placeholder)
        policy_objective = tf.reduce_sum(self.mask_placeholder * tf.minimum(ratio * self.adv_placeholder, min_adv))
        policy_objective /= self.total_timesteps
        # TODO: replace square difference with Huber loss?
        self.value_loss = tf.reduce_sum(self.mask_placeholder * ((self.value_batch - self.return_placeholder) ** 2))
        self.value_loss /= self.total_timesteps
        return policy_objective - self.value_loss * self.value_prop_placeholder

    def _create_policy_and_value(self, x, dummy_env):
        if not self.concat_net:
            x = tf.concat([x, self.obs_input], axis = 2)
        else:
            x = tf.concat([x, self.obs_input, self.concat_net(self.obs_input)], axis = 2)
        x = tf.layers.dense(x, dummy_env.action_space.n, activation = tf.tanh)
        policy = tf.nn.log_softmax(x)
        if self.separate_value_network:
            value = self.separate_value_network(self.obs_input)
        else:
            value = tf.layers.dense(x, 1)
            value = tf.squeeze(value, axis = 2)
        return policy, value

    def _create_rnn_variables(self, dummy_env):
        with tf.variable_scope("scope"):
            rnn_cells = [tf.nn.rnn_cell.LSTMCell(self.hidden_units, state_is_tuple = True)
                            for _ in range(self.rnn_stacks)]
            rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_cells, state_is_tuple = True)

            # we need to create separate operators for batch size of 1 (when actually running)
            # and for the actual full batch (when running gradient descent)
            # This came in handy:
            # https://stackoverflow.com/questions/39112622/how-do-i-set-tensorflow-rnn-state-when-state-is-tuple-true

            # Batch size 1
            # To stop us from passing the hidden state back and forth through a feed_dict, we instead create an operation
            # to assign the hidden state to a variable, which in turn is fed back into the RNN
            self.state_variable = tf.get_variable("hidden_state", [self.rnn_stacks, 2, 1, self.hidden_units],
                                        initializer = tf.keras.initializers.Zeros(dtype = tf.float32),
                                        trainable = False)
            zero_state = tf.zeros([self.rnn_stacks, 2, 1, self.hidden_units], dtype = tf.float32)
            self.reset_state_op = tf.assign(self.state_variable, zero_state)
            sp = tf.unstack(self.state_variable, axis = 0)
            tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(sp[idx][0], sp[idx][1]) for idx in range(self.rnn_stacks)])

            outputs, self.hidden_state_out = tf.nn.dynamic_rnn(rnn_cell, self.obs_input,
                                                        initial_state = tuple_state,
                                                        sequence_length = self.seqlen_placeholder,
                                                        dtype = tf.float32)
            self.update_state_op = tf.assign(self.state_variable, self.hidden_state_out)
            with tf.variable_scope("out_layers"):
                self.policy_1, self.value_1 = self._create_policy_and_value(outputs, dummy_env)
                self.distribution_1 = tf.distributions.Categorical(logits = self.policy_1)
                self.sample_op = self.distribution_1.sample()
                self.logprob_sample_1_op = self.distribution_1.log_prob(self.sample_op)
        
            # Batch size full
            outputs, out_state = tf.nn.dynamic_rnn(rnn_cell, self.obs_input,
                                                sequence_length = self.seqlen_placeholder,
                                                dtype = tf.float32)
            with tf.variable_scope("out_layers", reuse = True):
                self.policy_batch, self.value_batch = self._create_policy_and_value(outputs, dummy_env)
                self.distribution_batch = tf.distributions.Categorical(logits = self.policy_batch)
                self.logprob_op = self.distribution_batch.log_prob(self.action_placeholder, name = "logprob_for_action")

    def __init__(self, env_creator,
                 clip_ratio = 0.2,
                 max_policy_steps = 80,
                 max_kl = 0.015,
                 lambda_gae = 0.97,
                 lr_schedule = (lambda t: 2e-4),
                 value_prop_schedule = (lambda t: 0.01),
                 log = None,
                 gamma = 0.999,
                 min_observations_per_step = 4000,
                 render = False,
                 render_mod = 16,
                 preprocess_op = (lambda x: x),
                 rnn_stacks = 1,
                 hidden_units = 32,
                 separate_value_network = None,
                 concat_net = None):
        self.clip_ratio = clip_ratio
        self.max_policy_steps = max_policy_steps
        self.max_kl = max_kl
        self.value_prop_schedule = value_prop_schedule
        self.lambda_gae = lambda_gae
        self.env_creator = env_creator
        self.lr_schedule = lr_schedule
        self.log = log
        self.gamma = gamma
        self.min_observations_per_step = min_observations_per_step
        self.render = render
        self.render_mod = render_mod
        self.n_episodes = 0 # how many episodes have been simulated to completion
        self.preprocess_op = preprocess_op
        self.rnn_stacks = rnn_stacks
        self.hidden_units = hidden_units
        self.separate_value_network = separate_value_network
        self.concat_net = concat_net

        # We create a throwaway environment for the observation / action shape.
        # This shouldn't be too slow.
        dummy_env = env_creator.new_env()

        fp_observations = dummy_env.observation_space.dtype == np.dtype('float32')
        obs_dtype = tf.float32 if fp_observations else tf.int8
        self.obs_placeholder = tf.placeholder(obs_dtype,
                                              shape = [None, None] + list(dummy_env.observation_space.shape),
                                              name = "observation")
        self.seqlen_placeholder = tf.placeholder(tf.int32, shape = [None], name = "sequence_length_placeholder")
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
        self.return_placeholder = tf.placeholder(tf.float32, shape = [None, None], name = "returns")

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
        # Technically not initializing the path dict but we need to reset the hidden state
        # every time we get a new path dict.
        self.session.run(self.reset_state_op) 
        return super()._initialize_sample_path_dict()

    def _step_once(self, path, env, obs):
        path['observations'].append(obs)
        action, logprob, value, _ = self.session.run(
            [self.sample_op, self.logprob_sample_1_op, self.value_1, self.update_state_op],
            feed_dict = {
                self.obs_placeholder: obs[None, None],
                self.seqlen_placeholder: [1]
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

    def _pad_paths(self, ps, convert = True):
        """
        Takes in array of arrays, and turns them into a numpy array with
        shorter arrays padded with 0s on the right. Transposes for use with
        RNNs
        """
        if convert:
            ps = [np.array(p) for p in ps]
        maxlen = max([p.shape[0] for p in ps])

        # If ps has more than one axis, we will append zeros with shapes matching
        # the axes after the first
        target_shape = list(ps[0].shape[1:])
        ps = [np.concatenate((p, np.zeros([maxlen - p.shape[0]] + target_shape, dtype = ps[0].dtype))) for p in ps]

        return np.stack(ps)

    def _initialize_path_dict(self):
        path = super()._initialize_path_dict()
        path['path lengths'] = []
        path['rewards'] = []
        return path

    # Again the main change is that we don't create one long appended array for everything. Instead
    # we have to factor by batch and time since this is how dynamic_rnn takes its input.
    def sample_trajectories(self):
        path = self._initialize_path_dict()
        total_path_length = 0
        while total_path_length < self.min_observations_per_step:
            path_i = self.sample_trajectory()

            path_length = len(path_i['observations'])
            total_path_length += path_length
            path['path lengths'].append(path_length)
            path['observations'].append(path_i['observations'])
            path['actions'].append(path_i['actions'])
            path['logprobs'].append(path_i['logprobs'])

            path['rewards'].append(path_i['rewards'])
            path['reward_totals'].append(sum(path_i['rewards']))
            path['number of episodes'] += 1
            path['info'].append(path_i['info'])

            # cummulative rewards
            self._calculate_advantages(path, path_i)
        
        # now the environment creator can move on
        if self.env_creator.allow_change():
            # will trigger whenever actually changes
            print("Environment type changing!")

        mask_paths = [np.ones(len(obs_list), dtype = np.float32) for obs_list in path['observations']]
        path['mask'] = self._pad_paths(mask_paths, convert = False)

        path['observations'] = self._pad_paths(path['observations'])
        
        # normalizing advantages
        adv_concat = np.concatenate(path['advantages'])
        path['advantages'] = self._pad_paths(path['advantages'])
        path['advantages'] -= np.mean(adv_concat)
        path['advantages'] /= np.std(adv_concat) + 1e-8

        path['rewards'] = self._pad_paths(path['rewards'])
        path['reward_totals'] = np.array(path['reward_totals'])
        path['actions'] = self._pad_paths(path['actions'])
        path['logprobs'] = self._pad_paths(path['logprobs'])
        path['returns'] = self._pad_paths(path['returns'])

        return path

    def optimize(self, total_steps, early_stop = (lambda _: False)):
        steps = self.log.get_last('simulation steps', 0)
        while steps < total_steps:
            path = self.sample_trajectories()
            feed = {
                self.obs_placeholder: path['observations'],
                self.action_placeholder: path['actions'],
                self.adv_placeholder: path['advantages'],
                self.logprob_sample_placeholder: path['logprobs'],
                self.lr_placeholder: self.lr_schedule(steps),
                self.return_placeholder: path['returns'],
                self.seqlen_placeholder: path['path lengths'],
                self.mask_placeholder: path['mask'],
                self.value_prop_placeholder: self.value_prop_schedule(steps)
            }
            for inner_step in range(self.max_policy_steps):
                _, approximate_entropy, approximate_kl, value_loss = self.session.run(
                    [self.update_op, self.approx_entropy_op, self.approx_kl_divergence_op,
                     self.value_loss],
                    feed_dict = feed
                )

                if approximate_kl > self.max_kl:
                    break
            n_policy_steps = inner_step + 1

            total_path_lengths = np.sum(path['path lengths'])
            steps += total_path_lengths


            if self.log:
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
                    log_data['proportion useless actions'] = total_useless_actions / float(total_path_lengths)
                self.env_creator.add_logging_data(log_data)
                self.log.step(log_data, self.session)
                self.log.print_step()

            if early_stop(self):
                print("Early stop triggered!")
                break

if __name__ == '__main__':
    log = loggy.Log("maze-rnn-ppo", autosave_freq = 10.0)
    def dense_concat_net(*args, **varargs):
        return models.mlp(out_size = 16, output_activation = tf.tanh, scope = "concat_net",
                          flatten = False,
                          *args, **varargs)
    vpgae = RNN_PPO(
        # env_creator = schedules.GridMazeSchedule(),
        env_creator = schedules.ExploreCreatorSchedule(is_tree = False, history_size = 1,
                                        id_size = 1, reward_type = 'penalty+finished', scale_reward_by_difficulty = False),
        # env_creator = schedules.DummyGymSchedule('LunarLander-v2'),
        # env_creator = schedules.DummyGymSchedule('CartPole-v1'),
        lr_schedule = (lambda t: 1e-4),
        value_prop_schedule = (lambda t: 10.0),
        min_observations_per_step = 3000,
        log = log,
        render = False,
        render_mod = 256,
        rnn_stacks = 2,
        separate_value_network = (lambda *args, **varargs:
            tf.squeeze(models.mlp(scope = "value_network", out_size = 1, flatten = False, *args, **varargs), axis = 2)),
        concat_net = dense_concat_net
    )
    vpgae.initialize_variables()
    vpgae.optimize(200000)
    log.close(vpgae.session)
