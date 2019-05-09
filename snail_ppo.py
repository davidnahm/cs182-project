from rnn_ppo import RNN_PPO
from ppo import PPO_GAE
import tensorflow as tf
import numpy as np
import schedules
import loggy

# from the following paper:
# https://arxiv.org/pdf/1707.03141.pdf
class SNAIL_PPO(RNN_PPO):

    def _dense_causal_block(self, x, dilation_r, filters_d, scope):
        with tf.variable_scope(scope):
            in_d = x.shape[2]
            filter_2 = tf.ones([1, 2, in_d, filters_d], name = "filter_mask_left")
            filter_1 = tf.zeros([1, 1, in_d, filters_d], name = "filter_mask_right")
            filter_mask = tf.concat([filter_2, filter_1], axis = 1)
            filter_filters = tf.get_variable("filter_filters", [1, 3, in_d, filters_d], trainable = True)
            filter_filters *= filter_mask
            gate_filters = tf.get_variable("gate_filters", [1, 3, in_d, filters_d], trainable = True)
            gate_filters *= filter_mask

            x_expanded = tf.expand_dims(x, axis = 1)
            x_filter = tf.nn.conv2d(x_expanded, filter_filters, strides = [1, 1, 1, 1], padding = 'SAME',
                                    dilations = [1, 1, dilation_r, 1])
            x_gate = tf.nn.conv2d(x_expanded, gate_filters, strides = [1, 1, 1, 1], padding = 'SAME',
                                    dilations = [1, 1, dilation_r, 1])
            activations = tf.tanh(x_filter) * tf.sigmoid(x_gate)
            activations = tf.squeeze(activations, axis = 1)
        return tf.concat([activations, x], axis = 2)

    def _tc_block(self, x, desired_len, filters_d, scope):
        cur_dilation = 1
        with tf.variable_scope(scope):
            while cur_dilation < desired_len:
                inner_scope = 'dilation_%d' % cur_dilation
                x = self._dense_causal_block(x, cur_dilation, filters_d, inner_scope)
                cur_dilation *= 2
        return x

    def _attention_block(self, x, key_size, value_size, scope):
        with tf.variable_scope(scope):
            keys = tf.layers.dense(x, key_size, name = "keys_affine")
            query = tf.layers.dense(x, key_size, name = "query_affine")
            similarities = tf.linalg.matmul(query, tf.transpose(keys, (0, 2, 1)), name = "similarities") / (key_size ** 0.5)
            probabilities = tf.nn.softmax(similarities, name = "probabilities")

            n_fills = tf.cast((tf.shape(x)[1] + 1) * tf.shape(x)[1] / 2, dtype = tf.int32)
            mask_ones = tf.ones(n_fills, tf.float32)
            mask = tf.contrib.distributions.fill_triangular(mask_ones, upper = False, name = "fill_mask")
            mask = tf.expand_dims(mask, axis = 0, name = "expand_mask")
            probabilities *= mask

            # we perform normalization twice this way, but I believe that softmax in
            # tensorflow is more stable than just taking exponents and normalizing
            probabilities /= tf.expand_dims(tf.reduce_sum(probabilities, axis = 2), axis = 2)

            values = tf.layers.dense(x, value_size, name = "value_affine")
            read = tf.linalg.matmul(probabilities, values, name = "read_values")
        return tf.concat([read, x], axis = 2)
    
    def _create_snail_vars(self, dummy_env):
        T = 32
        # TODO: need to align
        # actions_ohot = tf.one_hot(self.action_placeholder, depth = dummy_env.action_space.n)
        # rewards = tf.expand_dims(self.reward_placeholder, axis = 2)
        # combined_input = tf.concat([actions_ohot, rewards, self.obs_input], axis = 2)
        x = self._attention_block(self.preprocess_op(self.obs_input), 16, 16, "attention_1")
        x = self._tc_block(x, self.temporal_span, 16, "temporal_conv_1")
        x = self._tc_block(x, self.temporal_span, 16, "temporal_conv_2")
        x = self._attention_block(x, 32, 32, "attention_2")
        
        # dummy_env = self.env_creator.new_env()
        policy = tf.layers.dense(x, dummy_env.action_space.n)
        self.policy = tf.nn.log_softmax(policy)
        self.value_batch = self.value = tf.squeeze(tf.layers.dense(x, 1), axis = 2)
        self.value_1 = self.value[0, -1]
        self.distribution_1 = tf.distributions.Categorical(logits = self.policy[0, -1, :])
        self.sample_op = self.distribution_1.sample()
        self.logprob_sample_1_op = self.distribution_1.log_prob(self.sample_op)
        self.distribution = tf.distributions.Categorical(logits = self.policy)
        self.logprob_op = self.distribution.log_prob(self.action_placeholder, name = "logprob_for_action")


    def __init__(self, env_creator,
                 clip_ratio = 0.2,
                 max_policy_steps = 80,
                 max_kl = 0.015,
                 lambda_gae = 0.95,
                 lr_schedule = (lambda t: 3e-4),
                 value_prop_schedule = (lambda t: 0.01),
                 log = None,
                 gamma = 0.9,
                 min_observations_per_step = 4000,
                 render = False,
                 render_mod = 16,
                 preprocess_op = (lambda x: x),
                 temporal_span = 32):
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
        self.temporal_span = temporal_span

        # We create a throwaway environment for the observation / action shape.
        # This shouldn't be too slow.
        dummy_env = env_creator.new_env()

        fp_observations = dummy_env.observation_space.dtype == np.dtype('float32')
        obs_dtype = tf.float32 if fp_observations else tf.int8
        self.obs_placeholder = tf.placeholder(obs_dtype,
                                              shape = [None, None] + list(dummy_env.observation_space.shape),
                                              name = "observation")
        if fp_observations:
            self.obs_input = self.obs_placeholder
        else:
            self.obs_input = tf.cast(self.obs_placeholder, tf.float32)
        self.action_placeholder = tf.placeholder(tf.int32, shape = [None, None],
                                              name = "action_placeholder")
        self.reward_placeholder = tf.placeholder(tf.float32, shape = [None, None],
                                              name = "reward_placeholder")
        self.adv_placeholder = tf.placeholder(tf.float32, shape = [None, None], name = "advantage")
        self.lr_placeholder = tf.placeholder(tf.float32, shape = [], name = "learning_rate")
        self.value_prop_placeholder = tf.placeholder(tf.float32, shape = [], name = "value_prop")
        self.logprob_sample_placeholder = tf.placeholder(tf.float32, shape = [None, None], name = "logprob_sample_placeholder")
        self.return_placeholder = tf.placeholder(tf.float32, shape = [None, None], name = "returns")

        # Helps us tell when sequences end
        self.mask_placeholder = tf.placeholder(tf.float32, shape = [None, None], name = "mask")
        self.seqlen_placeholder = tf.placeholder(tf.int32, shape = [None], name = "sequence_length_placeholder")
        self.total_timesteps = tf.reduce_sum(self.mask_placeholder)

        self._create_snail_vars(dummy_env)
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
        return PPO_GAE._initialize_sample_path_dict(self)

    def _step_once(self, path, env, obs):
        path['observations'].append(obs)
        action, logprob, value = self.session.run(
            [self.sample_op, self.logprob_sample_1_op, self.value_1],
            feed_dict = {
                self.obs_placeholder: [path['observations']]
            }
        )
        action = action
        path['actions'].append(action)
        path['logprobs'].append(logprob)
        path['values'].append(value)

        obs, reward, done, info = env.step(action)
        self.env_creator.update(done, info)
        path['rewards'].append(reward)

        return obs, reward, done, info

if __name__ == '__main__':
    log = loggy.Log("snail-rem16", autosave_freq = 15.0, autosave_vars_freq = 30.0, continuing = False)

    params = {
        # 'env_creator': schedules.ExploreCreatorSchedule(is_tree = False, history_size = 1,
        #                                 id_size = 1, reward_type = 'penalty+finished', scale_reward_by_difficulty = False),
        'env_creator': schedules.ConstantMazeSchedule('saved_mazes/rem16.dill'),
        'min_observations_per_step': 3000,
        'log': log,
        'render': False,
    }

    params = log.process_params(params)
    log.add_hyperparams(params)

    snail = SNAIL_PPO(**params)
    snail.initialize_variables()
    snail.optimize(1000000)
    log.close()
