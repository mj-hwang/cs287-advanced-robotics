import tensorflow as tf
import time

class SAC:
    """Soft Actor-Critic (SAC)
    Original code from Tuomas Haarnoja, Soroush Nasiriany, and Aurick Zhou for CS294-112 Fall 2018

    References
    ----------
    [1] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine, "Soft
        Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
        with a Stochastic Actor," ICML 2018.
    """

    def __init__(self,
                 alpha=1.0,
                 batch_size=256,
                 discount=0.99,
                 epoch_length=1000,
                 learning_rate=3e-3,
                 reparameterize=False,
                 tau=0.01,
                 **kwargs):
        """
        Args:
        """

        self._alpha = alpha
        self._batch_size = batch_size
        self._discount = discount
        self._epoch_length = epoch_length
        self._learning_rate = learning_rate
        self._reparameterize = reparameterize
        self._tau = tau

        self._training_ops = []

    def build(self, env, policy, q_function, q_function2, value_function,
              target_value_function):

        self._create_placeholders(env)

        policy_loss = self._policy_loss_for(policy, q_function, q_function2, value_function)
        value_function_loss = self._value_function_loss_for(
            policy, q_function, q_function2, value_function)
        q_function_loss = self._q_function_loss_for(q_function,
                                                    target_value_function)
        if q_function2 is not None:
            q_function2_loss = self._q_function_loss_for(q_function2,
                                                        target_value_function)

        optimizer = tf.train.AdamOptimizer(
            self._learning_rate, name='optimizer')
        policy_training_op = optimizer.minimize(
            loss=policy_loss, var_list=policy.trainable_variables)
        value_training_op = optimizer.minimize(
            loss=value_function_loss,
            var_list=value_function.trainable_variables)
        q_function_training_op = optimizer.minimize(
            loss=q_function_loss, var_list=q_function.trainable_variables)
        if q_function2 is not None:
            q_function2_training_op = optimizer.minimize(
                loss=q_function2_loss, var_list=q_function2.trainable_variables)

        self._training_ops = [
            policy_training_op, value_training_op, q_function_training_op
        ]
        if q_function2 is not None:
            self._training_ops += [q_function2_training_op]
        self._target_update_ops = self._create_target_update(
            source=value_function, target=target_value_function)

        tf.get_default_session().run(tf.global_variables_initializer())

    def _create_placeholders(self, env):
        observation_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, observation_dim),
            name='observation',
        )
        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, observation_dim),
            name='next_observation',
        )
        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, action_dim),
            name='actions',
        )
        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='rewards',
        )
        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='terminals',
        )

    def _policy_loss_for(self, policy, q_function, q_function2, value_function):
        if not self._reparameterize:
            """  CODE PROVIDED """

            actions, log_pis = policy(self._observations_ph)
            if q_function2 is None:
                q_vals = q_function([self._observations_ph, actions])
            else:
                """ YOUR CODE HERE FOR PROBLEM 3A.3"""
                q_vals = tf.minimum(
                    q_function([self._observations_ph, actions]),
                    q_function2([self._observations_ph, actions])
                )
            q_vals = tf.squeeze(q_vals, axis=1)
                
            baseline = value_function(self._observations_ph)
            baseline = tf.squeeze(baseline, axis=1)
            
            target = self._alpha * log_pis - q_vals + baseline
            
            target = tf.stop_gradient(target)
            result = tf.reduce_mean(log_pis * target)
            """ CODE ENDS """
            return result
        else:
            """ CODE PROVIDED"""
            actions, log_pis = policy(self._observations_ph)
            if q_function2 is None:
                q_vals = q_function([self._observations_ph, actions])
            else:
                q_vals = tf.minimum(
                    q_function([self._observations_ph, actions]),
                    q_function2([self._observations_ph, actions])
                )
            q_vals = tf.squeeze(q_vals, axis=1)    
            
            result = tf.reduce_mean(
                self._alpha * log_pis - q_vals
            )
            """ CODE ENDS """
            return result

    def _value_function_loss_for(self, policy, q_function, q_function2, value_function):
        """
        :param policy:
        :param q_function:
        :param q_function2:
        :param value_function:
        :return:
        """
        """ YOUR CODE HERE FOR PROBLEM 3A.2"""

        actions, log_pis = policy(self._observations_ph)
        vals = tf.squeeze(value_function(self._observations_ph), axis=1)

        if q_function2 is None:
            q_vals = tf.squeeze(q_function([self._observations_ph, actions]), axis=1)
        else:
            q_vals = tf.minimum(tf.squeeze(q_function([self._observations_ph, actions]), axis=1),
                                tf.squeeze(q_function2([self._observations_ph, actions]), axis=1))

        loss = tf.losses.mean_squared_error(q_vals - self._alpha * log_pis, 
                                            vals)

        return loss
        """ YOUR CODE HERE FOR PROBLEM 3A.3"""



    def _q_function_loss_for(self, q_function, target_value_function):
        """ q loss """
        """ YOUR CODE HERE FOR PROBLEM 3A.1"""
        q_vals = tf.squeeze(q_function([self._observations_ph, self._actions_ph]), axis=1)
        target_vals = tf.squeeze(target_value_function(self._next_observations_ph), axis=1) 
        target_q_vals = self._rewards_ph + target_vals * (1-self._terminals_ph) * self._discount
        loss = tf.losses.mean_squared_error(target_q_vals, 
                                            q_vals)

        return loss


    def _create_target_update(self, source, target):
        """Create tensorflow operations for updating target value function."""

        return [
            tf.assign(target, (1 - self._tau) * target + self._tau * source)
            for target, source in zip(target.trainable_variables, source.
                                      trainable_variables)
        ]

    def train(self, sampler, n_epochs=1000):
        """Return a generator that performs RL training.

        Args:
            env (`rllab.Env`): Environment used for training
            policy (`Policy`): Policy used for training
            initial_exploration_policy ('Policy'): Policy used for exploration
                If None, then all exploration is done using policy
            pool (`PoolBase`): Sample pool to add samples to
        """
        self._start = time.time()
        for epoch in range(n_epochs):
            for t in range(self._epoch_length):
                sampler.sample()

                batch = sampler.random_batch(self._batch_size)
                feed_dict = {
                    self._observations_ph: batch['observations'],
                    self._actions_ph: batch['actions'],
                    self._next_observations_ph: batch['next_observations'],
                    self._rewards_ph: batch['rewards'],
                    self._terminals_ph: batch['terminals'],
                }
                tf.get_default_session().run(self._training_ops, feed_dict)
                tf.get_default_session().run(self._target_update_ops)

            yield epoch

    def get_statistics(self):
        statistics = {
            'Time': time.time() - self._start,
            'TimestepsThisBatch': self._epoch_length,
        }

        return statistics
