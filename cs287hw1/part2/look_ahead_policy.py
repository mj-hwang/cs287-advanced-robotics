import numpy as np
from gym import spaces


class LookAheadPolicy(object):
    """
    Look ahead policy

    -- VARIABLES/FUNCTIONS YOU WILL NEED TO USE --
    * self.horizon (int): Horizon for the look ahead policy

    * act_dim (int): Dimension of the state space

    * value_fun (TabularValueFun):
                - get_values(states): if states is None returns the values of all the states. Otherwise, it returns the
                                      values of the specified states
    * env (Env):
                - vec_set_state(states): vectorized (multiple environments in parallel) version of reseting the
                environment to a state for a batch of states.
                - vec_step(actions): vectorized (multiple environments in parallel) version of stepping through the
                environment for a batch of actions. Returns the next observations, rewards, dones signals, env infos
                (last not used).
    """
    def __init__(self,
                 env,
                 value_fun,
                 horizon,
                 ):
        self.env = env
        self.discount = env.discount
        self._value_fun = value_fun
        self.horizon = horizon

    def get_action(self, state):
        """
        Get the best action by doing look ahead, covering actions for the specified horizon.
        HINT: use np.meshgrid to compute all the possible action sequences.
        :param state:
        :return: best_action (int)
           """
        assert isinstance(self.env.action_space, spaces.Discrete)
        act_dim = self.env.action_space.n
        """ INSERT YOUR CODE HERE"""
        actions = np.arange(act_dim)
        sequences = np.array(np.meshgrid(*np.tile(np.arange(act_dim), 
                                                  (self.horizon, 1)))).T.reshape(-1, self.horizon).T
        return sequences[0, np.argmax(self.get_returns(state, sequences))]

    def get_returns(self, state, actions):
        """
        :param state: current state of the policy
        :param actions: array of actions of shape [horizon, num_acts]
        :return: returns for the specified horizon + self.discount ^ H value_fun
        HINT: Make sure to take the discounting and done into acount!
        """
        assert self.env.vectorized
        """ INSERT YOUR CODE HERE"""
        num_acts = actions.shape[1]
        returns = np.zeros(num_acts)
        # self.env.set_state(state)
        # if len(actions.shape) < 3:
        #     self.env.vec_set_state(np.full(num_acts, state))
        # else:
        self.env.vec_set_state(np.tile(state, (num_acts, 1)))
        for h in range(self.horizon):
            observations, rewards, dones, env_infos = self.env.vec_step(actions[h])
            self.env.vec_set_state(observations)
            returns += self.discount ** h * rewards
        returns += self.discount ** self.horizon * self._value_fun.get_values(observations)
        return returns

    def update(self, actions):
        pass
