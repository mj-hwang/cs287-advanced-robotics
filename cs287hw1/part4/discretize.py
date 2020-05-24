import numpy as np
from utils.utils import DiscretizeWrapper


class Discretize(DiscretizeWrapper):
    """
    Discretize class: Discretizes a continous gym environment


    -- VARIABLES/FUNCTIONS YOU WILL NEED TO USE --
        * self.state_points (np.ndarray): grid that contains the real values of the discretization

        * self.obs_n (int): number of discrete points

        * self.transitions (np.ndarray): transition matrix of size (S+1, A, S+1). The last state corresponds to the sink
                                         state
        * self.rewards (np.ndarray): reward matrix of size (S+1, A, S+1). The last state corresponds to the sink state

        * self.get_id_from_coordinates(coordinate_vector) returns the id of the coordinate_vector

        * self.get_state_from_id(id_s): get the continuous state associated to that state id

        * self.get_action_from_id(id_a): get the contiouns action associated to that action id

        * env.set_state(s): resets the environment to the continous state s

        * env.step(a): applies the action a to the environment. Returns next_state, reward, done, env_infos. The
                            last element is not used.
    """

    def vec_get_discrete_state_from_cont_state(self, cont_state):
        """
        Get discrete state from continuous state.
            * self.mode (str): specifies if the discretization is to the nearest-neighbour (nn) or n-linear (linear).

        :param cont_state (np.ndarray): of shape env.observation_space.shape
        :return: A tuple of (states, probs). states is a np.ndarray of shape (1,) if mode=='nn'
                and (2 ^ obs_dim,) if mode=='linear'. probs is the probabability of going to such states,
                it has the same size than states.
        """
        """INSERT YOUR CODE HERE"""
        cont_state = np.expand_dims(cont_state, axis=-1) 
        obs_dim = cont_state.shape[0]
        if self.mode == 'nn':
            closest_i = np.argmin(np.abs(self.state_points - cont_state), 
                                  axis=1)    
            id_s = self.get_id_from_coordinates(closest_i)
            states = np.array([id_s])
            probs = np.array([1])


        elif self.mode == 'linear':
            upper_i = np.argmax(self.state_points > cont_state, axis=-1)
            lower_i = upper_i - 1
            
            too_small_i = np.sum(self.state_points < cont_state, axis=-1) == 0
            too_large_i = np.sum(self.state_points > cont_state, axis=-1) == 0

            upper_s = np.expand_dims(self.state_points[np.arange(obs_dim), 
                                                       upper_i], -1)
            lower_s = np.expand_dims(self.state_points[np.arange(obs_dim), 
                                                       lower_i], -1)
            upper_p = (cont_state - lower_s) / (upper_s - lower_s)
            lower_p = (cont_state - upper_s) / (lower_s - upper_s)

            cs = np.column_stack([lower_i, upper_i])
            ps = np.column_stack([lower_p, upper_p])

            ps[too_small_i] = [0, 1]
            ps[too_large_i] = [1, 0]

            c_combos = np.array(np.meshgrid(*cs)).T.reshape(-1, obs_dim)
            p_combos = np.array(np.meshgrid(*ps)).T.reshape(-1, obs_dim)

            states = self.get_id_from_coordinates(c_combos)
            # probs = np.prod(p_combos, axis=1).round(decimals=2).astype(np.float64) 
            probs = np.prod(p_combos, axis=1)
        else:
            raise NotImplementedError
        return states, probs

    def vec_add_transition(self, id_s, id_a):
        """
        Populates transition and reward matrix (self.transition and self.reward)
        :param id_s (int): discrete index of the the state
        :param id_a (int): discrete index of the the action

        """
        env = self._wrapped_env
        obs_n = self.obs_n

        """INSERT YOUR CODE HERE"""
        s = self.get_state_from_id(id_s)
        a = self.get_action_from_id(id_a)
        env.set_state(s)
        # id_next_s, reward, done, env_infos = self.step(id_a)
        ns, reward, done, env_infos = env.step(a)
        if done:
            self.transitions[id_s, id_a, obs_n] = 1
            self.rewards[id_s, id_a, obs_n] = reward
        else:
            # ns = self.get_state_from_id(id_next_s)
            nds, probs = self.vec_get_discrete_state_from_cont_state(ns)
            if self.mode == 'nn':
                self.transitions[id_s, id_a, nds] = probs
                self.rewards[id_s, id_a, nds] = reward
            if self.mode == 'linear':    
                for i in range(len(nds)):
                    self.transitions[id_s, id_a, nds[i]] = probs[i]
                    self.rewards[id_s, id_a, nds[i]] = reward

    def add_done_transitions(self):
        """
        Populates transition and reward matrix for the sink state (self.transition and self.reward). The sink state
        corresponds to the last state (self.obs_n or -1).
        """
        """INSERT YOUR CODE HERE"""
        self.transitions[-1, :, -1] = 1
        self.rewards[-1, :, -1] = 0



