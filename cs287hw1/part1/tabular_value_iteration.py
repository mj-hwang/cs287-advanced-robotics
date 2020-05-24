import matplotlib.pyplot as plt
from utils.plot import plot_contour, rollout, plot_returns
from utils.utils import upsample
import logger
import numpy as np
import moviepy.editor as mpy
import time


class ValueIteration(object):
    """
    Tabular Value Iteration algorithm.

    -- UTILS VARIABLES FOR RUNNING THE CODE -- (feel free to play with them but we do not ask it for the homework)
        * policy (TabularPolicy):

        * precision (float): tolerance for the final values (determines the amount of iterations)

        * policy_type (str): whether the policy is deterministic or max-ent

        * log_itr (int): number of iterations between logging

        * max_itr (int): maximum number of iterations

        * render (bool): whether to render or not

    -- VARIABLES/FUNCTIONS YOU WILL NEED TO USE --
        * value_fun (TabularValueFun):
                - get_values(states): if states is None returns the values of all the states. Otherwise, it returns the
                                      values of the specified states

        * self.transitions (np.ndarray): transition matrix of size (S, A, S)

        * self.rewards (np.ndarray): reward matrix of size (S, A, S)

        * self.discount (float): discount factor of the problem

        * self.temperature (float): temperature for the maximum entropy policies


    """
    def __init__(self,
                 env,
                 value_fun,
                 policy,
                 precision=1e-3,
                 log_itr=1,
                 render_itr=2,
                 policy_type='deterministic',
                 max_itr=50,
                 render=True,
                 num_rollouts=20,
                 temperature=1.,
                 ):
        self.env = env
        self.transitions = env.transitions
        self.rewards = env.rewards
        self.value_fun = value_fun
        self.policy = policy
        self.discount = env.discount
        self.precision = precision
        self.log_itr = log_itr
        assert policy_type in ['deterministic', 'max_ent']
        self.policy_type = policy_type
        self.max_itr = max_itr
        self.render_itr = render_itr
        self.render = render
        self.num_rollouts = num_rollouts
        self.temperature = temperature
        self.eps = 1e-8

    def train(self):
        next_v = 1e6
        v = self.value_fun.get_values()
        itr = 0
        videos = []
        contours = []
        returns = []
        fig = None

        while not self._stop_condition(itr, next_v, v) and itr < self.max_itr:
            log = itr % self.log_itr == 0
            render = (itr % self.render_itr == 0) and self.render
            if log:
                next_pi = self.get_next_policy()
                self.policy.update(next_pi)
                average_return, video = rollout(self.env, self.policy, render=render,
                                                num_rollouts=self.num_rollouts, iteration=itr)
                if render:
                    contour, fig = plot_contour(self.env, self.value_fun, fig=fig, iteration=itr)
                    contours += [contour] * len(video)
                    videos += video
                returns.append(average_return)
                logger.logkv('Iteration', itr)
                logger.logkv('Average Returns', average_return)
                logger.dumpkvs()
            next_v = self.get_next_values()
            self.value_fun.update(next_v)
            itr += 1

        next_pi = self.get_next_policy()
        self.policy.update(next_pi)
        contour, fig = plot_contour(self.env, self.value_fun, save=True, fig=fig, iteration=itr)
        average_return, video = rollout(self.env, self.policy,
                                        render=self.render, num_rollouts=self.num_rollouts, iteration=itr)
        plot_returns(returns)
        if self.render:
            videos += video
            contours += [contour]
        logger.logkv('Iteration', itr)
        logger.logkv('Average Returns', average_return)

        fps = int(4/getattr(self.env, 'dt', 0.1))
        if contours and contours[0] is not None:
            clip = mpy.ImageSequenceClip(contours, fps=fps)
            clip.write_videofile('%s/contours_progress.mp4' % logger.get_dir())

        if videos:
            clip = mpy.ImageSequenceClip(videos, fps=fps)
            clip.write_videofile('%s/roll_outs.mp4' % logger.get_dir())

        plt.close()

    def get_next_values(self):
        """
        Next values given by the Bellman equation

        :return np.ndarray with the values for each state, shape (num_states,)
        For the maximum entropy policy, to compute the unnormalized probabilities make sure:
                                    1) Before computing the exponientated value substract the maximum value per state
                                       over all the actions.
                                    2) Add self.eps to them
        """

        """ INSERT YOUR CODE HERE"""
        if self.policy_type == 'deterministic':
            q = np.multiply(self.transitions,
                            self.rewards + self.discount * self.value_fun.get_values()).sum(axis=2)
            next_v = q.max(axis=1)
        elif self.policy_type == 'max_ent':
            q = np.multiply(self.transitions,
                            self.rewards + self.discount * self.value_fun.get_values()).sum(axis=2)
            q_softmax = np.log(np.exp(((q - q.max(axis=1, keepdims=True)) / self.temperature)).sum(axis=1))
            next_v = self.temperature * q_softmax + q.max(axis=1)
        else:
            raise NotImplementedError
        return next_v

    def get_next_policy(self):
        """
        Next policy probabilities given by the Bellman equation

        :return np.ndarray with the policy probabilities for each state and actions, shape (num_states, num_actions)
        For the maximum entropy policy, to compute the unnormalized probabilities make sure:
                                    1) Before computing the exponientated value substract the maximum value per state
                                       over all the actions.
                                    2) Add self.eps to them
        """

        """INSERT YOUR CODE HERE"""
        if self.policy_type == 'deterministic':
            q = np.multiply(self.transitions,
                            self.rewards + self.discount * self.value_fun.get_values()).sum(axis=2)
            pi = q.argmax(axis=1)
        elif self.policy_type == 'max_ent':
            q = np.multiply(self.transitions,
                            self.rewards + self.discount * self.value_fun.get_values()).sum(axis=2)
            pi_unnormalized = np.exp(q - q.max(axis=1, keepdims=True)) + self.eps
            pi = pi_unnormalized / pi_unnormalized.sum(axis=1, keepdims=True)
        else:
            raise NotImplementedError
        return pi

    def _stop_condition(self, itr, next_v, v):
        rmax = np.max(np.abs(self.env.rewards))
        cond1 = np.max(np.abs(next_v - v)) < self.precision/(2 * self.discount/(1 - self.discount))
        cond2 = self.discount ** itr * rmax/(1 - self.discount) < self.precision
        return cond1 or cond2
