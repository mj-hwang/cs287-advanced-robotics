import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils.utils import DiscreteEnv


class Grid1DEnv(DiscreteEnv):
    """
    actions: 0 left and 1 right
    """
    def __init__(self, discount=0.99, size=51):
        self.__name__ = self.__class__.__name__
        self._size = size
        self.max_path_length = 50
        self._goal = size // 4
        self._state = 0
        self.discount = discount
        self._fig = None
        self.dt = 0.01
        self.obs_dims = 1
        DiscreteEnv.__init__(self, size + 1, 3)

    def step(self, action):
        probs = self._transitions[self._state, action]
        next_state = np.argmax(np.random.multinomial(1, probs))
        reward = self._rewards[self._state, action, next_state]
        done = False
        env_info = dict()
        self._state = next_state
        return next_state, reward, done, env_info

    def reset(self):
        self._state = np.random.randint(0, self._size)
        return self._state

    def vec_reset(self, num_states):
        states = np.random.randint(0, self._size, size=(num_states,))
        self._states = states
        return self._states

    def vec_step(self, actions):
        assert self._states is not None
        assert len(self._states) == len(actions)
        probs = self._transitions[self._states, actions]
        next_states = np.argmax(probs, axis=-1)
        rewards = self._rewards[self._states, actions, next_states]
        dones = self._states == self._size ** 2
        env_info = dict()
        self._states = next_states
        return next_states, rewards, dones, env_info

    def _build_transitions(self):
        self._transitions[0, 2, 1] = 1.
        self._transitions[0, 1, 0] = 1.
        self._transitions[0, 0, 0] = 1.

        self._transitions[self._size-1, 2, self._size-1] = 1.
        self._transitions[self._size-1, 1, self._size - 1] = 1.
        self._transitions[self._size-1, 0, self._size-2] = 1.

        for i in range(1, self._size-1):
            self._transitions[i, 2, i+1] = 1.
            self._transitions[i, 1, i] = 1.
            self._transitions[i, 0, i-1] = 1.

        self._transitions[-self._goal, :, :] = 0.
        self._transitions[-self._goal, :, -1] = 1.
        self._transitions[self._goal, :, :] = 0.
        self._transitions[self._goal, :, -1] = 1.
        self._transitions[-1, :, :] = 0.
        self._transitions[-1, :, -1] = 1.

    def _build_rewards(self):
        self._rewards[:, :, -self._goal] = 1.
        self._rewards[:, :, self._goal] = 1.

    def render(self, mode='human', iteration=None):
        if self._states is None:
            states = np.array([self._state - self._size//2])
        else:
            states = self._states - self._size // 2
        if self._fig is None:
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(111)
            self._agent_render, = self._ax.plot(states, np.zeros_like(states), 'ro')
            self._goal_render = [self._ax.plot([-(self._goal+1)], [0], 'y*'), self._ax.plot([self._goal], [0], 'y*')]
            self._ax.set_xlim(-self._size//2 - 0.5, self._size//2 + 0.5)
            self._ax.set_ylim(-1, 1)
            self._ax.set_aspect('equal')

        self._agent_render.set_data(states, np.zeros_like(states))
        if iteration is not None:
            self._ax.set_title('Iteration %d' % iteration)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        if matplotlib.get_backend().lower() != 'agg':
            plt.pause(self.dt)
        if mode == 'rgb_array':
            width, height = self._fig.get_size_inches() * self._fig.get_dpi()
            image = np.fromstring(self._fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            return image

    def close(self):
        plt.close()
        self._fig = None
        self._ax = None
