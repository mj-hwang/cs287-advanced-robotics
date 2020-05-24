import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time
from gym import spaces, Env


class DoubleIntegratorEnv(Env):
    """
    state: [pos, vel]
    """
    def __init__(self, discount=0.99):
        self._state = np.zeros((2,))
        self.dt = 0.05
        self.max_path_length = 200
        self._fig = None
        self.discount = discount
        self.vectorized = True
        self.action_space = spaces.Box(low=np.array((-3,)), high=np.array((3,)), dtype=np.float64)
        self.observation_space = spaces.Box(low=np.array((-4, -4)), high=np.array((4, 4)), dtype=np.float64)

    def step(self, action):
        next_state = self._state + np.array([self._state[1], action[0]]) * self.dt
        reward = -0.5 * (self._state[0] ** 2 + self._state[1] ** 2 + action ** 2)
        done = (next_state < self.observation_space.low).any() or (next_state > self.observation_space.high).any()
        env_info = dict()
        self._state = next_state
        if done:
            reward /= (1 - self.discount)
        return next_state.copy(), reward, done, env_info

    def reset(self):
        self._states = None
        # self._state = np.random.uniform(low=-2, high=2, size=2)
        self._state = np.ones((2,))
        return self._state.copy()

    def set_state(self, state):
        self._state = state

    def vec_step(self, actions):
        next_states = self._states + np.stack([self._states[:, 1], actions[:, 0]], axis=-1) * self.dt
        rewards = -0.5 * (self._states[:, 0] ** 2 + self._states[:, 1] ** 2 + actions[:, 0] ** 2)
        dones = np.sum([(next_states[:, i] < l) + (next_states[:, i] > h) for i, (l, h)
                        in enumerate(zip(self.observation_space.low, self.observation_space.high))], axis=0).astype(np.bool)
        env_infos = dict()
        self._states = next_states
        rewards[dones] /= (1 - self.discount)
        return next_states, rewards, dones, env_infos

    def vec_set_state(self, states):
        self._states = states

    def vec_reset(self, num_envs=None):
        if num_envs is None:
            assert self._num_envs is not None
            num_envs = self._num_envs
        else:
            self._num_envs = num_envs
        self._states = np.random.uniform(low=-2, high=2, size=(num_envs, 2))
        return self._states

    def render(self, mode='human', iteration=None):
        if self._fig is None:
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(111)
            self._agent_render, = self._ax.plot(self._state[0], 0, 'ro')
            self._goal_render, = self._ax.plot(0, 'y*')
            self._ax.set_xlim(-4.5, 4.5)
            self._ax.set_ylim(-.5, .5)
            self._ax.set_aspect('equal')
            self._canvas = FigureCanvas(self._fig)

        self._agent_render.set_data(self._state[0], 0)
        if iteration is not None:
            self._ax.set_title('Iteration %d' % iteration)
        self._canvas.draw()
        # time.sleep(self.dt)
        self._canvas.flush_events()
        if mode == 'rgb_array':
            width, height = self._fig.get_size_inches() * self._fig.get_dpi()
            image = np.fromstring(self._canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            return image

    def close(self):
        plt.close()
        self._fig = None
