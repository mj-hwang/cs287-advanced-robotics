import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time
from utils.utils import DiscreteEnv, upsample


class GridWorldEnv(DiscreteEnv):
    """
    Actions: 0 left and 1 right
    """
    def __init__(self, size=20, discount=0.99, seed=0):
        self.__name__ = self.__class__.__name__ + str(seed)
        self._state = 0
        self._states = None
        self._fig = None
        self.discount = discount
        self.max_path_length = 2 * size
        np.random.seed(seed)
        self._grid = np.random.binomial(1, 0.2, size=(size, size))
        self._grid[0, 0], self._grid[1, 0], self._grid[0, 1], self._grid[1, 1] = 0, 0, 0, 0
        self._grid[-1, -1], self._grid[-2, -1], self._grid[-1, -2], self._grid[-2, -2] = 0, 0, 0, 0
        self._rgb_grid = np.zeros((size, size, 3), dtype=np.uint8)
        self._rgb_grid[:, :, :] = np.expand_dims(((1-self._grid) * 255).astype(np.uint8), axis=-1)
        self._rgb_grid[size-1, size-1, :] = 255, 215, 0
        self._size = size
        self.dt = .02
        self.obs_dims = 2
        self._scale = 4
        self.vectorized = True
        DiscreteEnv.__init__(self, size * size + 1, 4)

    def step(self, action):
        probs = self._transitions[self._state, action]
        next_state = np.argmax(np.random.multinomial(1, probs))
        reward = self._rewards[self._state, action, next_state]
        done = self._state == self._size ** 2
        env_info = dict()
        self._state = next_state
        return next_state, reward, done, env_info

    def reset(self):
        self._states = None
        state = np.random.randint(0, self._size * self._size)
        while self._grid[state % self._size, state//self._size]:
            state = np.random.randint(0, self._size * self._size)
        self._state = state
        return self._state

    def vec_reset(self, num_states):
        states = np.random.randint(0, self._size * self._size, size=(num_states,))
        collisions = self._grid[states % self._size, states // self._size]
        num_collisions = np.sum(collisions)
        while num_collisions:
            states[collisions.astype(bool)] = np.random.randint(0, self._size * self._size, size=(num_collisions,))
            collisions = self._grid[states % self._size, states // self._size]
            num_collisions = np.sum(collisions)
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
        size = self._size
        for x in range(size):
            for y in range(size):
                for act in range(4):
                    id_s = x + y * size
                    if act == 0:
                        next_x = x - 1
                        next_y = y
                    elif act == 1:
                        next_x = x + 1
                        next_y = y
                    elif act == 2:
                        next_x = x
                        next_y = y - 1
                    else:
                        next_x = x
                        next_y = y + 1

                    if (next_x < 0) or (next_x >= size):
                        next_x = x
                    if (next_y < 0) or (next_y >= size):
                        next_y = y
                    if self._grid[next_x, next_y] or self._grid[x, y]:
                        next_x, next_y = x, y

                    id_next_s = next_x + next_y * size
                    self._transitions[id_s, act, id_next_s] = 1.
            self._transitions[-2, :, :] = 0.
            self._transitions[-2, :, -1] = 1.
            self._transitions[-1, :, -1] = 1.

    def _build_rewards(self):
        self._rewards[-2, :, -1] = 1.

    def render(self, mode='human', iteration=None):
        if self._fig is None:
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(111)
            data = upsample(self._rgb_grid, self._scale)
            self._render = self._ax.imshow(data, animated=True)
            self._ax.tick_params(
                axis='both',
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False)  # labels along the bottom edge are off
            self._ax.set_aspect('equal')
            self._canvas = FigureCanvas(self._fig)

        data = self._rgb_grid.copy()
        if self._states is None:
            x, y = self._state % self._size, self._state//self._size
            if self._state != self._size ** 2:
                data[x, y, :] = [255, 0, 0]

        else:
            x, y = self._states % self._size, self._states//self._size
            x = x[self._states != self._size ** 2]
            y = y[self._states != self._size ** 2]
            data[x, y, :] = [255, 0, 0]

        data = upsample(data, self._scale)
        self._render.set_data(data)
        if iteration is not None:
            self._ax.set_title('Iteration %d' % iteration)
        self._canvas.draw()
        self._canvas.flush_events()
        time.sleep(self.dt)

        if mode == 'rgb_array':
            width, height = self._fig.get_size_inches() * self._fig.get_dpi()
            image = np.fromstring(self._canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            return image

    def upsample(self, image, scale):
        up_image = np.repeat(image, self._scale, axis=0)
        up_image = np.repeat(up_image, self._scale, axis=1)
        return up_image

    def close(self):
        plt.close()
        self._fig = None
