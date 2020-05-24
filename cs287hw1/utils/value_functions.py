import autograd.numpy as np
from collections import OrderedDict


class MLPValueFun(object):
    _activations = {
        'tanh': np.tanh,
        None: lambda x: x,
        'relu': lambda x: np.maximum(x, 0)
    }

    def __init__(self, env, hidden_sizes=(256, 256), activation='relu'):
        self._env = env
        self._params = dict()
        self._build(hidden_sizes, activation)

    def _build(self, hidden_sizes=(256, 256), activation='relu', *args, **kwargs):
        self._activation = self._activations[activation]
        self._hidden_sizes = hidden_sizes
        prev_size = self._env.observation_space.shape[0]
        for i, hidden_size in enumerate(hidden_sizes):
            W = np.random.normal(loc=0, scale=1/prev_size, size=(hidden_size, prev_size))
            b = np.zeros((hidden_size,))

            self._params['W_%d' % i] = W
            self._params['b_%d' % i] = b

            prev_size = hidden_size

        W = np.random.normal(loc=0, scale=1/prev_size, size=(1, prev_size))
        b = np.zeros((1,))
        self._params['W_out'] = W
        self._params['b_out'] = b

    def get_values(self, states, params=None):
        params = self._params if params is None else params
        x = states
        for i, hidden_size in enumerate(self._hidden_sizes):
            x = np.dot(params['W_%d' % i], x.T).T + params['b_%d' % i]
            x = self._activation(x)
        values = np.dot(params['W_out'], x.T).T + params['b_out']
        return values[:, 0]

    def update(self, params):
        assert set(params.keys()) == set(self._params.keys())
        self._params = params

