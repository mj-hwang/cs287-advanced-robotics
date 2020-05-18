import numpy as np
import moviepy.editor as mpy
from scipy.optimize import minimize


class NNPolicy(object):
    def __init__(self, input_dim, output_dim, hidden_sizes):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = tuple(hidden_sizes)
        self.params = None

    def get_action(self, state, timestep=None):
        x = state
        params = self.params
        for i in range(len(self.hidden_sizes)):
            x = x.T @ params['W'][i] + params['b'][i]
            x = np.tanh(x)
        action = x.T @ params['W'][-1] + params['b'][-1]
        action = np.tanh(action)
        return action

    def set_params(self, params):
        sizes = (self.input_dim,) + self.hidden_sizes + (self.output_dim,)
        Ws, bs = [], []
        s_id = 0
        for i in range(len(self.hidden_sizes) + 1):
            w_shape = (sizes[i], sizes[i + 1])
            e_id = s_id + np.prod(w_shape)
            W = params[s_id:e_id].reshape(w_shape)
            s_id = e_id
            e_id = s_id + sizes[i + 1]
            b = params[s_id:e_id]
            s_id = e_id
            Ws.append(W)
            bs.append(b)
        self.params = dict(W=Ws, b=bs)

    def get_params(self):
        params = []
        for W, b in zip(self.params['W'], self.params['b']):
            params.extend([W.flatten(), b.flatten()])
        return np.concatenate(params)

    def init_params(self):
        sizes = (self.input_dim,) + self.hidden_sizes + (self.output_dim,)
        Ws, bs = [], []
        for i in range(len(self.hidden_sizes) + 1):
            W = np.random.uniform(size=(sizes[i], sizes[i + 1]))/np.sqrt(sizes[i] + sizes[i+1])
            b = np.zeros(shape=sizes[i + 1])
            Ws.append(W)
            bs.append(b)
        self.params = dict(W=Ws, b=bs)
        return dict(W=Ws, b=bs)


class ActPolicy(object):
    def __init__(self, env, actions):
        self._actions = actions.reshape(env.H, env.du)
        self.t = 0

    def get_action(self, state, timestep=None):
        act = self._actions[self.t]
        self.t = (self.t + 1) % len(self._actions)
        return act

    def reset(self):
        self.t = 0


def rollout(env, policy, noise=0., render=False):
    np.random.seed(0)
    s = env.reset()
    states = []
    imgs = []
    cost = 0
    for t in range(env.H):
        act = policy.get_action(s, t) + np.random.normal(0, scale=noise, size=(env.du,))
        s, c, d, _ = env.step(act)
        if render:imgs.append(env.render('rgb_array'))
        states.append(s)
        cost += c
        if d: break
    if render:
        clip = mpy.ImageSequenceClip(imgs, fps=8)
        clip.write_gif('./rollout.gif', verbose=False)

    return cost, states

