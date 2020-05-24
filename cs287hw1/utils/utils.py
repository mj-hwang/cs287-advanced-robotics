import numpy as np
import time
from collections.abc import Iterable
from gym import spaces, Env
from multiprocessing import Process, Pipe
import pickle
from autograd import grad


def upsample(image, scale):
    up_image = np.repeat(image, scale, axis=-2)
    up_image = np.repeat(up_image, scale, axis=-3)
    return up_image


def grad_step(loss, params, lr):
    gradient = grad(loss)(params)
    new_params = dict([(k, params[k] - lr * gradient[k]) for k in gradient.keys()])
    return new_params


class AdamOptimizer(object):
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., **kwargs):

        self.iterations = 0
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.decay = decay
        self.epsilon = epsilon
        self.initial_decay = decay

    def grad_step(self, loss, params):
        keys = list(params.keys())
        grads = grad(loss)(params)

        original_shapes = [params[k].shape for k in keys]
        params = [params[k].flatten() for k in keys]
        grads = [grads[k].flatten() for k in keys]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        lr_t = lr * (np.sqrt(1. - np.power(self.beta_2, t)) /
                     (1. - np.power(self.beta_1, t)))

        if not hasattr(self, 'ms'):
            self.ms = [np.zeros(p.shape) for p in params]
            self.vs = [np.zeros(p.shape) for p in params]

        ret = [None] * len(params)
        for i, p, g, m, v in zip(range(len(params)), params, grads, self.ms, self.vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * np.square(g)
            p_t = p - lr_t * m_t / (np.sqrt(v_t) + self.epsilon)
            self.ms[i] = m_t
            self.vs[i] = v_t
            ret[i] = p_t

        self.iterations += 1

        for i in range(len(ret)):
            ret[i] = ret[i].reshape(original_shapes[i])

        new_params = dict(zip(keys, ret))
        return new_params


class TabularValueFun(object):
    def __init__(self, env):
        self.obs_dim = env.observation_space.n
        self._value_fun = np.zeros(shape=(self.obs_dim,))

    def get_values(self, states=None):
        if states is None:
            return self._value_fun
        else:
            return self._value_fun[states]

    def update(self, values):
        self._value_fun = values


class TabularPolicy(object):
    def __init__(self, env):
        assert isinstance(env.action_space, spaces.Discrete)
        assert isinstance(env.observation_space, spaces.Discrete)
        self.act_dim = env.action_space.n
        self.obs_dim = env.observation_space.n
        self._policy = np.random.uniform(0, 1, size=(self.obs_dim, self.act_dim))

    def get_action(self, state):
        probs = np.array(self._policy[state])
        if probs.ndim == 2:
            probs = probs / np.expand_dims(np.sum(probs, axis=-1), axis=-1)
            s = probs.cumsum(axis=-1)
            r = np.expand_dims(np.random.rand(probs.shape[0]), axis=-1)
            action = (s < r).sum(axis=1)
        elif probs.ndim == 1:
            idxs = np.random.multinomial(1, probs / np.sum(probs))
            action = np.argmax(idxs)
        else:
            raise NotImplementedError
        return action

    def get_probs(self):
        return np.array(self._policy) / np.expand_dims(np.sum(self._policy, axis=-1), axis=-1)

    def update(self, actions):
        assert (actions >= 0).all()
        assert actions.shape[0] == self.obs_dim
        if actions.ndim == 1:
            self._policy[:, :] = 0
            self._policy[range(self.obs_dim), actions] = 1.
        elif actions.ndim == 2:
            self._policy = actions
        else:
            raise TypeError


class SparseArray(object):
    def __init__(self, obs_n, act_n, mode, obs_dims=None):
        if mode == 'nn':
            next_obs_n = 1
        elif mode == 'linear':
            assert obs_dims is not None
            next_obs_n = int(2 ** obs_dims)
        else:
            raise NotImplementedError
        self._obs_n = obs_n
        self._act_n = act_n
        self._mode = mode
        self._obs_dims = obs_dims
        self._values = np.zeros((obs_n, act_n, next_obs_n), dtype=np.float32)
        self._idxs = np.zeros((obs_n, act_n, next_obs_n), dtype=int)
        self._fill = np.zeros((obs_n, act_n), dtype=int)

    def __mul__(self, other):
        if isinstance(other, SparseArray):
            assert (self._idxs == other._idxs).all(), "other does not have the same sparsity"
            result = SparseArray(self._obs_n, self._act_n, self._mode, self._obs_dims)
            result._idxs = self._idxs
            result._values = self._values * other._values

        elif isinstance(other, np.ndarray):
            assert other.shape == (1, 1, self._obs_n)
            result = SparseArray(self._obs_n, self._act_n, self._mode, self._obs_dims)
            result._idxs = self._idxs
            result._values = self._values * other[self._idxs]

        else:
            raise NotImplementedError

        return result

    def __add__(self, other):
        if isinstance(other, SparseArray):
            assert (self._idxs == other._idxs).all(), "other does not have the same sparsity"
            result = SparseArray(self._obs_n, self._act_n, self._mode, self._obs_dims)
            result._idxs = self._idxs
            result._values = self._values + other._values

        elif isinstance(other, np.ndarray):
            assert other.shape == (self._obs_n,)
            result = SparseArray(self._obs_n, self._act_n, self._mode, self._obs_dims)
            result._idxs = self._idxs
            result._values = self._values + other[self._idxs]

        else:
            raise NotImplementedError

        return result

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        _inputs = tuple()
        for inp in inputs:
            if isinstance(inp, SparseArray):
                _inputs += (inp._values,)
            else:
                _inputs += (inp,)
        return getattr(ufunc, method)(*_inputs, **kwargs)

    def sum(self, *args, **kwargs):
        return self._values.sum(*args, **kwargs)

    def max(self, *args, **kwargs):
        return self._values.max(*args, **kwargs)

    def reshape(self, *args, **kwargs):
        return self._values.reshape(*args, **kwargs)

    def transpose(self, *args, **kwargs):
        return self._values.transpose(*args, **kwargs)

    def __setitem__(self, key, value):
        if type(key) is not tuple:
            self._values[key] = value
        elif len(key) == 2:
            obs, act = key
            self._values[obs, act] = value
        else:
            obs, act, n_obs = key
            if self._mode == 'nn':
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    assert (value.shape[0] == 1 or value.shape[1] == 1)
                    value = value.reshape(-1)
                if isinstance(obs, np.ndarray) and obs.ndim == 2:
                    assert (obs.shape[0] == 1 or obs.shape[1] == 1)
                    obs = obs.reshape(-1)
                if isinstance(act, np.ndarray) and act.ndim == 2:
                    assert (act.shape[0] == 1 or act.shape[1] == 1)
                    act = act.reshape(-1)
                if isinstance(n_obs, np.ndarray) and n_obs.ndim == 2:
                    assert (n_obs.shape[0] == 1 or n_obs.shape[1] == 1)
                    n_obs = n_obs.reshape(-1)

                self._values[obs, act, 0] = value
                self._idxs[obs, act, 0] = n_obs

            elif self._mode == 'linear':
                if isinstance(n_obs, np.ndarray) and n_obs.ndim == 2:
                    assert n_obs.shape[-1] == int(2 ** self._obs_dims)
                    if value.ndim == 1:
                        self._values[obs, act, :] = np.expand_dims(value, axis=-1)
                    else:
                        self._values[obs, act, :] = value
                    self._idxs[obs, act, :] = n_obs
                else:
                    self._values[obs, act, self._fill[obs, act]] = value
                    self._idxs[obs, act, self._fill[obs, act]] = n_obs
                    self._fill[obs, act] += 1

    def __getitem__(self, key):
        if type(key) is not tuple:
            return self._values[key]
        elif len(key) == 2:
            obs, act = key
            return self._values[obs, act]
        else:
            obs, act, n_obs = key
            if self._mode == 'nn':
                assert (n_obs == self._idxs[obs, act, 0]).all()
                return self._values[obs, act, 0]

            elif self._mode == 'linear':
                assert (n_obs == self._idxs[obs, act, self._fill[obs, act]]).all()
                return self._values[obs, act, self._fill[obs, act]]


class DiscretizeWrapper(object):
    def __init__(self,
                 env,
                 state_discretization=21,
                 action_discretization=5,
                 mode='linear',
                 ):

        self._wrapped_env = env
        self.state_discretization = state_discretization
        self.act_discretization = action_discretization
        self.mode = mode
        self._rewards = None
        self._transitions = None
        self._build()

    def vec_step(self, ids_a):
        actions = self.get_action_from_id(ids_a)
        next_obs, rewards, dones, env_info = self._wrapped_env.vec_step(actions)
        if hasattr(self, 'get_discrete_state_from_cont_state'):
            id_next_s, probs = [], []
            for n_o in next_obs:
                id_n_s, p = self.get_discrete_state_from_cont_state(n_o)
                id_next_s.append(id_n_s), probs.append(p)
            id_next_s, probs = np.array(id_next_s).T, np.array(probs).T
        else:
            id_next_s, probs = self.vec_get_discrete_state_from_cont_state(next_obs)
        id_next_s, probs = id_next_s.T, probs.T
        probs = probs/np.expand_dims(np.sum(probs, axis=-1), axis=-1)
        s = probs.cumsum(axis=-1)
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=-1)
        k = (s < r).sum(axis=1)
        id_next_s = id_next_s[range(ids_a.shape[0]), k]
        return id_next_s, rewards, dones, env_info

    def vec_set_state(self, ids_s):
        states = self.get_state_from_id(ids_s)
        self._wrapped_env.vec_set_state(states)

    def vec_reset(self, num_envs):
        obs = self._wrapped_env.vec_reset(num_envs)
        if hasattr(self, 'get_discrete_state_from_cont_state'):
            id_state, probs = [], []
            for o in obs:
                id_s, p = self.get_discrete_state_from_cont_state(o)
                id_state.append(id_s), probs.append(p)
            id_s, probs = np.array(id_state).T, np.array(probs).T
        else:
            id_s, probs = self.vec_get_discrete_state_from_cont_state(obs)
        id_s, probs = id_s.T, probs.T
        probs = probs / np.expand_dims(np.sum(probs, axis=-1), axis=-1)
        s = probs.cumsum(axis=-1)
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=-1)
        k = (s < r).sum(axis=1)
        id_s = id_s[range(id_s.shape[0]), k]
        return id_s

    def step(self, id_a):
        action = self.get_action_from_id(id_a)
        next_obs, reward, done, info = self._wrapped_env.step(action)
        if hasattr(self, 'get_discrete_state_from_cont_state'):
            s, probs = self.get_discrete_state_from_cont_state(next_obs)
        else:
            s, probs = self.vec_get_discrete_state_from_cont_state(next_obs)
        probs = np.array(probs).astype(np.float64)
        idxs = np.random.multinomial(1, probs/np.sum(probs)).astype(np.bool)
        id_next_s = np.array(s)[idxs][0]
        return id_next_s, reward, done, info

    def reset(self):
        obs = self._wrapped_env.reset()
        if hasattr(self, 'get_discrete_state_from_cont_state'):
            s, probs = self.get_discrete_state_from_cont_state(obs)
        else:
            s, probs = self.vec_get_discrete_state_from_cont_state(obs)
        probs = np.array(probs).astype(np.float64)
        idxs = np.random.multinomial(1, probs/np.sum(probs)).astype(np.bool)
        id_s = np.argmax(idxs)
        return id_s

    def get_state_from_id(self, id_s):
        """
        Get continuous state from id
        :param id_s:
        :return:
        """
        if self._disc_state:
            return id_s
        else:
            vec = self.get_coordinates_from_id(id_s)
            return self.state_points[range(self.obs_dims), vec]

    def get_action_from_id(self, id_a):
        """
        Get continous action from id
        :param id_a:
        :return:
        """
        if self._disc_act:
            return id_a
        else:
            vec = self.get_coordinates_from_id(id_a, state=False)
            return self.act_points[range(self.act_dims), vec]

    def get_coordinates_from_id(self, idx, state=True, base=None):
        """
        Get position in the grid from id
        :param idx:
        :param state:
        :param base:
        :return:
        """
        size = self.obs_dims if state else self.act_dims
        if isinstance(idx, Iterable): # probably if it's iterable
            vec = np.zeros((len(idx), size))
        else:
            vec = np.zeros((size,), dtype=np.int)

        num, i,  = idx, 0
        if base is None:
            base = self._state_bins_per_dim if state else self._act_bins_per_dim
        else:
            assert type(base) == int
            base = np.ones((size,), dtype=np.int) * base
        for i in range(size):
            vec[..., i] = num % base[i]
            num = num//base[i]
            i += 1
        return vec.astype(np.int)

    def get_id_from_coordinates(self, vec, state=True):
        """
        Get id from position in the grid
        :param vec:
        :param state:
        :return:
        """
        base_transf = self._state_base_transf if state else self._act_base_transf
        return np.squeeze(np.sum(vec * base_transf, axis=-1).astype(int))

    def get_discretization(self, discretization, low, high):
        """
        Get grid points
        :param discretization:
        :param low:
        :param high:
        :return:
        """
        if type(discretization) is int:
            bins_per_dim = np.ones_like(high, dtype=np.int) * discretization
            points = np.stack([np.linspace(l, h, discretization) for l, h in zip(low, high)])
        else:
            dim = high.shape[0]
            assert len(discretization) == high.shape[0]
            if not isinstance(discretization[0], Iterable):
                discretization = np.array(discretization)
                assert high.shape == discretization.shape
                bins_per_dim = discretization.astype(np.int)
                points = np.ones((dim, np.max(discretization))) * (high[:, None] + 1e3)
                for i, d in enumerate(discretization):
                    points[i, :d] = np.linspace(low[i], high[i], d)

            else:
                bins_per_dim = np.zeros(high.shape, dtype=np.int)
                points = np.ones((dim, np.max(discretization))) * (high + 1e3)
                for i, d in enumerate(discretization):
                    assert (d[i, 0] == low).all() and (d[i, -1] == high).all()
                    bins = len(d)
                    points[i, :bins] = d
                    bins_per_dim[i] = bins

        points = points.astype(np.float32)
        bins_per_dim = bins_per_dim.astype(np.int)
        return points, bins_per_dim

    def _build(self):
        # Create the discrete state space
        env = self._wrapped_env
        if isinstance(env.observation_space, spaces.Discrete):
            self.obs_n = env.observation_space.n
            self._disc_state = True

        else:
            assert self.state_discretization is not None
            self._low_state, self._high_state = np.array(env.observation_space.low),\
                                    np.array(env.observation_space.high)
            self.state_points, self._state_bins_per_dim = self.get_discretization(self.state_discretization,
                                                                                   self._low_state,
                                                                                   self._high_state)
            self.obs_dims = len(self._low_state)
            self.obs_n = int(np.prod(self._state_bins_per_dim))
            self.observation_space = spaces.Discrete(self.obs_n + 1)
            self._state_base_transf = np.cumprod(np.concatenate([[1], self._state_bins_per_dim[:-1]]))
            self._all_coordinates = self.get_coordinates_from_id(np.arange(2 ** self.obs_dims), base=2).T
            self._disc_state = False

        # Create the discrete action space
        if isinstance(env.action_space, spaces.Discrete):
            self.act_n = env.action_space.n
            self._disc_act = True
        else:
            assert self.act_discretization is not None
            self._low_act, self._high_act = np.array(env.action_space.low), \
                                            np.array(env.action_space.high)
            self.act_points, self._act_bins_per_dim = self.get_discretization(self.act_discretization,
                                                                              self._low_act,
                                                                              self._high_act)
            self.act_dims = len(self._low_act)
            self.act_n = int(np.prod(self._act_bins_per_dim))
            self.action_space = spaces.Discrete(self.act_n)
            self._act_base_transf = np.cumprod(np.concatenate([[1], self._act_bins_per_dim[:-1]]))
            self._disc_act = False

        # Creation of reward and transition matrix
        self.rewards = SparseArray(self.obs_n + 1, self.act_n, self.mode, self.obs_dims)
        self.transitions = SparseArray(self.obs_n + 1, self.act_n, self.mode, self.obs_dims)

        if hasattr(self, 'add_transition') or not getattr(env, 'vectorized', False):
            for id_a in range(self.act_n):
                for id_s in range(self.obs_n):
                    self.add_transition(id_s, id_a)
        else:
            id_acts, id_obs = np.mgrid[:self.act_n, :self.obs_n].reshape(2, -1)
            n_points = len(id_acts)
            max_points = int(5e3)
            n_iters = np.ceil(n_points/max_points).astype(np.int)
            for i in range(n_iters):
                self.vec_transitions(id_obs[i * max_points:(i+1) * max_points],
                                      id_acts[i * max_points:(i+1) * max_points])
            if isinstance(env, VectorizeMujocoEnv):
                env.vec_close()

        # Transitions for done
        self.add_done_transitions()


    def __getattr__(self, attr):
        """
        If normalized env does not have the attribute then call the attribute in the wrapped_env
        Args:
            attr: attribute to get

        Returns:
            attribute of the wrapped_env

        """
        # orig_attr = self._wrapped_env.__getattribute__(attr)
        if hasattr(self._wrapped_env, '_wrapped_env'):
            orig_attr = self._wrapped_env.__getattr__(attr)
        else:
            orig_attr = self._wrapped_env.__getattribute__(attr)

        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr


class DiscreteEnv(Env):
    """
    actions: 0 left and 1 right
    """
    def __init__(self, obs_dim, act_dim):
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._states = np.arange(obs_dim)
        self._rewards = np.zeros((obs_dim, act_dim, obs_dim), dtype=np.float32)
        self._transitions = np.zeros((obs_dim, act_dim, obs_dim))
        self._build_rewards()
        self._build_transitions()
        assert (np.sum(self._transitions, axis=-1) == np.ones((obs_dim, act_dim))).all
        self._state = 0

        self.observation_space = spaces.Discrete(obs_dim)
        self.action_space = spaces.Discrete(act_dim)

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def rewards(self):
        return self._rewards.copy()

    @property
    def states(self):
        return self._states.copy()

    @property
    def transitions(self):
        return self._transitions.copy()

    def _build_transitions(self):
        raise NotImplementedError

    def _build_rewards(self):
        raise NotImplementedError


class VectorizeMujocoEnv(object):
    def __init__(self, env, n_parallel):
        self._wrapped_env = env
        self.discount = env.discount
        self.n_parallel = int(n_parallel)
        self.vectorized = True
        self._created = False
        self._num_envs = -1
        self.ps = []
        self.remotes = []

    def vec_close(self):
        for remote in self.remotes:
            remote.send(('close', None))

        for p in self.ps:
            p.terminate()

    def _create_envs(self):
        self.vec_close()

        self.envs_per_proc = [int(self._num_envs/self.n_parallel)] * self.n_parallel
        if self._num_envs % self.n_parallel != 0:
            self.envs_per_proc[-1] += self._num_envs % self.n_parallel
        self._envs_idxs = np.cumsum([0] + self.envs_per_proc)

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_parallel)])
        self.ps = [
            Process(target=worker, args=(work_remote, remote, pickle.dumps(self._wrapped_env), n_envs))
            for (work_remote, remote, n_envs) in zip(self.work_remotes, self.remotes, self.envs_per_proc)]  # Why pass work remotes?

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self._created = True

    def _get_chunks(self, x):
        return [x[self._envs_idxs[i]:self._envs_idxs[i+1]] for i in range(self.n_parallel)]

    def vec_set_state(self, states):
        num_envs = len(states)
        if num_envs != self._num_envs:
            self._num_envs = num_envs
            self._create_envs()
        states_per_proc = self._get_chunks(states)
        for remote, task in zip(self.remotes, states_per_proc):
            remote.send(('set_state', task))
        [remote.recv() for remote in self.remotes]

    def vec_reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return sum([remote.recv() for remote in self.remotes], [])

    def vec_step(self, actions):
        assert len(actions) == self._num_envs

        # split list of actions in list of list of actions per meta tasks
        actions_per_meta_task = self._get_chunks(actions)

        # step remote environments
        for remote, action_list in zip(self.remotes, actions_per_meta_task):
            remote.send(('step', action_list))

        results = [remote.recv() for remote in self.remotes]

        obs, rewards, dones, env_infos = map(lambda x: sum(x, []), zip(*results))

        return np.array(obs), np.array(rewards), np.array(dones), env_infos

    def __getattr__(self, attr):
        """
        If normalized env does not have the attribute then call the attribute in the wrapped_env
        Args:
            attr: attribute to get

        Returns:
            attribute of the wrapped_env

        """
        # orig_attr = self._wrapped_env.__getattribute__(attr)
        if hasattr(self._wrapped_env, '_wrapped_env'):
            orig_attr = self._wrapped_env.__getattr__(attr)
        else:
            orig_attr = self._wrapped_env.__getattribute__(attr)

        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr


def worker(remote, parent_remote, env_pickle, n_envs):
    """
    Instantiation of a parallel worker for collecting samples. It loops continually checking the task that the remote
    sends to it.

    Args:
        remote (multiprocessing.Connection):
        parent_remote (multiprocessing.Connection):
        env_pickle (pkl): pickled environment
        n_envs (int): number of environments per worker
        max_path_length (int): maximum path length of the task
        seed (int): random seed for the worker
    """
    parent_remote.close()

    envs = [pickle.loads(env_pickle) for _ in range(n_envs)]

    while True:
        # receive command and data from the remote
        cmd, data = remote.recv()

        # do a step in each of the environment of the worker
        if cmd == 'step':
            all_results = [env.step(a) for (a, env) in zip(data, envs)]
            obs, rewards, dones, infos = map(list, zip(*all_results))
            for i in range(n_envs):
                if dones[i]:
                    dones[i] = True
                    obs[i] = envs[i].reset()
            remote.send((obs, rewards, dones, infos))

        # reset all the environments of the worker
        elif cmd == 'reset':
            obs = [env.reset() for env in envs]
            remote.send(obs)

        # set the specified task for each of the environments of the worker
        elif cmd == 'set_state':
            for state, env in zip(data, envs):
                env.set_state(state)
            remote.send(None)

        # close the remote and stop the worker
        elif cmd == 'close':
            remote.close()
            break

        else:
            raise NotImplementedError
