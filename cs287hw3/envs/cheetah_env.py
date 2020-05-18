import copy
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv


class CheetahModEnv(HalfCheetahEnv, mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        self.perturb_joints = True
        self.count = 0
        mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 8)
        utils.EzPickle.__init__(self)
        self.H = 30
        self.du = self.action_space.shape[0]
        self.dx = self.observation_space.shape[0]

    def step(self, a):
        self.count += 1
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward = (xposafter - xposbefore)/self.dt
        done = False
        return ob, -reward, done, dict()

    def set_state(self, state):
        nq, nv = self.model.nq, self.model.nv
        self.sim.reset()
        qpos = copy.deepcopy(self.init_qpos)

        qpos[1:nq] = state[:nq - 1]
        qvel = state[nq - 1:]

        mujoco_env.MujocoEnv.set_state(self, qpos, qvel)

    def reset_model(self):
        mujoco_env.MujocoEnv.set_state(self, self.init_qpos, self.init_qvel)
        return self._get_obs()
