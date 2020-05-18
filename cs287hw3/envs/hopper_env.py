import numpy as np
from gym import utils
import copy
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.hopper import HopperEnv


class HopperModEnv(HopperEnv, mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        self.perturb_joints = True
        self.components = np.array(['thigh', 'leg', 'foot'])
        self.affected_part = 'thigh'
        self.count = 0
        mujoco_env.MujocoEnv.__init__(self, "hopper.xml", 8)
        utils.EzPickle.__init__(self)

        self.init_geom_rgba = self.model.geom_rgba.copy()
        self.dx = self.observation_space.shape[0]
        self.du = self.action_space.shape[0]
        self.H = 50

    def step(self, a):
        self.count += 1
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        reward = (posafter - posbefore) / self.dt
        done = False
        ob = self._get_obs()
        return ob, -reward, done, {}

    def set_state(self, state):
        nq, nv = self.model.nq, self.model.nv
        self.sim.reset()
        qpos = copy.deepcopy(self.init_qpos)
        qvel = copy.deepcopy(self.init_qvel)
        
        qpos[1:6] = state[:nq-1]
        qvel[:6] = state[nq-1:]
      
        mujoco_env.MujocoEnv.set_state(self, qpos, qvel)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:6],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):

        self.affected_part = self.components[np.random.randint(0,3)]
        self.count = 0
        qpos = self.init_qpos
        qvel = self.init_qvel
        mujoco_env.MujocoEnv.set_state(self, qpos, qvel)

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
