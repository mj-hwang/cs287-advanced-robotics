import numpy as np
from gym import utils
import copy
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.hopper import HopperEnv
import os

class HopperModEnv(HopperEnv, mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        self.perturb_joints = True
        self.components = np.array(['thigh', 'leg', 'foot'])
        self.affected_part = 'thigh'
        self.count = 0
        mujoco_env.MujocoEnv.__init__(self, "hopper.xml", 4)
        utils.EzPickle.__init__(self)

        self.init_geom_rgba = self.model.geom_rgba.copy()

    def step(self, a, perturb=.01):
        self.count += 1
        if self.perturb_joints and self.count%2==0:
            self.affected_part = self.components[np.random.randint(0,3)]
            perturb_idx = np.where(self.components == self.affected_part)[0][0]
            a[perturb_idx] += np.random.choice(np.array([-1*perturb,perturb]))
            model_id = self.model.geom_name2id(self.affected_part + '_geom')
            geom_rgba = self.init_geom_rgba.copy()
            geom_rgba[model_id] = [1, 0, 0 ,1]
            self.model.geom_rgba[:] = geom_rgba
        else:
            if self.count > 1 and self.count%3==0:
                model_id = self.model.geom_name2id(self.affected_part + '_geom')
                geom_rgba = self.init_geom_rgba.copy()
                self.model.geom_rgba[:] = geom_rgba
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()


        return ob, reward, done, {}


    def f_sim(self, x0, u, dt, rollout=False, perturb=.01):
        nq, nv = self.model.nq, self.model.nv
        self.sim.reset()
        qpos = copy.deepcopy(self.init_qpos)
        qvel = copy.deepcopy(self.init_qvel)
        
        qpos[1:6] = x0[:nq-1]
        qvel[:6] = x0[nq-1:]
      
        self.set_state(qpos, qvel)
        if rollout:
            self.step(u, perturb=perturb)
        else:
            self.perturb_joints = False
            self.step(u)
            self.perturb_joints = True
        return np.concatenate([
            self.sim.data.qpos.flat[1:6],
            self.sim.data.qvel.flat[:6]
        ])


    def _get_obs(self):

        return np.concatenate([
            self.sim.data.qpos.flat[1:6],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):

        self.affected_part = self.components[np.random.randint(0,3)]
        self.count = 0
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
