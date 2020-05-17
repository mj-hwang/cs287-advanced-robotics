import numpy as np
import copy
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
import os

class CheetahModEnv(HalfCheetahEnv, mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        self.perturb_joints = True
        self.count = 0
        mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 4)
        utils.EzPickle.__init__(self)
        self.perturb_idx=0
        self.init_geom_rgba = self.model.geom_rgba.copy()


    def f_sim(self, x0, u, dt, rollout=False, perturb=.01):
        nq, nv = self.model.nq, self.model.nv
        self.sim.reset()
        qpos = copy.deepcopy(self.init_qpos)
        qvel = copy.deepcopy(self.init_qvel)
        
        qpos[:] = x0[:nq]
        qvel[:] = x0[nq:]
      
        self.set_state(qpos, qvel)
        if rollout:
            self.step(u, perturb=perturb)
        else:
            self.perturb_joints = False
            self.step(u)
            self.perturb_joints = True
        return np.concatenate([
            self.sim.data.qpos.flat[:],
            self.sim.data.qvel.flat[:]
        ])
    
    def step(self, a, perturb=.01):
        self.count += 1
        if self.perturb_joints and self.count%5==0:
            self.perturb_idx = np.random.randint(0,6)
            a[self.perturb_idx] += np.random.choice(np.array([-1*perturb,perturb]))
            model_id = self.model.geom_names.index(self.model.joint_names[self.perturb_idx+3])
            geom_rgba = self.init_geom_rgba.copy()
            geom_rgba[model_id] = [0, 1, 1 ,1]
            self.model.geom_rgba[:] = geom_rgba
        else:
            if self.count > 1 and self.count%8==0:
                model_id = self.model.geom_names.index(self.model.joint_names[self.perturb_idx+3])
                geom_rgba = self.init_geom_rgba.copy()
                self.model.geom_rgba[:] = geom_rgba
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(a).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)
