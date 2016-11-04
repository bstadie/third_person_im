from rllab.envs.base import Step
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
import numpy as np
import math
from rllab.mujoco_py import glfw
from rllab.envs.mujoco.mujoco_env import MujocoEnv


class ReacherEnv(MujocoEnv, Serializable):

    FILE = 'reacher.xml'

    def __init__(self, *args, **kwargs):
        super(ReacherEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.forward_dynamics(a)
        next_obs = self.get_current_obs()
        return Step(next_obs, reward, False)
        #done = False
        #return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid=0

    def reset_mujoco(self, init_state=None):
        qpos = np.random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos.flat
        while True:
            self.goal = np.random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2: break
        qpos[-2:] = self.goal
        qvel = self.init_qvel.flat + np.random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl
        return self.get_current_obs()

    def get_current_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])