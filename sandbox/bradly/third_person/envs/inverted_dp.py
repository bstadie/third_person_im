from rllab.envs.base import Step
from rllab.core.serializable import Serializable
import numpy as np
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.mujoco_py import MjViewer


class InvertedPendulumEnv(MujocoEnv, Serializable):

    FILE = 'inverted_pend.xml'

    def __init__(self, *args, **kwargs):
        super(InvertedPendulumEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def step(self, a):
        reward = 1.0
        self.forward_dynamics(a)
        ob = self.get_current_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        #done = not notdone
        done = False
        if not notdone:
            reward = 0
        return Step(ob, reward, done)

    def get_viewer(self):
        if self.viewer is None:
            self.viewer = MjViewer(init_width=25, init_height=25)
            self.viewer.start()
            self.viewer.set_model(self.model)
            #self.viewer.cam.elevation = -42.59999990463257
        return self.viewer

    def reset_mujoco(self, init_state=None):
        qpos = self.init_qpos + np.random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + np.random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl
        return self.get_current_obs()

    reset_trial = reset_mujoco

    def get_current_obs(self):
        return np.concatenate([self.model.data.qpos, self.model.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid=0
        v.cam.distance = v.model.stat.extent
