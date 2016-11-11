from rllab.envs.base import Step
from rllab.core.serializable import Serializable
import numpy as np
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.mujoco_py import MjViewer



class ReacherEnv(MujocoEnv, Serializable):

    FILE = 'reacher.xml'

    def __init__(self, *args, **kwargs):
        super(ReacherEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
        self.goal = None

    def step(self, a):
        #print(self.viewer.cam.__dict__)
        #print(self.viewer.cam.distance)
        #print(self.viewer.cam.azimuth)
        #print(self.viewer.cam.elevation)
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl =  0 #- np.square(a).sum()
        #reward_close = 0.01*math.log(-reward_dist)
        reward = reward_dist + reward_ctrl #+ reward_close
        self.forward_dynamics(a)
        next_obs = self.get_current_obs()
        return Step(next_obs, reward, False)
        #done = False
        #return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid=0

    def get_viewer(self):
        if self.viewer is None:
            self.viewer = MjViewer(init_width=25, init_height=25)
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer.cam.elevation = -20.59999990463257
        return self.viewer

    def reset_mujoco(self, init_state=None):
        qpos = np.random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos.flat
        #while True:
        #    self.goal = np.random.uniform(low=-.2, high=.2, size=2)
        #    if np.linalg.norm(self.goal) < 2: break
        self.goal = np.array([0.1, 0.1])
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

    reset_trial = reset_mujoco  # shortcut for compatibility.
