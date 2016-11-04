from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.mujoco.reacher import ReacherEnv


import numpy as np


class RandomPolicy:
    def __init__(self, env_sepc):
        self.action_size = env_sepc.action_space.flat_dim

    def get_action(self, o=None):
        return np.random.randn(self.action_size), dict()

    def reset(self):
        pass


#env = TfEnv(ReacherEnv())
#from sandbox.rocky.analogy.utils import unwrap
#env_two = unwrap(env)
#rp = RandomPolicy(env.spec)
#print(rp.get_action())
