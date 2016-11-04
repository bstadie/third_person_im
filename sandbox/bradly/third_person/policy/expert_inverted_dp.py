from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.bradly.third_person.envs.inverted_dp import InvertedPendulumEnv
from rllab.sampler.utils import rollout
import pickle
import tensorflow as tf


def generate_expert_dp():
    env = TfEnv(normalize(InvertedPendulumEnv()))
    policy = GaussianMLPPolicy(
        name="expert_policy",
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(64, 64),
        std_hidden_sizes=(64, 64),
        adaptive_std=True,

    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=65,
        discount=0.995,
        step_size=0.01,
        optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
        gae_lambda=0.97,

    )

    with tf.Session() as sess:
        algo.train(sess=sess)

        t = rollout(env=env, agent=policy, max_path_length=100, animated=True)
        print(sum(t['rewards']))
        with open('expert_dp.pickle', 'wb') as handle:
            pickle.dump(policy, handle)
        while True:
            rollout(env=env, agent=policy, max_path_length=100, animated=True)


def load_expert_inverted_dp():
    with open('expert_dp.pickle', 'rb') as handle:
        policy = pickle.load(handle)

    return policy


if __name__ == '__main__':
    generate_expert_dp()
    #with tf.Session() as sess:
    #    env = TfEnv(normalize(InvertedPendulumEnv()))
    #    expert = load_expert_reacher(env, sess)
    #    while True:
    #        t = rollout(env=env, agent=expert, max_path_length=100, animated=True)
