from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.bradly.third_person.envs.conopt_particle_env_two import ConoptParticleEnv as ConoptParticleEnvTwo
from sandbox.bradly.third_person.envs.conopt_particle_env import ConoptParticleEnv
from sandbox.bradly.third_person.policy.conopt_particle_tracking_policy import ConoptParticleTrackingPolicy
from sandbox.bradly.third_person.policy.random_policy import RandomPolicy
from sandbox.bradly.third_person.algos.cyberpunk_trainer import CyberPunkTrainer
from sandbox.bradly.third_person.discriminators.discriminator import DomainConfusionVelocityDiscriminator

import tensorflow as tf


expert_env = TfEnv(ConoptParticleEnv())
novice_env = TfEnv(ConoptParticleEnvTwo())
expert_fail_pol = RandomPolicy(expert_env.spec)

policy = GaussianMLPPolicy(
    name="policy",
    env_spec=novice_env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)

baseline = LinearFeatureBaseline(env_spec=expert_env.spec)

algo = TRPO(
    env=novice_env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))

)

with tf.Session() as sess:

    algo.n_itr = 0
    algo.start_itr = 0
    algo.train(sess=sess)

    im_size = 50
    im_channels = 3

    dim_input = [im_size, im_size, im_channels]

    disc = DomainConfusionVelocityDiscriminator(input_dim=dim_input, output_dim_class=2, output_dim_dom=2,
                                                tf_sess=sess)

    expert_policy = ConoptParticleTrackingPolicy(expert_env)

    #from rllab.sampler.utils import rollout
    #while True:
    #        t = rollout(env=expert_env, agent=expert_policy, max_path_length=50, animated=True)

    algo.n_itr = 40
    trainer = CyberPunkTrainer(disc=disc, novice_policy_env=novice_env, expert_fail_pol=expert_fail_pol,
                               expert_env=expert_env, novice_policy=policy,
                               novice_policy_opt_algo=algo, expert_success_pol=expert_policy,
                               im_width=im_size, im_height=im_size, im_channels=im_channels,
                               tf_sess=sess, horizon=50)

    iterations = 10

    for iter_step in range(0, iterations):
        trainer.take_iteration(n_trajs_cost=12000, n_trajs_policy=12000)

    trainer.log_and_finish()

