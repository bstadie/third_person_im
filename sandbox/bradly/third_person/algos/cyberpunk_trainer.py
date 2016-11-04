import numpy as np
from rllab.misc import tensor_utils
from rllab.sampler.base import BaseSampler
from sandbox.rocky.analogy.utils import unwrap


class CyberPunkTrainer:
    def __init__(self, disc, novice_policy_env, expert_env, novice_policy, novice_policy_opt_algo,
                 expert_success_pol, expert_fail_pol, im_width, im_height, im_channels=3, tf_sess=None,
                 horizon=None):

        #from rllab.sampler.utils import rollout
        #while True:
        #        t = rollout(env=expert_env, agent=expert_success_pol, max_path_length=50, animated=True)

        self.sess = tf_sess

        self.novice_policy_env = unwrap(novice_policy_env)
        self.expert_env = unwrap(expert_env)

        self.expert_success_pol = expert_success_pol
        self.expert_fail_pol = expert_fail_pol
        self.novice_policy = novice_policy
        self.novice_policy_training_algo = novice_policy_opt_algo

        self.batch_size = 32
        self.horizon = horizon
        self.im_height = im_height
        self.im_width = im_width
        self.im_channels = im_channels
        self.iteration = 0

        self.disc = disc

        e_10 = np.zeros((2,))
        e_10[0] = 1
        self.expert_basis = e_10
        e_01 = np.zeros((2,))
        e_01[1] = 1
        self.novice_basis = e_01

        #expert_fails = 20
        #self.expert_fail_data = self.collect_trajs_for_cost(expert_fails, self.expert_fail_pol, self.expert_env,
        #                                                    dom=self.expert_basis, cls=self.novice_basis)
        self.expert_fail_data = None
        self.sampler = BaseSampler(self.novice_policy_training_algo)

        self.gan_rew_means = []
        self.true_rew_means = []

    def collect_trajs_for_cost(self, n_trajs, pol, env, dom, cls):
        paths = []
        #print(n_trajs)
        for iter_step in range(0, n_trajs):
            paths.append(self.cyberpunk_rollout(agent=pol, env=env, max_path_length=self.horizon,
                                                reward_extractor=None))

        data_matrix = tensor_utils.stack_tensor_list([p['im_observations'] for p in paths])
        class_matrix = np.tile(cls, (n_trajs, self.horizon, 1))
        dom_matrix = np.tile(dom, (n_trajs, self.horizon, 1))

        #data_matrix = np.zeros(shape=(n_trajs, self.horizon, self.im_height, self.im_width, self.im_channels))
        #class_matrix = np.zeros(shape=(n_trajs, self.horizon, 2))
        #dom_matrix = np.zeros(shape=(n_trajs, self.horizon, 2))
        #for path, path_step in zip(paths, range(0, len(paths))):
        #    for sub_path, time_step in zip(path['im_observations'], range(0, self.horizon)):
        #        data_matrix[path_step, time_step, :, :, :] = sub_path
        #        class_matrix[path_step, time_step, :] = path['class']
        #        dom_matrix[path_step, time_step, :] = path['dom']

        return dict(data=data_matrix, classes=class_matrix, domains=dom_matrix)

    def collect_trajs_for_policy(self, n_trajs, pol, env):
        paths = []
        for iter_step in range(0, n_trajs):
            paths.append(self.cyberpunk_rollout(agent=pol, env=env, max_path_length=self.horizon,
                                                reward_extractor=self.disc))
        return paths

    def take_iteration(self, n_trajs_cost, n_trajs_policy):
        expert_data = self.collect_trajs_for_cost(n_trajs=n_trajs_cost, pol=self.expert_success_pol,
                                                  env=self.expert_env,
                                                  dom=self.expert_basis, cls=self.expert_basis)
        on_policy_data = self.collect_trajs_for_cost(n_trajs=n_trajs_cost, pol=self.novice_policy,
                                                     env=self.novice_policy_env,
                                                     dom=self.novice_basis, cls=self.novice_basis)
        self.expert_fail_data = self.collect_trajs_for_cost(n_trajs_cost, self.expert_fail_pol, self.expert_env,
                                                            dom=self.expert_basis, cls=self.novice_basis)

        training_data_one, training_data_two, training_doms, training_classes = self.shuffle_to_training_data(expert_data,
                                                                                                              on_policy_data,
                                                                                                              self.expert_fail_data)

        self.train_cost(training_data_one, training_data_two, training_classes, training_doms, n_epochs=2)

        policy_training_paths = self.collect_trajs_for_policy(n_trajs_policy, pol=self.novice_policy,
                                                              env=self.novice_policy_env)
        gan_rew_mean = np.mean(np.array([path['rewards'] for path in policy_training_paths]))
        gan_rew_std = np.std(np.array([path['rewards'] for path in policy_training_paths]))
        print('on policy GAN reward is ' + str(gan_rew_mean))
        true_rew_mean = np.mean(np.array([sum(path['true_rewards']) for path in policy_training_paths]))
        print('on policy True reward is ' + str(true_rew_mean))

        self.true_rew_means.append(true_rew_mean)
        self.gan_rew_means.append(gan_rew_mean)

        #for path in policy_training_paths:
        #    path['rewards'] = (path['rewards'] - gan_rew_mean)/gan_rew_std
        policy_training_samples = self.sampler.process_samples(itr=self.iteration, paths=policy_training_paths)
        self.novice_policy_training_algo.optimize_policy(itr=self.iteration, samples_data=policy_training_samples)

        self.iteration += 1
        print(self.iteration)

    def log_and_finish(self):
        print('true rews were ' + str(self.true_rew_means))
        print('gan rews were ' + str(self.gan_rew_means))
        #import pickle
        #pickle

    def train_cost(self, data_one, data_two, classes, domains, n_epochs):
        for iter_step in range(0, n_epochs):
            batch_losses = []
            lab_acc = []
            for batch_step in range(0, data_one.shape[0], self.batch_size):

                data_batch_zero = data_one[batch_step: batch_step+self.batch_size]
                data_batch_one = data_two[batch_step: batch_step+self.batch_size]
                data_batch = [data_batch_zero, data_batch_one]

                classes_batch = classes[batch_step: batch_step+self.batch_size]
                domains_batch = domains[batch_step: batch_step+self.batch_size]
                targets = dict(classes=classes_batch, domains=domains_batch)

                batch_losses.append(self.disc.train(data_batch, targets))
                lab_acc.append(self.disc.get_lab_accuracy(data_batch, targets['classes']))
            print('loss is ' + str(np.mean(np.array(batch_losses))))
            print('acc is ' + str(np.mean(np.array(lab_acc))))

    def shuffle_to_training_data(self, expert_data, on_policy_data, expert_fail_data):
        data = np.vstack([expert_data['data'], on_policy_data['data'], expert_fail_data['data']])
        classes = np.vstack([expert_data['classes'], on_policy_data['classes'], expert_fail_data['classes']])
        domains = np.vstack([expert_data['domains'], on_policy_data['domains'], expert_fail_data['domains']])

        sample_range = data.shape[0]*data.shape[1]
        all_idxs = np.random.permutation(sample_range)

        t_steps = data.shape[1]

        data_matrix = np.zeros(shape=(sample_range, self.im_height, self.im_width, self.im_channels))
        data_matrix_two = np.zeros(shape=(sample_range, self.im_height, self.im_width, self.im_channels))
        class_matrix = np.zeros(shape=(sample_range, 2))
        dom_matrix = np.zeros(shape=(sample_range, 2))
        for one_idx, iter_step in zip(all_idxs, range(0, sample_range)):
            traj_key = np.floor(one_idx/t_steps)
            time_key = one_idx % t_steps
            time_key_plus_one = min(time_key + 3, t_steps-1)
            data_matrix[iter_step, :, :, :] = data[traj_key, time_key, :, :, :]
            data_matrix_two[iter_step, :, :, :] = data[traj_key, time_key_plus_one, :, :, :]
            class_matrix[iter_step, :] = classes[traj_key, time_key, :]
            dom_matrix[iter_step, :] = domains[traj_key, time_key, :]
        return data_matrix, data_matrix_two, dom_matrix, class_matrix

    def cyberpunk_rollout(self, agent, env, max_path_length, reward_extractor=None, animated=True, speedup=1):
        observations = []
        im_observations = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []
        #o = env.reset()
        o = env.reset_trial()
        path_length = 0

        if animated:
            env.render()
        else:
            env.render(mode='robot')

        while path_length < max_path_length:
            a, agent_info = agent.get_action(o)
            next_o, r, d, env_info = env.step(a)
            observations.append(env.observation_space.flatten(o))
            rewards.append(r)
            actions.append(env.action_space.flatten(a))
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            path_length += 1
            if d:
                break
            o = next_o
            if animated:
                im = env.render()
                im_observations.append(im)
            else:
                im = env.render(mode='robot')
                im_observations.append(im)
                #timestep = 0.05
                #time.sleep(timestep / speedup)
        if animated:
            env.render(close=True)

        im_observations = tensor_utils.stack_tensor_list(im_observations)

        observations = tensor_utils.stack_tensor_list(observations)

        if reward_extractor is not None:
            true_rewards = tensor_utils.stack_tensor_list(rewards)
            obs_pls_three = np.copy(im_observations)
            for iter_step in range(0, obs_pls_three.shape[0]):  # cant figure out how to do this with indexing.
                idx_plus_three = min(iter_step+3, obs_pls_three.shape[0]-1)
                obs_pls_three[iter_step, :, :, :] = im_observations[idx_plus_three, :, :, :]
            rewards = reward_extractor.get_reward(data=[im_observations, obs_pls_three], softmax=True)[:, 0]  # this is the prob of being an expert.
            #print(rewards)
        else:
            rewards = tensor_utils.stack_tensor_list(rewards)
            true_rewards = rewards

        return dict(
            observations=observations,
            im_observations=im_observations,
            actions=tensor_utils.stack_tensor_list(actions),
            rewards=rewards,
            true_rewards=true_rewards,
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
        )

