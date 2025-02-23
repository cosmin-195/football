from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import argparse
import gfootball.env as football_env
import gym
import ray
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

parser = argparse.ArgumentParser()

parser.add_argument('--num-agents', type=int, default=3)
parser.add_argument('--num-policies', type=int, default=3)
parser.add_argument('--num-iters', type=int, default=100000)
parser.add_argument('--simple', action='store_true')


class RllibGFootball(MultiAgentEnv):
    """An example of a wrapper for GFootball to make it compatible with rllib."""

    def __init__(self, num_agents):
        self.env = football_env.create_environment(
            env_name='test_example_multiagent', stacked=False,
            logdir=os.path.join(R'C:\Users\Legion\Desktop\gfootball_dumps', 'rllib_test'),
            write_goal_dumps=True, write_full_episode_dumps=True, render=False,
            dump_frequency=1,
            number_of_left_players_agent_controls=num_agents,
            channel_dimensions=(42, 42))
        self.action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low[0],
            high=self.env.observation_space.high[0],
            dtype=self.env.observation_space.dtype)
        self.num_agents = num_agents

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for x in range(self.num_agents):
            if self.num_agents > 1:
                obs['agent_%d' % x] = original_obs[x]
            else:
                obs['agent_%d' % x] = original_obs
        return obs

    def step(self, action_dict):
        actions = []
        for key, value in sorted(action_dict.items()):
            actions.append(value)
        # observation, reward, done, info
        o, r, d, i = self.env.step(actions)
        rewards = {}
        obs = {}
        infos = {}
        for pos, key in enumerate(sorted(action_dict.keys())):
            infos[key] = i
            if self.num_agents > 1:
                rewards[key] = r[pos]
                obs[key] = o[pos]
            else:
                rewards[key] = r
                obs[key] = o
        dones = {'__all__': d}
        return obs, rewards, dones, infos


if __name__ == '__main__':
    args = parser.parse_args()
    ray.init(num_gpus=0, local_mode=True)

    # Simple environment with `num_agents` independent players
    register_env('gfootball', lambda _: RllibGFootball(args.num_agents))
    single_env = RllibGFootball(args.num_agents)
    obs_space = single_env.observation_space
    act_space = single_env.action_space


    def gen_policy(_):
        return (None, obs_space, act_space, {})


    def policy_map(agent_id, episode, worker, **kwargs):
        return policy_ids[int(agent_id[6:])]


    # Setup PPO with an ensemble of `num_policies` different policies
    policies = {
        'policy_{}'.format(i): gen_policy(i) for i in range(args.num_policies)
    }
    policy_ids = list(policies.keys())

    tune.run(
        'A2C',
        stop={'training_iteration': args.num_iters},
        checkpoint_freq=50,
        config={
            'env': 'gfootball',
            'lambda': 0.95,
            'train_batch_size': 2000,
            'num_workers': 12,
            'num_envs_per_worker': 1,
            'batch_mode': 'truncate_episodes',
            'observation_filter': 'NoFilter',
            'num_gpus': 0,
            'lr': 2e-4,
            'log_level': 'DEBUG',
            'simple_optimizer': args.simple,
            'multiagent': {
                'policies': policies,
                'policy_mapping_fn': (lambda agent_id, episode, **kwargs: policy_ids[int(agent_id[6:])]),
            },
        },
        resume=True,
        local_dir=R'C:\Users\Legion\Desktop\gfootball_dumps\ray'
    )

