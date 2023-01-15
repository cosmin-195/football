from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from collections import OrderedDict
from typing import Dict, List, Optional

import gym
import ray
from gym.spaces import Tuple
from ray import tune
from ray.tune.registry import register_env
import gfootball.env as football_env
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--num-agents', type=int, default=3)
parser.add_argument('--num-policies', type=int, default=1)
parser.add_argument('--num-iters', type=int, default=100000)
parser.add_argument('--simple', action='store_true')
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.typing import AgentID, MultiAgentDict, MultiEnvDict


class RllibGFootball(MultiAgentEnv):

    def __init__(self, num_agents):
        super().__init__()
        self.env = football_env.create_environment(
            env_name='test_example_multiagent', stacked=True,
            logdir=os.path.join(R'C:\Users\Legion\Desktop\gfootball_dumps', 'rllib_test'),
            write_goal_dumps=True, write_full_episode_dumps=True, render=False,
            dump_frequency=0,
            number_of_left_players_agent_controls=num_agents,
            channel_dimensions=(42, 42))
        self.action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
        self.observation_space = gym.spaces.Dict(
            {"obs": gym.spaces.Box(
                low=self.env.observation_space.low[0],
                high=self.env.observation_space.high[0],
                dtype=np.uint8),
                ENV_STATE: gym.spaces.Box(
                    low=self.env.observation_space.low[0],
                    high=self.env.observation_space.high[0],
                    dtype=np.uint8)}
        )
        self.num_agents = num_agents
        self._agent_ids = {"agent_0", "agent_1", "agent_2"}

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        state = original_obs[0]
        for x in range(self.num_agents):
            if self.num_agents > 1:
                obs['agent_%d' % x] = {"obs": original_obs[x], ENV_STATE: state}
            else:
                obs['agent_%d' % x] = original_obs
        return obs

    def step(self, action_dict):
        # print(action_dict)
        actions = []
        for key, value in sorted(action_dict.items()):
            actions.append(value)
        # observations, rewards, dones, infos
        o, r, d, i = self.env.step(actions)
        rewards = {}
        obs = {}
        infos = {}
        state = o[0]
        for pos, key in enumerate(sorted(action_dict.keys())):
            infos[key] = i
            if self.num_agents > 1:
                rewards[key] = r[pos]
                obs[key] = {"obs": o[pos], ENV_STATE:state}
            else:
                rewards[key] = r
                obs[key] = o
        dones = {'__all__': d}
        return obs, rewards, dones, infos

    def observation_space_sample(self, agent_ids: list = None) -> MultiEnvDict:
        print("called obs space sample")
        samples = {}
        if agent_ids is None:
            agent_ids = ["agent_%d" % i for i in range(self.num_agents)]
        for agent_id in agent_ids:
            samples[agent_id] = self.observation_space.sample()
        return samples

    def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        print("called action space sample")
        samples = {}
        if agent_ids is None:
            agent_ids = ["agent_%d" % i for i in range(self.num_agents)]
        for agent_id in agent_ids:
            samples[agent_id] = self.action_space.sample()
        return samples


# info key for the individual rewards of an agent, for example:
# info: {
#   group_1: {
#      _group_rewards: [5, -1, 1],  # 3 agents in this group
#   }
# }
GROUP_REWARDS = "_group_rewards"

# info key for the individual infos of an agent, for example:
# info: {
#   group_1: {
#      _group_infos: [{"foo": ...}, {}],  # 2 agents in this group
#   }
# }
GROUP_INFO = "_group_info"


@DeveloperAPI
class GroupAgentsWrapperFootball(MultiAgentEnv):
    """Wraps a MultiAgentEnv environment with agents grouped as specified.

    See multi_agent_env.py for the specification of groups.

    This API is experimental.
    """

    def __init__(
            self,
            env: MultiAgentEnv,
            groups: Dict[str, List[AgentID]],
            obs_space: Optional[gym.Space] = None,
            act_space: Optional[gym.Space] = None,
    ):
        """Wrap an existing MultiAgentEnv to group agent ID together.

        See `MultiAgentEnv.with_agent_groups()` for more detailed usage info.

        Args:
            env: The env to wrap and whose agent IDs to group into new agents.
            groups: Mapping from group id to a list of the agent ids
                of group members. If an agent id is not present in any group
                value, it will be left ungrouped. The group id becomes a new agent ID
                in the final environment.
            obs_space: Optional observation space for the grouped
                env. Must be a tuple space. If not provided, will infer this to be a
                Tuple of n individual agents spaces (n=num agents in a group).
            act_space: Optional action space for the grouped env.
                Must be a tuple space. If not provided, will infer this to be a Tuple
                of n individual agents spaces (n=num agents in a group).
        """
        super().__init__()
        self.env = env
        # Inherit wrapped env's `_skip_env_checking` flag.
        if hasattr(self.env, "_skip_env_checking"):
            self._skip_env_checking = self.env._skip_env_checking
        self.groups = groups
        self.agent_id_to_group = {}
        for group_id, agent_ids in groups.items():
            for agent_id in agent_ids:
                if agent_id in self.agent_id_to_group:
                    raise ValueError(
                        "Agent id {} is in multiple groups".format(agent_id)
                    )
                self.agent_id_to_group[agent_id] = group_id
        if obs_space is not None:
            self.observation_space = obs_space
        if act_space is not None:
            self.action_space = act_space
        for group_id in groups.keys():
            self._agent_ids.add(group_id)

    def seed(self, seed=None):
        if not hasattr(self.env, "seed"):
            # This is a silent fail. However, OpenAI gyms also silently fail
            # here.
            return

        self.env.seed(seed)

    def reset(self):
        obs = self.env.reset()
        return self._group_items(obs)

    def step(self, action_dict):
        # Ungroup and send actions
        action_dict = self._ungroup_items(action_dict)
        obs, rewards, dones, infos = self.env.step(action_dict)

        # Apply grouping transforms to the env outputs
        obs = self._group_items(obs)
        rewards = self._group_items(rewards, agg_fn=lambda gvals: list(gvals.values()))
        dones = self._group_items(dones, agg_fn=lambda gvals: all(gvals.values()))
        infos = self._group_items(
            infos, agg_fn=lambda gvals: {GROUP_INFO: list(gvals.values())}
        )

        # Aggregate rewards, but preserve the original values in infos
        for agent_id, rew in rewards.items():
            if isinstance(rew, list):
                rewards[agent_id] = sum(rew)
                if agent_id not in infos:
                    infos[agent_id] = {}
                infos[agent_id][GROUP_REWARDS] = rew

        return obs, rewards, dones, infos

    def _ungroup_items(self, items):
        out = {}
        for agent_id, value in items.items():
            if agent_id in self.groups:
                assert len(value) == len(self.groups[agent_id]), (
                    agent_id,
                    value,
                    self.groups,
                )
                for a, v in zip(self.groups[agent_id], value):
                    out[a] = v
            else:
                out[agent_id] = value
        return out

    def _group_items(self, items, agg_fn=lambda gvals: list(gvals.values())):
        grouped_items = {}
        for agent_id, item in items.items():
            if agent_id in self.agent_id_to_group:
                group_id = self.agent_id_to_group[agent_id]
                if group_id in grouped_items:
                    continue  # already added
                group_out = OrderedDict()
                for a in self.groups[group_id]:
                    if a in items:
                        group_out[a] = items[a]
                    else:
                        raise ValueError(
                            "Missing member of group {}: {}: {}".format(
                                group_id, a, items
                            )
                        )
                grouped_items[group_id] = agg_fn(group_out)
            else:
                grouped_items[agent_id] = item
        return grouped_items

    def observation_space_sample(self, agent_ids: list = None) -> MultiEnvDict:
        print("called obs space sample")
        samples = {}
        if agent_ids is None:
            agent_ids = ["agent_%d" % i for i in range(3)]
        for agent_id in agent_ids:
            samples[agent_id] = self.observation_space.sample()
        return {"agents": samples}

    def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        print("called action space sample")
        samples = {}
        if agent_ids is None:
            agent_ids = ["agent_%d" % i for i in range(self.num_agents)]
        for agent_id in agent_ids:
            samples[agent_id] = self.action_space.sample()
        return samples


if __name__ == '__main__':
    args = parser.parse_args()
    ray.init(num_gpus=0, local_mode=True)

    def create_env(config):
        env = RllibGFootball(args.num_agents)
        return GroupAgentsWrapperFootball(
            env=env,
            groups={"agents": ['agent_0', 'agent_1', 'agent_2']},
            act_space=Tuple([env.action_space, env.action_space, env.action_space]),
            obs_space=Tuple([env.observation_space, env.observation_space, env.observation_space])
        )


    register_env('gfootball', env_creator=create_env)

    tune.run(
        'QMIX',
        stop={'training_iteration': args.num_iters},
        checkpoint_freq=50,

        config={
            'env': 'gfootball',
            'clip_rewards': False,
            'train_batch_size': 250,
            'num_workers': 0,
            'num_envs_per_worker': 1,
            'batch_mode': 'truncate_episodes',
            'observation_filter': 'NoFilter',
            'num_gpus': 0,
            'lr': 5e-6,
            'log_level': 'DEBUG',
            'simple_optimizer': True,
            'optimizer': {'simple_optimizer': True},
        },
        resume=False,
        local_dir=R'C:\Users\Legion\Desktop\gfootball_dumps\ray'
    )
