from .flatten import FlatEnv

import logging

from threading import Lock, Thread
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from agilerl.algorithms import IPPO

logger = logging.getLogger(__name__)

class Barrier:
    def __init__(
        self, consumers: list[str],
        producer: callable, 
        aggregate: callable, get_default: callable
    ):
        self._produce=producer
        self._aggregate=aggregate
        self.get_default=get_default
        self.elem_ids = consumers
        self.reset()

    def reset_consumed_lut(self): 
        self.consumed_lut = {
            elem_id: False
            for elem_id in self.elem_ids
        }

    def set_consumed(self, elem_id: str):
        if not elem_id in self.elem_ids:
            raise ValueError(f"{elem_id} not in batch consumers")
        self.consumed_lut[elem_id]=True

    def has_consumed(self, elem_id: str) -> bool:
        if not elem_id in self.elem_ids:
            raise ValueError(f"{elem_id} not in batch consumers")
        return self.consumed_lut[elem_id]

    def all_consumed(self) -> bool:
        return all([self.has_consumed(elem_id) for elem_id in self.elem_ids])

    def aggregate(self, elem_id, elem_info):
        self._aggregate(self.aggregation, elem_id, elem_info)

    def reset(self):
        self.reset_consumed_lut()
        self.aggregation = self.get_default()

    def produce(self, elem_id: str, elem_info):
        if self.has_consumed(elem_id):
            raise ValueError(f"Consumer {elem_id} tried to consume twice, when other consumers have not consumed yet")
        self.aggregate(elem_id, elem_info)
        if self.all_consumed():
            production = self._produce(self.aggregation)
            self.reset()
            return production

class IPPOImplementation:
    def __init__(self, env):
        self.env=env
        self.flat_env = FlatEnv(env)
        agilerl_env = AsyncPettingZooVecEnv([lambda: self.flat_env])
        observation_spaces = [agilerl_env.single_observation_space(agent) for agent in agilerl_env.agents]
        action_spaces = [agilerl_env.single_action_space(agent) for agent in agilerl_env.agents]
        agent_ids = [agent_id for agent_id in agilerl_env.agents]
        self.agent_ids = agent_ids
        self.agent = IPPO(
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            agent_ids=agent_ids,
        )

        self.learn_barrier = Barrier(
            consumers = self.agent.agent_ids, producer = self.agent.learn,
            get_default = lambda: {}, aggregate = lambda agg, aid, info: agg | {aid: info}
        )

    def learn(self, from_: tuple, from_agent_id: str):
        agent_id = from_agent_id
        learn_args = from_
        self.learn_barrier.produce(elem_info=learn_args, elem_id=agent_id)


    def make_observations(self, agent_id: str, observation):
        other_agents_ids = set(self.agent_ids) - {agent_id}
        observations = {
            agent_id: observation,
            **{
                other_agent_id: self.env.observation_spaces[other_agent_id].sample()
                for other_agent_id in other_agents_ids
            }
        }
        flat_obs = self.flat_env._flatten(observations)
        return flat_obs

    def make_infos(self, agent_id):
        other_agents_ids = set(self.agent_ids) - {agent_id}

        def sample_action(for_agent_id: str):
            return self.flat_env.action_space(for_agent_id).sample()

        infos = {
            agent_id: {'env_defined_actions': None},
            **{
                other_agent_id: {
                    'env_defined_actions': sample_action(for_agent_id= other_agent_id)
                }
                for other_agent_id in other_agents_ids
            }
        }

    def act(self, agent_id: str, observation):
        observations = self.make_observations(agent_id, observation)
        infos = self.make_infos(agent_id)
        all_agents_actions, _, _, _ = self.agent.get_action(observations, infos)
        action = all_agents_actions[agent_id][0]
        return action



