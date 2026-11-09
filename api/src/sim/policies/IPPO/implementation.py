import logging
import numpy as np

from dataclasses import dataclass
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
        self.aggregation = self._aggregate(self.aggregation, elem_id, elem_info)

    def reset(self):
        self.reset_consumed_lut()
        self.aggregation = self.get_default()

    def produce(self, elem_id: str, elem_info):
        if self.has_consumed(elem_id):
            raise ValueError(f"Consumer {elem_id} tried to consume twice, when other consumers have not consumed yet")
        self.aggregate(elem_id, elem_info)
        self.set_consumed(elem_id)
        if self.all_consumed():
            production = self._produce(self.aggregation)
            self.reset()
            return production



@dataclass(frozen=True)
class Experience:
    state: any #Receive as arg
    action: any #Get from get_action
    log_prob: any #Get from get_action
    reward: any #Receive as arg
    done: bool #Buffer
    value: any #Get from get_action
    next_ob: any #Receive as arg
    next_done: bool #Receive as arg

    def as_experience_tuple(self) -> tuple:
        # Each call represents exactly one timestep. Preserve the leading
        # timestep dimension for observations and scalar fields.
        return (
            np.expand_dims(np.asarray(self.state, dtype=np.float32), axis=0),
            np.asarray(self.action).reshape(-1),
            np.asarray(self.log_prob).reshape(1),
            np.asarray(self.reward).reshape(1),
            np.asarray([self.done], dtype=np.float32),
            np.asarray(self.value).reshape(1),
            np.expand_dims(np.asarray(self.next_ob, dtype=np.float32), axis=0),
            np.asarray([self.next_done], dtype=np.float32),
        )

@dataclass(frozen=True)
class Action:
    action: int
    log_prob: float
    value: float

class IPPOImplementation:
    rollout_length = 2

    def __init__(self, env, agent_ids: list[str]):
        self.env=env
        #self.flat_env = FlatEnv(env)
        agilerl_env = AsyncPettingZooVecEnv([lambda: self.env])
        observation_spaces = [
            agilerl_env.single_observation_space(agent) for agent in agent_ids
        ]
        action_spaces = [
            agilerl_env.single_action_space(agent) for agent in agent_ids
        ]
        self.agent_ids = agent_ids
        self.agent = IPPO(
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            agent_ids=agent_ids,
        )

        def aggr(agg, aid, info):
            num_infos = len(info)
            if agg is None:
                agg = tuple({} for _ in range(num_infos))
            for idx, info_field in enumerate(info):
                agg[idx][aid] = info_field
            return agg

        self.learn_barrier = Barrier(
            consumers = self.agent.agent_ids, producer = self._collect_and_learn,
            get_default = lambda: None, aggregate = aggr
        )

        self.rollout = {agent_id: [] for agent_id in self.agent_ids}

    def _collect_and_learn(self, experiences):
        # The barrier gives one transition for every agent at each timestep.
        for agent_id in self.agent_ids:
            transition = tuple(field[agent_id] for field in experiences)
            self.rollout[agent_id].append(transition)

        if len(next(iter(self.rollout.values()))) < self.rollout_length:
            return

        # AgileRL expects lists of per-timestep arrays for the six rollout
        # fields, plus only the final next state and done for GAE bootstrap.
        rollout_fields = tuple(
            {
                agent_id: [row[field_index] for row in rows]
                for agent_id, rows in self.rollout.items()
            }
            for field_index in range(6)
        )
        next_states = {
            agent_id: rows[-1][6]
            for agent_id, rows in self.rollout.items()
        }
        next_dones = {
            agent_id: rows[-1][7]
            for agent_id, rows in self.rollout.items()
        }
        self.agent.learn((*rollout_fields, next_states, next_dones))
        self.rollout = {agent_id: [] for agent_id in self.agent_ids}

    def learn(
        self, 
        from_: Experience,
        for_agent_id: str,
    ):
        agent_id = for_agent_id
        experience=from_
        experience_tuple = experience.as_experience_tuple()
        self.learn_barrier.produce(elem_info=experience_tuple, elem_id=agent_id)


    def make_observations(self, agent_id: str, observation):
        other_agents_ids = set(self.agent_ids) - {agent_id}
        observations = {
            agent_id: observation,
            **{
                other_agent_id: self.env.observation_space(other_agent_id).sample()
                for other_agent_id in other_agents_ids
            }
        }
        #flat_obs = self.flat_env._flatten(observations)
        return observations

    def make_infos(self, agent_id):
        other_agents_ids = set(self.agent_ids) - {agent_id}

        def sample_action(for_agent_id: str):
            return self.env.action_space(for_agent_id).sample()

        infos = {
            agent_id: {'env_defined_actions': None},
            **{
                other_agent_id: {
                    'env_defined_actions': sample_action(for_agent_id= other_agent_id)
                }
                for other_agent_id in other_agents_ids
            }
        }

        return infos

    def act(self, agent_id: str, observation):
        observations = self.make_observations(agent_id, observation)
        infos = self.make_infos(agent_id)
        actions_tuple: tuple = self.agent.get_action(observations, infos)
        action_tuple = tuple(
            elem[agent_id] for elem in actions_tuple
        )
        (action, log_prob, entropy, value) = action_tuple
        action = Action(
            action=action,
            log_prob=log_prob,
            value=value
        )
        return action


