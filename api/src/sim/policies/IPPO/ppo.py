from .agilerl_ippo import IPPOAgent
from src.sim.policies.base_policy import BasePolicy

from agilerl.algorithms import IPPO
from collections import defaultdict
from typing import Callable

class IPPO(BasePolicy):

    _ippo_agent = None
    
    @classmethod
    def get_ippo_agent(cls, env):
        if cls._ippo_agent is None:
            cls._ippo_agent = IPPOAgent(env)
        return cls._ippo_agent
    
    def __init__(self, spec, n_actions: int, seed: int, env):
        super().__init__(spec, n_actions, seed)
        self.ippo = IPPO.get_ippo_agent(env)

    def act(self, observation, training: bool) -> int:
        agent_id=self.spec.id
        action=self.ippo.act(agent_id=agent_id, observation=observation)
        return action

    def learn(
        self,
        observation,
        action,
        reward,
        next_observation,
        done,
    ):
        agent_id=self.spec.id
        self.ippo.learn(from_=(
            observation,
            action,
            reward,
            next_observation,
            done
        ), from_agent_id=agent_id)

    def load_state(self, data: dict):
        pass
    
    def save_state(self):
        return {}


