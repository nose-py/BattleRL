import random
from abc import ABC, abstractmethod

from src.sim.config import AgentSpec

class BasePolicy(ABC):
    def __init__(
        self,
        spec: AgentSpec,
        n_actions: int,
        seed: int,
    ):
        self.spec = spec
        self.n_actions = n_actions
        self.rng = random.Random(seed)

    @abstractmethod
    def act(self, observation, training: bool) -> int:
        ...

    def learn(
        self,
        observation,
        action,
        reward,
        next_observation,
        done,
    ):
        pass

    def end_episode(self):
        pass

    @abstractmethod
    def save_state(self) -> dict:
        ...

    @abstractmethod
    def load_state(self, data: dict):
        ...