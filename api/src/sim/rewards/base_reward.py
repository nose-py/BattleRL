from abc import ABC, abstractmethod

from src.sim.core import World

class BaseReward(ABC):
    @abstractmethod
    def calculate(
        self,
        world: World,
        before: dict[str, tuple[int, int]],
        participants: list[str],
    ) -> dict[str, float]:
        ...