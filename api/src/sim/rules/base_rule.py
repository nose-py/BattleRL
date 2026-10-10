from abc import ABC, abstractmethod

from src.sim.core import World

class BaseRule(ABC):
    @abstractmethod
    def apply(
        self,
        world: World,
        actions: dict[str, int],
    ) -> None:
        ...