from abc import ABC, abstractmethod
from typing import ClassVar

from src.sim.core import World

class BaseRule(ABC):
    ACTIONS: ClassVar[list[int]] = []

    @classmethod
    def declared_actions(cls) -> list[int]:
        return cls.ACTIONS

    @abstractmethod
    def apply(
        self,
        world: World,
        actions: dict[str, int],
    ) -> None:
        ...