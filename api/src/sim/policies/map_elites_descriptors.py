from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class BehaviorTrace:
    n_actions: int

    steps: int = 0
    switches: int = 0
    total_observation_change: float = 0.0

    counts: np.ndarray = field(
        init=False,
        repr=False,
    )

    unique_observations: set[bytes] = field(
        default_factory=set,
        repr=False,
    )

    previous_action: int | None = None

    previous_features: np.ndarray | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self):
        if (
            type(self.n_actions) is not int
            or self.n_actions < 1
        ):
            raise ValueError(
                "n_actions must be a positive integer"
            )

        self.counts = np.zeros(
            self.n_actions,
            dtype=np.int64,
        )

    def reset(self):
        self.steps = 0
        self.switches = 0
        self.total_observation_change = 0.0

        self.counts.fill(0)
        self.unique_observations.clear()

        self.previous_action = None
        self.previous_features = None

    def observe(
        self,
        features: np.ndarray,
        action: int,
    ):
        if not 0 <= action < self.n_actions:
            raise ValueError(
                f"Action index out of range: {action}"
            )

        features = np.asarray(
            features,
            dtype=np.float64,
        ).reshape(-1)

        if not np.all(np.isfinite(features)):
            raise ValueError(
                "Non-finite observation features"
            )

        if self.previous_features is not None:
            if (
                self.previous_features.shape
                != features.shape
            ):
                raise ValueError(
                    "Observation dimensionality changed"
                )

            self.total_observation_change += float(
                np.mean(
                    np.abs(
                        features - self.previous_features
                    )
                ) / 2.0
            )

            if self.previous_action != action:
                self.switches += 1

        self.steps += 1
        self.counts[action] += 1

        self.unique_observations.add(
            features.tobytes()
        )

        self.previous_action = action
        self.previous_features = features.copy()

    @property
    def action_entropy(self) -> float:
        if self.steps == 0 or self.n_actions == 1:
            return 0.0

        frequencies = (
            self.counts[self.counts > 0]
            / self.steps
        )

        return float(
            -np.sum(
                frequencies * np.log(frequencies)
            ) / math.log(self.n_actions)
        )

    @property
    def observation_coverage(self) -> float:
        if self.steps == 0:
            return 0.0

        return (
            len(self.unique_observations)
            / self.steps
        )

    @property
    def action_switch_rate(self) -> float:
        if self.steps <= 1:
            return 0.0

        return self.switches / (self.steps - 1)

    @property
    def observation_change(self) -> float:
        if self.steps <= 1:
            return 0.0

        return (
            self.total_observation_change
            / (self.steps - 1)
        )


class BaseBehaviorDescriptor(ABC):
    """
    Base contract for two-dimensional
    MAP-Elites behavior descriptors.
    """

    name: str
    axis_labels: tuple[str, str]

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def new_trace(self) -> BehaviorTrace:
        return BehaviorTrace(self.n_actions)

    def reset(self, trace: BehaviorTrace):
        trace.reset()

    def observe(
        self,
        trace: BehaviorTrace,
        features: np.ndarray,
        action: int,
    ):
        trace.observe(features, action)

    def describe(
        self,
        trace: BehaviorTrace,
    ) -> tuple[float, float]:
        a, b = self._compute(trace)

        values = (float(a), float(b))

        if not all(
            math.isfinite(v)
            and -1e-10 <= v <= 1.0 + 1e-10
            for v in values
        ):
            raise ValueError(
                f"Descriptor {self.name!r} "
                f"produced out-of-range values: {values}"
            )

        return (float(np.clip(values[0], 0.0, 1.0)), 
                float(np.clip(values[1], 0.0, 1.0)))

    @abstractmethod
    def _compute(
        self,
        trace: BehaviorTrace,
    ) -> tuple[float, float]:
        ...


class EntropyCoverageDescriptor(BaseBehaviorDescriptor):
    """
    Action entropy + unique observation ratio.
    """

    name = "entropy_coverage"

    axis_labels = (
        "Action entropy",
        "Unique observation ratio",
    )

    def _compute(self, trace):
        return (
            trace.action_entropy,
            trace.observation_coverage,
        )


class EntropySwitchingDescriptor(BaseBehaviorDescriptor):
    """
    Action diversity + switching frequency.
    """

    name = "entropy_switching"

    axis_labels = (
        "Action entropy",
        "Action switching rate",
    )

    def _compute(self, trace):
        return (
            trace.action_entropy,
            trace.action_switch_rate,
        )


class CoverageChangeDescriptor(BaseBehaviorDescriptor):
    """
    Observation diversity + observation dynamics.
    """

    name = "coverage_change"

    axis_labels = (
        "Unique observation ratio",
        "Mean observation change",
    )

    def _compute(self, trace):
        return (
            trace.observation_coverage,
            trace.observation_change,
        )


class SwitchingChangeDescriptor(BaseBehaviorDescriptor):
    """
    Action switching + observation dynamics.
    """

    name = "switching_change"

    axis_labels = (
        "Action switching rate",
        "Mean observation change",
    )

    def _compute(self, trace):
        return (
            trace.action_switch_rate,
            trace.observation_change,
        )


DESCRIPTOR_REGISTRY: dict[
    str,
    type[BaseBehaviorDescriptor],
] = {
    cls.name: cls
    for cls in (
        EntropyCoverageDescriptor,
        EntropySwitchingDescriptor,
        CoverageChangeDescriptor,
        SwitchingChangeDescriptor,
    )
}


def build_behavior_descriptor(
    name: str,
    n_actions: int,
) -> BaseBehaviorDescriptor:
    try:
        cls = DESCRIPTOR_REGISTRY[name]

    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Unknown MAP-Elites behavior descriptor: "
            f"{name!r}. Valid choices: "
            f"{', '.join(DESCRIPTOR_REGISTRY)}"
        ) from exc

    return cls(n_actions)
