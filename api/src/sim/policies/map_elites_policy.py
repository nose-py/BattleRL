import math
from copy import deepcopy
from collections.abc import Mapping

import numpy as np

from src.sim.policies.base_policy import BasePolicy

from src.sim.policies.map_elites_descriptors import (
    BehaviorTrace,
    build_behavior_descriptor,
)

class MAPElitesPolicy(BasePolicy):
    """MAP-Elites policy using quality-diversity neuroevolution."""

    kind = "map_elites"

    def __init__(
        self,
        spec,
        n_actions: int,
        seed: int,
        behavior_descriptor: str,
        iterations: int,
        evaluation_episodes: int,
        hidden_size: int,
        bins: int,
        mutation_sigma: float,
        mutation_rate: float,
        random_injection: float,
    ):
        super().__init__(spec, n_actions, seed)

        for name, value, low, high in (
            ("n_actions", n_actions, 1, 1000),
            ("iterations", iterations, 1, 50000),
            ("evaluation_episodes", evaluation_episodes, 1, 100),
            ("hidden_size", hidden_size, 1, 128),
            ("bins", bins, 2, 20),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError(
                    f"{name} must be an integer in [{low}, {high}]"
                )

        for name, value, minimum, maximum, allow_zero in (
            ("mutation_sigma", mutation_sigma, 0.0, 5.0, False),
            ("mutation_rate", mutation_rate, 0.0, 1.0, False),
            ("random_injection", random_injection, 0.0, 1.0, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not minimum <= value <= maximum
                or (not allow_zero and value == 0)
            ):
                raise ValueError(
                    f"Invalid {name}: {value!r}"
                )

        self.iterations = iterations
        self.evaluation_episodes = evaluation_episodes
        self.hidden_size = hidden_size
        self.bins = bins

        self.mutation_sigma = float(mutation_sigma)
        self.mutation_rate = float(mutation_rate)
        self.random_injection = float(random_injection)

        self.behavior_descriptor_name = behavior_descriptor

        self._descriptor = build_behavior_descriptor(
            behavior_descriptor,
            n_actions,
        )

        self._np_rng = np.random.default_rng(seed)

        self._input_dim: int | None = None
        self._genome: np.ndarray | None = None
        self._optimized = False

        self._best_fitness: float | None = None
        self._archive: dict[str, dict] = {}
        self._evaluations = 0

        self._trace_sink: BehaviorTrace | None = None

        self._episode_reward = 0.0
        self._training_trace = self._new_trace()
        self._new_genome_next_episode = False

    def _new_trace(self) -> BehaviorTrace:
        return self._descriptor.new_trace()

    @staticmethod
    def _flatten(observation) -> np.ndarray:
        parts = []

        def visit(value):
            if isinstance(value, Mapping):
                for key in sorted(value):
                    visit(value[key])

            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)

            else:
                array = np.asarray(
                    value, dtype=np.float64
                ).reshape(-1)

                if array.size:
                    parts.append(array)

        visit(observation)

        if not parts:
            raise ValueError(
                "Observation contains no numeric features"
            )

        result = np.concatenate(parts)

        if not np.all(np.isfinite(result)):
            raise ValueError(
                "Observation contains non-finite values"
            )

        return result / (1.0 + np.abs(result))

    def _genome_size(self, input_dim: int) -> int:
        return (
            input_dim * self.hidden_size
            + self.hidden_size
            + self.hidden_size * self.n_actions
            + self.n_actions
        )

    def _random_genome(self) -> np.ndarray:
        if self._input_dim is None:
            raise RuntimeError(
                "Observation size has not been initialized"
            )

        dim = self._input_dim
        h = self.hidden_size
        a = self.n_actions

        w1 = self._np_rng.normal(
            0.0, 1.0 / math.sqrt(dim), (dim, h)
        )
        b1 = np.zeros(h, dtype=np.float64)

        w2 = self._np_rng.normal(
            0.0, 1.0 / math.sqrt(h), (h, a)
        )
        b2 = np.zeros(a, dtype=np.float64)

        return np.concatenate(
            (w1.ravel(), b1, w2.ravel(), b2)
        )

    def _ensure_input(self, feature_count: int):
        if self._input_dim is None:
            self._input_dim = feature_count
            self._genome = self._random_genome()

        elif self._input_dim != feature_count:
            raise ValueError(
                f"Observation size changed from "
                f"{self._input_dim} to {feature_count}"
            )

    def _forward(self, features: np.ndarray) -> np.ndarray:
        if self._input_dim is None or self._genome is None:
            raise RuntimeError(
                "Policy genome is not initialized"
            )

        dim = self._input_dim
        h = self.hidden_size
        a = self.n_actions

        genome = self._genome

        offset = dim * h
        w1 = genome[:offset].reshape(dim, h)

        b1 = genome[offset:offset + h]
        offset += h

        w2 = genome[
            offset:offset + h * a
        ].reshape(h, a)

        offset += h * a
        b2 = genome[offset:offset + a]

        hidden = np.tanh(features @ w1 + b1)

        return hidden @ w2 + b2

    def act(
        self,
        observation,
        training: bool = False,
    ) -> int:

        features = self._flatten(
            observation
        )

        self._ensure_input(
            features.size
        )

        if (
            training
            and self._new_genome_next_episode
        ):
            self._genome = self._random_genome()
            self._new_genome_next_episode = False


        action = int(
            np.argmax(
                self._forward(features)
            )
        )

        sink = (
            self._training_trace
            if training
            else self._trace_sink
        )

        if sink is not None:
            self._descriptor.observe(
                sink,
                features,
                action,
            )

        return action

    def learn(
        self,
        observation,
        action,
        reward,
        next_observation,
        done,
    ):
        value = float(reward)

        if not math.isfinite(value):
            raise ValueError(
                "Non-finite training reward"
            )

        self._episode_reward += value

    def end_episode(self):
        if (
            self._genome is not None
            and self._training_trace.steps > 0
        ):
            descriptor = self._behavior_descriptor(self._training_trace)

            cell = self._cell(descriptor)
            previous = self._archive.get(cell)

            if (
                previous is None
                or self._episode_reward > previous["fitness"]
            ):
                self._archive[cell] = {
                    "fitness": self._episode_reward,
                    "descriptor": list(descriptor),
                    "genome": self._genome.tolist(),
                }

            self._evaluations += 1

            self._new_genome_next_episode = True

            self._optimized = False
            self._best_fitness = None

        self._episode_reward = 0.0
        self._training_trace = self._new_trace()

    def requires_optimization(self) -> bool:
        return True

    def prepare_evaluation_episode(self, seed: int):
        super().prepare_evaluation_episode(seed)

        if self._trace_sink is not None:
            self._descriptor.reset(
                self._trace_sink
            )

    def copy_to(self, other):
        if (
            not isinstance(other, MAPElitesPolicy)
            or other.n_actions != self.n_actions
            or other.hidden_size != self.hidden_size
            or other.bins != self.bins
            or other.behavior_descriptor_name != self.behavior_descriptor_name
        ):
            raise ValueError(
                "Incompatible MAP-Elites destination policy"
            )

        other._input_dim = self._input_dim

        other._genome = (
            None
            if self._genome is None
            else self._genome.copy()
        )

        other._optimized = self._optimized
        other._best_fitness = self._best_fitness
        other._evaluations = self._evaluations

        other._archive = deepcopy(self._archive)

        other._trace_sink = self._trace_sink

    def _behavior_descriptor(
        self,
        sink: BehaviorTrace | None = None,
    ) -> tuple[float, float]:

        if sink is None:
            sink = self._trace_sink

        if sink is None:
            return (0.0, 0.0)

        return self._descriptor.describe(sink)

    def _cell(
        self,
        descriptor: tuple[float, float],
    ) -> str:
        a, b = descriptor

        i = min(
            int(a * self.bins),
            self.bins - 1,
        )
        j = min(
            int(b * self.bins),
            self.bins - 1,
        )

        return f"{i}:{j}"

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        child = genome.copy()

        mask = (
            self._np_rng.random(child.size)
            < self.mutation_rate
        )

        if not np.any(mask):
            mask[
                self._np_rng.integers(0, child.size)
            ] = True

        child[mask] += self._np_rng.normal(
            0.0,
            self.mutation_sigma,
            size=int(mask.sum()),
        )

        return np.clip(child, -5.0, 5.0)

    def optimize(self, evaluator, report=None):
        if self._input_dim is None:
            raise RuntimeError(
                "MAP-Elites requires at least one observation "
                "before optimization; run a training episode first"
            )

        for elite in self._archive.values():
            elite["fitness"] = float("-inf")

        self._np_rng = np.random.default_rng(self.seed)

        best_genome = None
        best_score = None

        self._best_fitness = None

        self._new_genome_next_episode = False

        evaluated = 0

        for iteration in range(self.iterations):
            if (
                not self._archive
                or self._np_rng.random()
                < self.random_injection
            ):
                genome = self._random_genome()

            else:
                keys = tuple(self._archive)

                key = keys[
                    int(
                        self._np_rng.integers(
                            len(keys)
                        )
                    )
                ]

                parent = np.asarray(
                    self._archive[key]["genome"],
                    dtype=np.float64,
                )

                genome = self._mutate(parent)

            self._genome = genome.copy()

            self._trace_sink = self._new_trace()

            episode_descriptors = []

            def episode_score(result):
                episode_descriptors.append(
                    self._behavior_descriptor()
                )

                return float(
                    result["agent_rewards"][self.spec.id]
                )

            try:
                fitness = float(
                    evaluator.evaluate_episodes(
                        repetitions=self.evaluation_episodes,
                        score=episode_score,
                    )
                )

            finally:
                self._trace_sink = None

            if (
                not math.isfinite(fitness)
                or not episode_descriptors
            ):
                raise ValueError(
                    "Evaluation returned invalid fitness "
                    "or descriptors"
                )

            descriptor = tuple(
                float(np.mean([
                    d[i] for d in episode_descriptors
                ]))
                for i in range(2)
            )

            cell = self._cell(descriptor) # pyright: ignore[reportArgumentType]

            previous = self._archive.get(cell)

            if (
                previous is None
                or fitness > previous["fitness"]
            ):
                self._archive[cell] = {
                    "fitness": fitness,
                    "descriptor": list(descriptor),
                    "genome": genome.tolist(),
                }

            if best_score is None or fitness > best_score:
                best_score = fitness
                best_genome = genome.copy()

            evaluated += 1
            self._evaluations += 1

            if report is not None and (
                iteration == 0
                or (iteration + 1) % max(
                    1, self.iterations // 20
                ) == 0
                or iteration + 1 == self.iterations
            ):
                report({
                    "evaluations": iteration + 1,
                    "total_evaluations": self.iterations,
                    "evaluated_candidates": evaluated,
                    "archive_cells": len(self._archive),
                    "archive_capacity": self.bins * self.bins,
                    "coverage": (
                        len(self._archive)
                        / (self.bins * self.bins)
                    ),
                    "best_reward": best_score,
                })

        self._genome = best_genome
        self._best_fitness = best_score
        self._optimized = True

        return {
            "best_reward": best_score,
            "evaluated_candidates": evaluated,
            "evaluation_episodes": self.evaluation_episodes,
            "archive_cells": len(self._archive),
            "archive_capacity": self.bins * self.bins,
            "coverage": (
                len(self._archive) / (self.bins * self.bins)
            ),
            "best_cell": max(
                self._archive,
                key=lambda k: self._archive[k]["fitness"],
            ),
            "behavior_descriptor": self.behavior_descriptor_name,
            "behavior_axes": list(self._descriptor.axis_labels),
        }

    def save_state(self) -> dict:
        return {
            "kind": self.kind,
            "n_actions": self.n_actions,
            "hidden_size": self.hidden_size,
            "bins": self.bins,
            "input_dim": self._input_dim,
            "genome": (
                None if self._genome is None
                else self._genome.tolist()
            ),
            "optimized": self._optimized,
            "best_fitness": self._best_fitness,
            "evaluations": self._evaluations,
            "archive": {
                key: {
                    "fitness": (
                        None
                        if item["fitness"] == float("-inf")
                        else item["fitness"]
                    ),
                    "descriptor": list(item["descriptor"]),
                    "genome": list(item["genome"]),
                }
                for key, item in self._archive.items()
            },
            "new_genome_next_episode": self._new_genome_next_episode,
            "behavior_descriptor": self.behavior_descriptor_name,
        }

    def load_state(self, data: dict):
        if data.get("kind") != self.kind:
            raise ValueError(
                "Checkpoint policy type mismatch"
            )

        if data.get("n_actions") != self.n_actions:
            raise ValueError(
                "Checkpoint action count mismatch"
            )

        if data.get("behavior_descriptor", "entropy_coverage") != self.behavior_descriptor_name:
            raise ValueError(
                "Checkpoint behavior descriptor mismatch: "
                f"{data.get("behavior_descriptor", "entropy_coverage")!r} vs "
                f"{self.behavior_descriptor_name!r}"
            )

        if (
            data.get("hidden_size") != self.hidden_size
            or data.get("bins") != self.bins
        ):
            raise ValueError(
                "Checkpoint MAP-Elites architecture mismatch"
            )

        dim = data.get("input_dim")

        if dim is not None and (
            type(dim) is not int or dim < 1
        ):
            raise ValueError(
                "Invalid observation dimension in checkpoint"
            )

        expected_size = (
            None if dim is None
            else self._genome_size(dim)
        )

        def validated_genome(raw):
            if raw is None:
                return None

            vector = np.asarray(
                raw, dtype=np.float64
            ).reshape(-1)

            if (
                expected_size is None
                or vector.size != expected_size
            ):
                raise ValueError(
                    "Invalid checkpoint genome dimension"
                )

            if not np.all(np.isfinite(vector)):
                raise ValueError(
                    "Non-finite checkpoint genome"
                )

            return vector

        genome = validated_genome(data.get("genome"))
        archive = {}

        for key, item in data.get("archive", {}).items():
            try:
                i, j = map(int, key.split(":"))

            except Exception as exc:
                raise ValueError(
                    "Invalid archive cell"
                ) from exc

            if not (
                0 <= i < self.bins
                and 0 <= j < self.bins
            ):
                raise ValueError(
                    "Archive cell out of bounds"
                )

            fitness = (
                float("-inf")
                if item["fitness"] is None
                else float(item["fitness"])
            )

            descriptor = np.asarray(
                item["descriptor"],
                dtype=np.float64,
            )

            elite = validated_genome(
                item["genome"]
            )

            if (
                not (
                    math.isfinite(fitness)
                    or fitness == float("-inf")
                )
                or descriptor.shape != (2,)
                or not np.all(np.isfinite(descriptor))
                or np.any(descriptor < 0)
                or np.any(descriptor > 1)
                or elite is None
                or self._cell((
                    float(descriptor[0]),
                    float(descriptor[1]),
                )) != key
            ):
                raise ValueError(
                    "Invalid MAP-Elites archive record"
                )

            archive[key] = {
                "fitness": fitness,
                "descriptor": descriptor.tolist(),
                "genome": elite.tolist(),
            }

        best = data.get("best_fitness")

        if best is not None:
            best = float(best)

            if not math.isfinite(best):
                raise ValueError(
                    "Invalid best_fitness"
                )

        optimized = data.get("optimized", False)

        if (
            type(optimized) is not bool
            or (
                optimized
                and (genome is None or not archive)
            )
        ):
            raise ValueError(
                "Invalid optimized checkpoint state"
            )

        evaluations = data.get("evaluations", 0)

        if (
            type(evaluations) is not int
            or evaluations < 0
        ):
            raise ValueError(
                "Invalid evaluation count"
            )

        new_genome_next_episode = data.get("new_genome_next_episode", False)

        if type(new_genome_next_episode) is not bool:
            raise ValueError(
                "Invalid training episode state"
            )

        self._input_dim = dim
        self._genome = genome
        self._archive = archive

        self._best_fitness = best
        self._optimized = optimized
        self._evaluations = evaluations
        self._new_genome_next_episode = new_genome_next_episode
        self._episode_reward = 0.0
        self._training_trace = self._new_trace()
