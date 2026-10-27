from copy import deepcopy
from hashlib import blake2b
import pickle
import random
from collections.abc import Callable

import numpy as np


def _observation_key(observation):
    payload = pickle.dumps(
        observation,
        protocol=5,
    )

    return blake2b(
        payload,
        digest_size=20,
    ).digest()


class PolicyEvaluator:
    """
    Evaluates agent decisions within a predefined
    roster of reference policies.
    """

    def __init__(
        self,
        runner,
        agent_id: str,
        sources: dict,
    ):
        self.runner = runner
        self.agent_id = agent_id
        self.sources = sources
        self.seed = runner.config.world.seed

        self.agent_salt = int.from_bytes(
            blake2b(
                agent_id.encode("utf-8"),
                digest_size=8,
            ).digest(),
            "big",
        ) % 1_000_000

    def close(self):
        self.runner.env.close()

    def _restore_policies(self):
        for agent_id, source in self.sources.items():
            source.copy_to(
                self.runner.policies[agent_id]
            )

    def _prepare_randomness(self, seed):
        np.random.seed(
            seed % (2 ** 32)
        )

        for index, policy in enumerate(
            self.runner.policies.values()
        ):
            policy.prepare_evaluation_episode(
                seed + index + 1
            )

    def evaluate_episodes(
        self,
        repetitions: int,
        score: Callable[[dict], float],
    ) -> float:
        if repetitions < 1:
            raise ValueError(
                "repetitions must be >= 1"
            )

        previous_np_state = (
            np.random.get_state()
        )

        results = []

        try:
            self._restore_policies()

            for episode in range(repetitions):
                episode_id = 100_000 + episode
                seed = self.seed + episode_id

                self._prepare_randomness(seed)

                result = self.runner.run_episode(
                    episode_id,
                    training=False,
                )

                results.append(
                    float(score(result))
                )

        finally:
            np.random.set_state(
                previous_np_state
            )

        return float(np.mean(results))

    def _capture(self, observation, key):
        return {
            "_key": key,

            "world": deepcopy(
                self.runner.env.world
            ),

            "active_agents": list(
                self.runner.env.agents
            ),

            "observations": deepcopy(
                self.runner.observations
            ),

            "observation": deepcopy(
                observation
            ),
        }

    def sample_contexts(
        self,
        limit: int,
        episodes: int,
    ) -> list[dict]:
        if limit < 1 or episodes < 1:
            raise ValueError(
                "limit and episodes must be >= 1"
            )

        rng = random.Random(
            self.seed
            + 300_000
            + self.agent_salt
        )

        samples = []

        selected_indices = {}
        occurrences = {}

        seen = set()

        unique_count = 0

        previous_np_state = (
            np.random.get_state()
        )

        try:
            self._restore_policies()

            for episode in range(episodes):
                seed = (
                    self.seed
                    + 300_000
                    + episode
                )

                self._prepare_randomness(seed)

                self.runner.reset(
                    seed=seed
                )

                while self.runner.env.agents:
                    if self.agent_id in self.runner.env.agents:
                        observation = (
                            self.runner.observations[
                                self.agent_id
                            ]
                        )

                        key = _observation_key(
                            observation
                        )

                        if key not in seen:
                            seen.add(key)
                            unique_count += 1

                            index = None

                            if len(samples) < limit:
                                index = len(samples)

                            else:
                                candidate = rng.randrange(
                                    unique_count
                                )

                                if candidate < limit:
                                    index = candidate

                            if index is not None:

                                if index < len(samples):
                                    old_key = (
                                        samples[index]["_key"]
                                    )

                                    selected_indices.pop(
                                        old_key
                                    )

                                    occurrences.pop(
                                        old_key
                                    )

                                selected_indices[key] = index
                                occurrences[key] = 1

                                context = self._capture(
                                    observation,
                                    key,
                                )

                                if index == len(samples):
                                    samples.append(context)

                                else:
                                    samples[index] = context

                        elif key in selected_indices:
                            occurrences[key] += 1

                            if rng.randrange(
                                occurrences[key]
                            ) == 0:
                                samples[
                                    selected_indices[key]
                                ] = self._capture(
                                    observation,
                                    key,
                                )

                    self.runner.step(
                        training=False,
                        return_state=False,
                    )

        finally:
            np.random.set_state(
                previous_np_state
            )

        for context in samples:
            del context["_key"]

        return samples

    def evaluate_decision(
        self,
        context: dict,
        decide: Callable,
        repetitions: int,
        score: Callable[[dict], float],
    ) -> float:
        if repetitions < 1:
            raise ValueError(
                "repetitions must be >= 1"
            )

        scores = []

        previous_np_state = (
            np.random.get_state()
        )

        try:
            for index in range(repetitions):
                self._restore_policies()

                seed = (
                    self.seed
                    + 400_000
                    + index
                )

                self._prepare_randomness(seed)

                self.runner.env.world = deepcopy(
                    context["world"]
                )

                self.runner.env.agents = list(
                    context["active_agents"]
                )

                self.runner.observations = deepcopy(
                    context["observations"]
                )

                self.runner.total_rewards = {
                    aid: 0.0
                    for aid in self.runner.env.possible_agents
                }

                if self.agent_id not in self.runner.env.agents:
                    raise ValueError(
                        "Agent not active in captured context"
                    )

                action = decide(
                    self.runner.policies[
                        self.agent_id
                    ],
                    self.runner.observations[
                        self.agent_id
                    ],
                    seed,
                )

                self.runner.step(
                    manual_actions={
                        self.agent_id: int(action),
                    },
                    training=False,
                    return_state=False,
                )

                while self.runner.env.agents:
                    self.runner.step(
                        training=False,
                        return_state=False,
                    )

                team_rewards = {}

                for spec in self.runner.config.agents:
                    team_rewards[spec.team] = (
                        team_rewards.get(
                            spec.team,
                            0.0,
                        )
                        + self.runner.total_rewards[
                            spec.id
                        ]
                    )

                result = {
                    "agent_rewards": dict(
                        self.runner.total_rewards
                    ),
                    "team_rewards": team_rewards,
                }

                scores.append(
                    float(score(result))
                )

        finally:
            np.random.set_state(
                previous_np_state
            )

        return float(np.mean(scores))