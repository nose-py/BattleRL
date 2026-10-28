from src.sim.config import SimulationConfig
from src.sim.environment import GridWorldEnv

from src.sim.evaluation import PolicyEvaluator

from src.sim.policies.registry import build_policies


class SimulationRunner:
    """
    Class responsible for running simulations of the grid world environment.
    It manages the environment, policies, observations, and rewards.
    """
    def __init__(self, config: SimulationConfig):
        self.config = config

        self.env = GridWorldEnv(config)
        self.policies = build_policies(config, n_actions=self.env.action_catalog.n_actions)

        self.observations = {}
        self.total_rewards = {}

        self.reset()

    def reset(self, seed=None):
        self.observations, _ = self.env.reset(seed=seed)

        self.total_rewards = {
            agent_id: 0.0
            for agent_id in self.env.possible_agents
        }

        return self.snapshot()

    def snapshot(self):
        state = self.env.state_snapshot()
        state["total_rewards"] = dict(self.total_rewards)
        return state

    def step(
        self,
        manual_actions=None,
        training=False,
        return_state=True
    ):
        if not self.env.agents:
            raise RuntimeError("The episode has already ended.")

        manual_actions = manual_actions or {}
        for agent_id in manual_actions:
            manual_actions[agent_id] = self.env.action_catalog.index_for_code(manual_actions[agent_id])

        unknown = set(manual_actions) - set(self.env.agents)

        if unknown:
            raise ValueError(
                f"Inactive or unknown agents: {sorted(unknown)}"
            )

        previous = self.observations

        actions = {
            agent_id: self.policies[agent_id].act(
                previous[agent_id],
                training=training,
            )
            for agent_id in self.env.agents
            if agent_id not in manual_actions
        }

        actions.update(manual_actions)

        (
            next_observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(actions)

        for agent_id, reward in rewards.items():
            self.total_rewards[agent_id] += reward

            if training:
                done = (
                    terminations[agent_id]
                    or truncations[agent_id]
                )

                self.policies[agent_id].learn(
                    observation=previous[agent_id],
                    action=actions[agent_id],
                    reward=reward,
                    next_observation=next_observations.get(
                        agent_id
                    ),
                    done=done,
                )

        self.observations = next_observations

        if not return_state:
            return None

        state = self.snapshot()

        state.update({
            "actions": actions,
            "rewards": rewards,
            "terminations": terminations,
            "truncations": truncations,
        })

        return state

    def run_episode(self, episode_number, training=True):
        self.reset(
            seed=self.config.world.seed + episode_number
        )

        while self.env.agents:
            self.step(training=training, return_state=False)

        if training:
            for policy in self.policies.values():
                policy.end_episode()

        team_rewards = {}

        for spec in self.config.agents:
            team_rewards.setdefault(spec.team, 0.0)

            team_rewards[spec.team] += (
                self.total_rewards[spec.id]
            )

        return {
            "episode": episode_number,
            "steps": self.env.world.step_count,
            "winner": self.env.world.winner,
            "finish_reason": self.env.world.finish_reason,
            "agent_rewards": dict(self.total_rewards),
            "team_rewards": team_rewards,
            "kills": {
                actor.id: actor.kills
                for actor in self.env.world.actors.values()
            },
            "goals": {
                actor.id: actor.goals
                for actor in self.env.world.actors.values()
            },
        }

    def optimize_policies(self, on_event=None):
        results = {}
        optimized = set()

        if not any(
            policy.requires_optimization()
            for policy in self.policies.values()
        ):
            return results

        baseline = SimulationRunner(
            self.config
        )

        for agent_id, source in self.policies.items():
            source.copy_to(
                baseline.policies[agent_id]
            )

        teams = {
            spec.id: spec.team
            for spec in self.config.agents
        }

        try:
            for agent_id, policy in self.policies.items():

                if not policy.requires_optimization():
                    continue

                sources = {
                    other_id: (
                        self.policies[other_id]
                        if (
                            other_id != agent_id
                            and teams[other_id] == teams[agent_id]
                            and other_id in optimized
                        )
                        else baseline.policies[other_id]
                    )
                    for other_id in self.policies
                }

                sources[agent_id] = policy

                evaluator = PolicyEvaluator(
                    runner=SimulationRunner(
                        self.config
                    ),
                    agent_id=agent_id,
                    sources=sources,
                )

                try:
                    if on_event is not None:
                        on_event(
                            "optimization_started",
                            {
                                "agent_id": agent_id,
                            },
                        )

                    def report(progress):
                        if on_event is not None:
                            on_event(
                                "optimization_progress",
                                {
                                    "agent_id": agent_id,
                                    **progress,
                                },
                            )

                    result = policy.optimize(
                        evaluator,
                        report=report,
                    )

                    results[agent_id] = result

                    optimized.add(agent_id)

                    if on_event is not None:
                        on_event(
                            "optimization_completed",
                            result,
                        )

                finally:
                    evaluator.close()

        finally:
            baseline.env.close()

        return results
    