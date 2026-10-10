from src.sim.config import SimulationConfig
from src.sim.environment import GridWorldEnv
from src.sim.policies.registry import build_policies


class SimulationRunner:
    """
    Class responsible for running simulations of the grid world environment.
    It manages the environment, policies, observations, and rewards.
    """
    def __init__(self, config: SimulationConfig):
        self.config = config

        self.env = GridWorldEnv(config)
        self.policies = build_policies(config)

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