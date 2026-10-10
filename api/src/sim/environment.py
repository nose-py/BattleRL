import random

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from src.sim.config import SimulationConfig
from src.sim.core import World
from src.sim.rules.actions import ACTION_NAMES
from src.sim.rules.registry import build_rules
from src.sim.rewards.registry import build_reward


class GridWorldEnv(ParallelEnv):
    """
    Custom GridWorld environment compatible with PettingZoo's ParallelEnv interface.
    """
    metadata = {
        "name": "modular_grid_v0",
        "render_modes": [],
        "is_parallelizable": True,
    }

    def __init__(self, config: SimulationConfig):
        super().__init__()

        self.config = config
        self.world = World(config)

        self.rules = build_rules(config)
        self.reward_engine = build_reward(config)

        self.possible_agents = [
            spec.id for spec in config.agents
        ]
        self.agents = []

        cfg = config.world
        side = 2 * cfg.vision_radius + 1

        self.observation_spaces = {
            agent_id: spaces.Dict({
                "view": spaces.Box(
                    low=0,
                    high=5,
                    shape=(side, side),
                    dtype=np.int8,
                ),
                "self": spaces.MultiDiscrete([
                    cfg.width,
                    cfg.height,
                    cfg.max_health + 1,
                ]),
            })
            for agent_id in self.possible_agents
        }

        self.action_spaces = {
            agent_id: spaces.Discrete(len(ACTION_NAMES))
            for agent_id in self.possible_agents
        }

        self.rng = random.Random(cfg.seed)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed)

        self.world.reset()
        self.agents = self.possible_agents.copy()

        observations = {
            agent_id: self.observe(agent_id)
            for agent_id in self.agents
        }

        infos = {
            agent_id: {}
            for agent_id in self.agents
        }

        return observations, infos

    def observe(self, agent_id):
        actor = self.world.actors[agent_id]
        cfg = self.config.world
        radius = cfg.vision_radius

        side = 2 * radius + 1

        # 0 = free
        # 1 = obstacle
        # 2 = ally
        # 3 = enemy
        # 4 = goal
        # 5 = out of map

        view = np.zeros((side, side), dtype=np.int8)

        occupants = {
            a.position: a
            for a in self.world.alive_actors()
        }

        walls = set(cfg.obstacles)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                position = (
                    actor.x + dx,
                    actor.y + dy,
                )

                row = dy + radius
                col = dx + radius

                if not self.world.inside(position):
                    value = 5
                elif position in walls:
                    value = 1
                elif (
                    position in occupants
                    and position != actor.position
                ):
                    other = occupants[position]

                    value = (
                        2 if other.team == actor.team else 3
                    )
                elif position == cfg.goal:
                    value = 4
                else:
                    value = 0

                view[row, col] = value

        return {
            "view": view,
            "self": np.array(
                [actor.x, actor.y, actor.health],
                dtype=np.int64,
            ),
        }

    def step(self, actions):
        if not self.agents:
            raise RuntimeError("The simulation has ended")

        live_before = self.agents.copy()

        if set(actions) != set(live_before):
            raise ValueError(
                "An action is required for each active agent"
            )

        for agent_id, action in actions.items():
            if not self.action_space(agent_id).contains(action):
                raise ValueError(
                    f"Invalid action for {agent_id}"
                )

        before = {
            agent_id: self.world.actors[agent_id].position
            for agent_id in live_before
        }

        self.world.events = []
        self.world.step_count += 1

        for rule in self.rules:
            if self.world.finished:
                break

            rule.apply(self.world, actions)

        alive = self.world.alive_actors()

        if not self.world.finished and not alive:
            self.world.finished = True
            self.world.finish_reason = "no_agents"

        if (
            not self.world.finished
            and self.world.step_count >= self.config.world.max_steps
        ):
            self.world.finished = True
            self.world.finish_reason = "time_limit"

        rewards = self.reward_engine.calculate(
            self.world,
            before,
            live_before,
        )

        timed_out = self.world.finish_reason == "time_limit"

        terminations = {
            agent_id: (
                not self.world.actors[agent_id].alive
                or (self.world.finished and not timed_out)
            )
            for agent_id in live_before
        }

        truncations = {
            agent_id: (
                timed_out
                and self.world.actors[agent_id].alive
            )
            for agent_id in live_before
        }

        self.agents = [
            agent_id
            for agent_id in live_before
            if (
                not terminations[agent_id]
                and not truncations[agent_id]
            )
        ]

        observations = {
            agent_id: self.observe(agent_id)
            for agent_id in self.agents
        }

        infos = {
            agent_id: {
                "events": list(self.world.events),
            }
            for agent_id in live_before
        }

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )

    def state_snapshot(self):
        return self.world.snapshot()

    def render(self):
        return self.state_snapshot()

    def close(self):
        pass