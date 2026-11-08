import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class FlatEnv(ParallelEnv):
    def __init__(self, env):
        self.env = env
        self.possible_agents = env.possible_agents
        self.metadata = env.metadata
        self.render_mode = getattr(env, "render_mode", None)

        self._observation_spaces = {
            agent: spaces.flatten_space(env.observation_space(agent))
            for agent in self.possible_agents
        }

    @property
    def agents(self):
        return self.env.agents

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self.env.action_space(agent)

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        return self._flatten(observations), infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return (
            self._flatten(observations),
            rewards,
            terminations,
            truncations,
            infos,
        )

    def _flatten_observation(self, observation, for_agent: str):
        agent=for_agent
        return spaces.flatten(
                self.env.observation_space(agent), observation
        ).astype(np.float32)

    def _flatten(self, observations):
        d = {
            agent: self._flatten_observation(observation, for_agent=agent)
            for agent, observation in observations.items()
        }
        return d

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

