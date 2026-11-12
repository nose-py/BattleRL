import numpy as np
import torch

from dataclasses import dataclass
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from agilerl.algorithms import IPPO

@dataclass(frozen=True)
class Experience:
    state: any #Receive as arg
    action: any #Get from get_action
    log_prob: any #Get from get_action
    reward: any #Receive as arg
    done: bool #Buffer
    value: any #Get from get_action
    next_ob: any #Receive as arg
    next_done: bool #Receive as arg

    def as_experience_tuple(self) -> tuple:
        # Each call represents exactly one timestep. Preserve the leading
        # timestep dimension for observations and scalar fields.
        return (
            np.expand_dims(np.asarray(self.state, dtype=np.float32), axis=0),
            np.asarray(self.action).reshape(-1),
            np.asarray(self.log_prob).reshape(1),
            np.asarray(self.reward).reshape(1),
            np.asarray([self.done], dtype=np.float32),
            np.asarray(self.value).reshape(1),
            np.expand_dims(np.asarray(self.next_ob, dtype=np.float32), axis=0),
            np.asarray([self.next_done], dtype=np.float32),
        )

@dataclass(frozen=True)
class Action:
    action: int
    log_prob: float
    value: float

class IPPOImplementation:
    rollout_length = 2

    def __init__(self, env, agent_ids: list[str]):
        self.env=env
        #self.flat_env = FlatEnv(env)
        raw_env = getattr(env, "env", env)
        configured_agents = getattr(
            getattr(raw_env, "config", None), "agents", ()
        )
        team_by_agent_id = {
            agent.id: agent.team for agent in configured_agents
        }
        team_indexes = {}
        self.network_agent_ids = {}
        for agent_id in agent_ids:
            team = team_by_agent_id.get(agent_id, agent_id)
            team_index = team_indexes.get(team, 0)
            team_indexes[team] = team_index + 1
            self.network_agent_ids[agent_id] = f"{team}_{team_index}"

        agilerl_env = AsyncPettingZooVecEnv([lambda: self.env])
        observation_spaces = [
            agilerl_env.single_observation_space(agent) for agent in agent_ids
        ]
        action_spaces = [
            agilerl_env.single_action_space(agent) for agent in agent_ids
        ]
        self.agent_ids = agent_ids
        self.agent = IPPO(
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            agent_ids=[self.network_agent_ids[agent_id] for agent_id in agent_ids],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.rollout = {agent_id: [] for agent_id in self.agent_ids}
        self.last_transitions = {agent_id: None for agent_id in self.agent_ids}

    def _collect_and_learn(self, transition, agent_id):
        self.rollout[agent_id].append(transition)
        self.last_transitions[agent_id] = transition

        if len(self.rollout[agent_id]) < self.rollout_length:
            return

        # IPPO.learn expects all configured agent groups. A peer that has fewer
        # than two fresh transitions gets a neutral terminal sample so a dead
        # or masked agent cannot hold up the other agents' updates.
        batches = {}
        remaining_rollouts = {}
        for current_agent_id in self.agent_ids:
            rows = list(self.rollout[current_agent_id])
            if len(rows) >= self.rollout_length:
                batches[current_agent_id] = rows[: self.rollout_length]
                remaining_rollouts[current_agent_id] = rows[self.rollout_length :]
                continue

            previous = rows[-1] if rows else self.last_transitions[current_agent_id]
            if previous is None:
                return

            def neutral_terminal_sample(transition):
                return (
                    transition[0],
                    transition[1],
                    transition[2],
                    np.zeros_like(transition[3]),
                    np.ones_like(transition[4]),
                    np.zeros_like(transition[5]),
                    transition[6],
                    np.ones_like(transition[7]),
                )

            batches[current_agent_id] = rows + [
                neutral_terminal_sample(previous)
                for _ in range(self.rollout_length - len(rows))
            ]
            remaining_rollouts[current_agent_id] = []

        # AgileRL expects lists of per-timestep arrays for the six rollout
        # fields, plus only the final next state and done for GAE bootstrap.
        rollout_fields = tuple(
            {
                self.network_agent_ids[current_agent_id]: [
                    row[field_index] for row in rows
                ]
                for current_agent_id, rows in batches.items()
            }
            for field_index in range(6)
        )
        next_states = {
            self.network_agent_ids[current_agent_id]: rows[-1][6]
            for current_agent_id, rows in batches.items()
        }
        next_dones = {
            self.network_agent_ids[current_agent_id]: rows[-1][7]
            for current_agent_id, rows in batches.items()
        }
        self.agent.learn((*rollout_fields, next_states, next_dones))
        self.rollout = remaining_rollouts

    def learn(
        self, 
        from_: Experience,
        for_agent_id: str,
    ):
        agent_id = for_agent_id
        experience=from_
        experience_tuple = experience.as_experience_tuple()
        self._collect_and_learn(experience_tuple, agent_id)


    def make_observations(self, agent_id: str, observation):
        other_agents_ids = set(self.agent_ids) - {agent_id}
        network_agent_id = self.network_agent_ids[agent_id]
        observations = {
            network_agent_id: observation,
            **{
                self.network_agent_ids[other_agent_id]: (
                    self.env.observation_space(other_agent_id).sample()
                )
                for other_agent_id in other_agents_ids
            }
        }
        #flat_obs = self.flat_env._flatten(observations)
        return observations

    def make_infos(self, agent_id):
        other_agents_ids = set(self.agent_ids) - {agent_id}
        network_agent_id = self.network_agent_ids[agent_id]

        def sample_action(for_agent_id: str):
            return self.env.action_space(for_agent_id).sample()

        infos = {
            network_agent_id: {'env_defined_actions': None},
            **{
                self.network_agent_ids[other_agent_id]: {
                    'env_defined_actions': sample_action(for_agent_id= other_agent_id)
                }
                for other_agent_id in other_agents_ids
            }
        }

        return infos

    def act(self, agent_id: str, observation):
        observations = self.make_observations(agent_id, observation)
        infos = self.make_infos(agent_id)
        actions_tuple: tuple = self.agent.get_action(observations, infos)
        action_tuple = tuple(
            elem[self.network_agent_ids[agent_id]] for elem in actions_tuple
        )
        (action, log_prob, entropy, value) = action_tuple
        action = Action(
            action=action,
            log_prob=log_prob,
            value=value
        )
        return action


