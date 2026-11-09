from pathlib import Path

from .flatten import FlatEnv
from .implementation import IPPOImplementation, Action, Experience
from src.sim.policies.base_policy import BasePolicy

from collections import defaultdict
from typing import Callable


class Buffer:
    def __init__(self, items: list[str]):
        self.memory = {
            item: None
            for item in items
        }

    def write(self, value, for_item: str):
        item = for_item
        if self.memory[item] is not None:
            raise RuntimeError(f"Tried to overwrite value for item {item}")
        self.memory[item] = value

    def read(self, for_item: str):
        item = for_item
        if self.memory[item] is None:
            raise RuntimeError(f"No value for item {item}")
        value = self.memory[item]
        self.memory[item] = None
        return value

class Latch:
    """
    set/reset latch in which a value can be read many times,
    but written to only once before reset
    """
    def __init__(self):
        self.value = False

    def read(self):
        return self.value

    def set(self):
        if self.value:
            raise RuntimeError("Tried to set latch that was already set without resetting")
        self.value = True

    def reset(self):
        self.value = False

class IPPO(BasePolicy):

    kind = "ippo"
    _ippo_agent = None
    _curr_env = None

    @classmethod
    @property
    def agent_not_set(cls) -> bool:
        return cls._ippo_agent is None
    
    @classmethod
    def env_changed(cls, possible_new_env) -> bool:
        if cls._curr_env is not None:
            return possible_new_env != cls._curr_env
        return possible_new_env is not None

    @classmethod
    def get_ippo_agent(cls, env, agent_ids):
        if cls.env_changed(possible_new_env=env) or cls.agent_not_set:
            cls._ippo_agent = IPPOImplementation(env, agent_ids)
            cls._curr_env = env
        return cls._ippo_agent
    
    def __init__(self, spec, n_actions: int, seed: int, env):
        super().__init__(spec, n_actions, seed)
        agent_ids = [
            agent.id 
            for agent in env.config.agents
            if agent.policy == "ippo"
        ]
        self.flat_env = FlatEnv(env)
        self.ippo = IPPO.get_ippo_agent(self.flat_env, agent_ids)
        self.action_buffer = Buffer(items=agent_ids)
        self.latches = {agent_id: Latch() for agent_id in agent_ids}

    def act(self, observation, training: bool) -> int:
        agent_id=self.spec.id
        observation = self.flat_env._flatten_observation(observation, for_agent=agent_id)
        action=self.ippo.act(agent_id=agent_id, observation=observation)
        if training:
            self.action_buffer.write(action, for_item=agent_id)
        return int(action.action[0])

    def learn(
        self,
        observation,
        action,
        reward,
        next_observation,
        done,
    ):
        agent_id=self.spec.id
        observation = self.flat_env._flatten_observation(observation, for_agent=agent_id)
        # PettingZoo may omit observations for agents that just terminated.
        # The terminal mask prevents bootstrapping from this placeholder.
        if next_observation is None:
            next_observation = observation
        else:
            next_observation = self.flat_env._flatten_observation(
                next_observation, for_agent=agent_id
            )
        next_done = done
        done = self.latches[agent_id].read()
        if done:
            self.latches[agent_id].reset()
        action = self.action_buffer.read(for_item=agent_id)
        if next_done:
            self.latches[agent_id].set()


        experience=Experience(
            state=observation,
            action=action.action,
            log_prob=action.log_prob,
            reward=reward,
            done=done,
            value=action.value,
            next_ob=next_observation,
            next_done=next_done
        )
        

        self.ippo.learn(from_=experience, for_agent_id=agent_id)

    def load_state(self, data: dict):
        if data.get("kind") != self.kind:
            raise ValueError("Checkpoint is not an IPPO policy state")

        checkpoint_name = data.get("checkpoint")
        if (
            not isinstance(checkpoint_name, str)
            or Path(checkpoint_name).name != checkpoint_name
            or Path(checkpoint_name).suffix != ".ckpt"
        ):
            raise ValueError("IPPO state must name a local .ckpt artifact")

        self._checkpoint_name = checkpoint_name
    
    def save_state(self):
        return {
            "kind": self.kind,
            "checkpoint": getattr(
                self, "_checkpoint_name", f"{self.spec.id}.ckpt"
            ),
        }

    def save_artifacts(self, folder):
        checkpoint_path = Path(folder) / self.save_state()["checkpoint"]
        self.ippo.agent.save_checkpoint(str(checkpoint_path))

    def load_artifacts(self, folder):
        checkpoint_name = getattr(
            self, "_checkpoint_name", f"{self.spec.id}.ckpt"
        )
        checkpoint_path = Path(folder) / checkpoint_name
        self.ippo.agent.load_checkpoint(str(checkpoint_path))


