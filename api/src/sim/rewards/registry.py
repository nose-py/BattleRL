from src.sim.rewards.base_reward import BaseReward
from src.sim.rewards.event_reward import EventReward
from src.sim.rewards.teamevent_reward import TeamEventReward
from src.sim.rewards.progress_reward import ProgressReward
from src.sim.rewards.teamprogress_reward import TeamProgressReward
from src.sim.rewards.firstgoal_reward import FirstGoalReward
from src.sim.rewards.teamrank_reward import TeamRankReward
from src.sim.rewards.constant_reward import ConstantReward

from src.sim.rewards.utils import empty_rewards

REWARD_REGISTRY = {
    "event": EventReward,
    "team_event": TeamEventReward,
    "progress": ProgressReward,
    "team_progress": TeamProgressReward,
    "first_goal": FirstGoalReward,
    "team_rank": TeamRankReward,
    "constant": ConstantReward,
}

class CompositeReward(BaseReward):
    """
    Composite reward that combines multiple reward components,
    each with an associated weight.
    """
    def __init__(self, components):
        self.components = components

    def calculate(self, world, before, participants):
        result = empty_rewards(participants)

        for component, weight in self.components:
            partial = component.calculate(
                world, before, participants
            )

            for agent_id in participants:
                result[agent_id] += (
                    weight * partial[agent_id]
                )

        return result

def build_reward(config) -> CompositeReward:
    components = []

    for spec in config.rewards:
        reward_class = REWARD_REGISTRY[spec.kind]

        if spec.kind in ("event", "team_event"):
            reward = reward_class(spec.values)
        else:
            if spec.values:
                raise ValueError(
                    f"{spec.kind} does not accept values"
                )

            reward = reward_class()

        components.append((reward, spec.weight))

    return CompositeReward(components)