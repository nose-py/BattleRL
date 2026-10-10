from src.sim.rewards.base_reward import BaseReward
from src.sim.rewards.utils import empty_rewards

class EventReward(BaseReward):
    """
    Reward based on specific events occurring in the world.
    """
    def __init__(self, values):
        self.values = values

    def calculate(self, world, before, participants):
        result = empty_rewards(participants)

        for event in world.events:
            actor = event["actor"]

            if actor in result:
                result[actor] += self.values.get(
                    event["type"], 0.0
                )

        return result