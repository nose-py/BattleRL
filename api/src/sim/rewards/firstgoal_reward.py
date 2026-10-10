from src.sim.rewards.base_reward import BaseReward
from src.sim.rewards.utils import empty_rewards


class FirstGoalReward(BaseReward):
    """
    Reward given to the first agent(s) to reach the goal.
    """
    def calculate(self, world, before, participants):
        result = empty_rewards(participants)

        for event in world.events:
            if event["type"] == "first_goal":
                actor = event["actor"]

                if actor in result:
                    result[actor] += 1.0

        return result