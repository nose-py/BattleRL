from src.sim.rewards.base_reward import BaseReward
from src.sim.rewards.utils import empty_rewards


class TeamFirstGoalReward(BaseReward):
    """
    Reward given to the first team(s) to reach the goal.
    """
    def calculate(self, world, before, participants):
        result = empty_rewards(participants)

        for event in world.events:
            if event["type"] == "first_goal":
                actor = event["actor"]
                team = world.actors[actor].team
                
                for p in participants:
                    if world.actors[p].team == team:
                        result[p] += 1.0

        return result