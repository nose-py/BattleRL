from src.sim.rewards.base_reward import BaseReward
from src.sim.rewards.utils import empty_rewards

class TeamDamageReward(BaseReward):
    """
    Reward given based on the damage dealt by a team.
    """

    def calculate(self, world, before, participants):
        result = empty_rewards(participants)

        for event in world.events:
            if event["type"] == "hit":
                actor = event["actor"]
                
                team = world.actors[actor].team
                for p in participants:
                    if world.actors[p].team == team:
                        result[p] += 1.0

        return result