from src.sim.rewards.base_reward import BaseReward
from src.sim.rewards.utils import empty_rewards


class InvalidMoveReward(BaseReward):
    """
    Reward removed from agents for making invalid moves.
    """
    def calculate(self, world, before, participants):
        result = empty_rewards(participants)

        for event in world.events:
            if event["type"] == "invalid_move":
                actor = event["actor"]

                if actor in result:
                    result[actor] -= 0.1

        return result