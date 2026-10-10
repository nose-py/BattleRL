from src.sim.rewards.base_reward import BaseReward
from src.sim.rewards.utils import empty_rewards, distance


class ProgressReward(BaseReward):
    """
    Reward based on the progress of agents towards the goal, if a goal is defined.
    """
    def calculate(self, world, before, participants):
        result = empty_rewards(participants)
        goal = world.config.world.goal

        if goal is None:
            return result

        for agent_id in participants:
            previous = before[agent_id]
            current = world.actors[agent_id].position

            result[agent_id] = (
                distance(previous, goal)
                - distance(current, goal)
            )

        return result