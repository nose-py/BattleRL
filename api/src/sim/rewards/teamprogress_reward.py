from collections import defaultdict

from src.sim.rewards.base_reward import BaseReward
from src.sim.rewards.utils import empty_rewards, distance

class TeamProgressReward(BaseReward):
    """
    Reward based on the collective progress of team members towards the goal, if a goal is defined.
    """
    def calculate(self, world, before, participants):
        result = empty_rewards(participants)
        goal = world.config.world.goal

        if goal is None:
            return result

        team_progress = defaultdict(float)

        for agent_id in participants:
            actor = world.actors[agent_id]

            team_progress[actor.team] += (
                distance(before[agent_id], goal)
                - distance(actor.position, goal)
            )

        for agent_id in participants:
            team = world.actors[agent_id].team
            result[agent_id] = team_progress[team]

        return result