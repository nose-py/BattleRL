from collections import defaultdict

from src.sim.rewards.base_reward import BaseReward
from src.sim.rewards.utils import empty_rewards

class TeamEventReward(BaseReward):
    """
    Reward based on specific events occurring in the world,
    but points are shared among team members.
    """
    def __init__(self, values):
        self.values = values

    def calculate(self, world, before, participants):
        result = empty_rewards(participants)
        team_points = defaultdict(float)

        for event in world.events:
            actor = world.actors[event["actor"]]
            team_points[actor.team] += self.values.get(
                event["type"], 0.0
            )

        for agent_id in participants:
            team = world.actors[agent_id].team
            result[agent_id] = team_points[team]

        return result