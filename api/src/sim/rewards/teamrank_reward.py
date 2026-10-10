from collections import defaultdict

from src.sim.rewards.base_reward import BaseReward
from src.sim.rewards.utils import empty_rewards
from src.sim.rewards.utils import distance

class TeamRankReward(BaseReward):
    """
    Terminal reward for the team whose sum of distances
    to the goal is the smallest.
    """
    def calculate(self, world, before, participants):
        result = empty_rewards(participants)

        if not world.finished:
            return result

        goal = world.config.world.goal

        if goal is None:
            return result

        team_distances = defaultdict(float)

        for actor in world.actors.values():
            team_distances[actor.team] += distance(
                actor.position, goal
            )

        best = min(team_distances.values())

        for agent_id in participants:
            team = world.actors[agent_id].team

            if team_distances[team] == best:
                result[agent_id] = 1.0

        return result