from src.sim.core import World
from src.sim.rules.base_rule import BaseRule


class GoalRule(BaseRule):
    """
    Rule that handles agents reaching the goal within the simulation.
    """
    def __init__(self, end_episode: bool = False):
        self.end_episode = end_episode

    def apply(self, world: World, actions: dict[str, int]):
        goal = world.config.world.goal

        if goal is None:
            return

        arrivals = []

        for actor in world.alive_actors():
            if actor.position != goal or actor.reached_goal:
                continue

            actor.reached_goal = True
            actor.goals += 1

            world.emit("goal", actor.id)
            arrivals.append(actor)

        if not arrivals:
            return

        if not world.first_goal_claimed:
            world.first_goal_claimed = True

            # All agents that reach the goal in the same
            # step share the first position.
            for actor in arrivals:
                world.emit("first_goal", actor.id)

        if self.end_episode:
            teams = {a.team for a in arrivals}

            world.finished = True
            world.finish_reason = "goal"
            world.winner = (
                next(iter(teams)) if len(teams) == 1 else None
            )