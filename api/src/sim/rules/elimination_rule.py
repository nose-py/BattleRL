from src.sim.core import World
from src.sim.rules.base_rule import BaseRule

class EliminationRule(BaseRule):
    """
    Rule that handles the elimination condition within the simulation.
    """
    def apply(self, world: World, actions: dict[str, int]):
        teams = {
            actor.team
            for actor in world.alive_actors()
        }

        if len(teams) <= 1:
            world.finished = True
            world.finish_reason = "elimination"
            world.winner = (
                next(iter(teams)) if teams else None
            )