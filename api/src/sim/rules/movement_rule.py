from collections import Counter

from src.sim.rules.actions import DIRECTIONS
from src.sim.rules.base_rule import BaseRule
from src.sim.core import World


class MovementRule(BaseRule):
    """
    Rule that handles the movement of agents within the simulation.
    """
    def apply(self, world: World, actions: dict[str, int]):
        occupied = {
            a.position
            for a in world.alive_actors()
        }

        proposals = {}

        # All movement proposals are based on the positions
        # at the start of this phase.
        for agent_id, action in actions.items():
            actor = world.actors[agent_id]

            if not actor.alive or action not in DIRECTIONS:
                continue

            dx, dy = DIRECTIONS[action]
            destination = (actor.x + dx, actor.y + dy)

            if (
                not world.free_of_walls(destination)
                or destination in occupied
            ):
                world.emit("invalid_move", agent_id)
                continue

            proposals[agent_id] = destination

        counts = Counter(proposals.values())

        # Movements to the same cell cancel each other out.
        # Entering a cell occupied at the start of the step is not allowed,
        # even if its occupant is about to leave.
        for agent_id, destination in proposals.items():
            if counts[destination] > 1:
                world.emit("invalid_move", agent_id)
                continue

            actor = world.actors[agent_id]
            actor.x, actor.y = destination
            world.emit("move", agent_id)