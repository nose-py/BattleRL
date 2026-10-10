from collections import defaultdict

from src.sim.core import World
from src.sim.rules.actions import SHOOT_DIRECTIONS
from src.sim.rules.base_rule import BaseRule


class ShootingRule(BaseRule):
    """
    Rule that handles shooting actions of agents within the simulation.
    """
    def apply(self, world: World, actions: dict[str, int]):
        occupants = {
            a.position: a
            for a in world.alive_actors()
        }

        damage = defaultdict(int)
        attackers = defaultdict(list)

        # Shooting actions are calculated before applying damage.
        # This allows two agents to eliminate each other simultaneously.
        for agent_id, action in actions.items():
            shooter = world.actors[agent_id]

            if not shooter.alive or action not in SHOOT_DIRECTIONS:
                continue

            dx, dy = SHOOT_DIRECTIONS[action]

            target_position = (
                shooter.x + dx,
                shooter.y + dy,
            )

            target = occupants.get(target_position)

            if (
                target is None
                or target.team == shooter.team
            ):
                world.emit("miss", agent_id)
                continue

            damage[target.id] += 1
            attackers[target.id].append(agent_id)

            world.emit("hit", agent_id, target.id)

        for target_id, amount in damage.items():
            target = world.actors[target_id]
            target.health = max(0, target.health - amount)

            if target.health == 0:
                target.alive = False

                # Only one kill is awarded per victim, even if
                # multiple attackers participated.
                killer_id = sorted(attackers[target_id])[0]
                world.actors[killer_id].kills += 1

                world.emit("kill", killer_id, target_id)