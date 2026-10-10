from dataclasses import dataclass, field

from src.sim.config import Point, SimulationConfig


@dataclass
class Actor:
    """
    Representation of an actor (agent) in the simulation.
    """
    id: str
    team: str
    x: int
    y: int
    health: int

    alive: bool = True
    kills: int = 0
    goals: int = 0
    reached_goal: bool = False

    @property
    def position(self) -> Point:
        return (self.x, self.y)


@dataclass
class World:
    """
    Representation of the simulation world, including its configuration,
    actors, and current state.
    """
    config: SimulationConfig

    actors: dict[str, Actor] = field(default_factory=dict)
    step_count: int = 0

    finished: bool = False
    winner: str | None = None
    finish_reason: str | None = None

    first_goal_claimed: bool = False

    events: list[dict] = field(default_factory=list)

    def reset(self):
        self.actors = {
            spec.id: Actor(
                id=spec.id,
                team=spec.team,
                x=spec.position[0],
                y=spec.position[1],
                health=self.config.world.max_health,
            )
            for spec in self.config.agents
        }

        self.step_count = 0
        self.finished = False
        self.winner = None
        self.finish_reason = None
        self.first_goal_claimed = False
        self.events = []

    def inside(self, position: Point) -> bool:
        cfg = self.config.world

        return (
            0 <= position[0] < cfg.width
            and 0 <= position[1] < cfg.height
        )

    def free_of_walls(self, position: Point) -> bool:
        return (
            self.inside(position)
            and position not in self.config.world.obstacles
        )

    def alive_actors(self) -> list[Actor]:
        return [
            actor
            for actor in self.actors.values()
            if actor.alive
        ]

    def emit(
        self,
        event_type: str,
        actor: str,
        target: str | None = None,
    ):
        self.events.append({
            "type": event_type,
            "actor": actor,
            "target": target,
            "step": self.step_count,
        })

    def snapshot(self) -> dict:
        cfg = self.config.world

        return {
            "step": self.step_count,
            "width": cfg.width,
            "height": cfg.height,
            "goal": cfg.goal,
            "obstacles": cfg.obstacles,
            "actors": [
                {
                    "id": a.id,
                    "team": a.team,
                    "position": [a.x, a.y],
                    "health": a.health,
                    "alive": a.alive,
                    "kills": a.kills,
                    "goals": a.goals,
                }
                for a in self.actors.values()
            ],
            "events": list(self.events),
            "finished": self.finished,
            "winner": self.winner,
            "finish_reason": self.finish_reason,
        }