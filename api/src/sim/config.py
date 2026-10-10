from pydantic import BaseModel, Field, model_validator

from src.sim.rules.names import RuleName
from src.sim.rewards.names import RewardName
from src.sim.policies.names import PolicyName

Point = tuple[int, int]

class AgentSpec(BaseModel):
    """
    Specification for an agent in the simulation.
    """
    id: str = Field(
        pattern=r"^[A-Za-z0-9_-]{1,40}$"
    )
    team: str = Field(
        pattern=r"^[A-Za-z0-9_-]{1,40}$"
    )
    position: Point

    policy: PolicyName

    alpha: float = Field(default=0.15, gt=0, le=1)
    gamma: float = Field(default=0.95, ge=0, le=1)

    epsilon: float = Field(default=1.0, ge=0, le=1)
    epsilon_min: float = Field(default=0.05, ge=0, le=1)
    epsilon_decay: float = Field(default=0.995, gt=0, le=1)

    @model_validator(mode="after")
    def validate_epsilon(self):
        if self.epsilon_min > self.epsilon:
            raise ValueError(
                "epsilon_min cannot exceed epsilon"
            )
        return self


class WorldConfig(BaseModel):
    """
    Configuration for the simulation world.
    """
    width: int = Field(default=7, ge=3, le=100)
    height: int = Field(default=7, ge=3, le=100)

    max_steps: int = Field(default=100, ge=1, le=10000)
    max_health: int = Field(default=3, ge=1, le=100)
    vision_radius: int = Field(default=1, ge=1, le=5)

    goal: Point | None = None
    obstacles: list[Point] = Field(default_factory=list)

    seed: int = 42


class RuleSpec(BaseModel):
    """
    Specification for a rule in the simulation.
    """
    name: RuleName

    options: dict[str, bool | int | float | str] = Field(
        default_factory=dict
    )


class RewardSpec(BaseModel):
    """
    Specification for a reward in the simulation.
    """
    kind: RewardName

    weight: float = Field(default=1.0, ge=-10000, le=10000)

    values: dict[str, float] = Field(default_factory=dict)


class SimulationConfig(BaseModel):
    """
    Configuration for the entire simulation, including the world,
    agents, rules, and rewards.
    """
    world: WorldConfig
    agents: list[AgentSpec] = Field(min_length=2, max_length=64)

    rules: list[RuleSpec] = Field(min_length=1)
    rewards: list[RewardSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_simulation(self):
        world = self.world

        def inside(p: Point) -> bool:
            return (
                0 <= p[0] < world.width
                and 0 <= p[1] < world.height
            )

        ids = [a.id for a in self.agents]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate agent IDs")

        teams = {a.team for a in self.agents}
        if len(teams) < 2:
            raise ValueError("At least two teams are required")

        positions = [a.position for a in self.agents]
        if len(set(positions)) != len(positions):
            raise ValueError("Duplicate initial positions")

        if len(set(world.obstacles)) != len(world.obstacles):
            raise ValueError("Duplicate obstacles")

        obstacles = set(world.obstacles)

        if any(not inside(p) for p in obstacles):
            raise ValueError("Obstacle out of bounds")

        if any(not inside(p) or p in obstacles for p in positions):
            raise ValueError("Invalid initial position")

        if world.goal is not None:
            if not inside(world.goal) or world.goal in obstacles:
                raise ValueError("Invalid goal")

        names = [r.name for r in self.rules]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate rules")

        return self