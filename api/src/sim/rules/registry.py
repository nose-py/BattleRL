from src.sim.rules.movement_rule import MovementRule
from src.sim.rules.shooting_rule import ShootingRule
from src.sim.rules.goal_rule import GoalRule
from src.sim.rules.elimination_rule import EliminationRule
from src.sim.rules.base_rule import BaseRule

RULE_REGISTRY = {
    "movement": MovementRule,
    "shooting": ShootingRule,
    "goal": GoalRule,
    "elimination": EliminationRule,
}


def build_rules(config) -> list[BaseRule]:
    rules = []

    for spec in config.rules:
        rule_class = RULE_REGISTRY[spec.name]

        if spec.name == "goal":
            allowed = {"end_episode"}

            if set(spec.options) - allowed:
                raise ValueError("Goal's options are invalid")

            value = spec.options.get("end_episode", False)

            if not isinstance(value, bool):
                raise ValueError("end_episode must be a boolean")

            rules.append(GoalRule(end_episode=value))
        else:
            if spec.options:
                raise ValueError(
                    f"The rule {spec.name} does not accept options"
                )

            rules.append(rule_class())

    return rules