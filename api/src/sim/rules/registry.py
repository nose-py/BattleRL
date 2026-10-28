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

        try:
            rules.append(rule_class(
                **spec.options
            ))
        except Exception as e:
            raise RuntimeError(
                f"Error building rule "
                f"{spec.name}: {e}"
            ) from e

    return rules