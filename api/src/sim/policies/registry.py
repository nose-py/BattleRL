from src.sim.policies.qlearning_policy import QLearningPolicy
from src.sim.policies.random_policy import RandomPolicy
from src.sim.policies.double_qlearning_policy import DoubleQLearningPolicy

POLICY_REGISTRY = {
    "random": RandomPolicy,
    "q_learning": QLearningPolicy,
    "double_q": DoubleQLearningPolicy,
}


def build_policies(config):
    result = {}

    for index, spec in enumerate(config.agents):
        policy_class = POLICY_REGISTRY[spec.policy]

        result[spec.id] = policy_class(
            spec=spec,
            n_actions=9,
            seed=config.world.seed + index + 1,
        )

    return result