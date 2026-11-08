from src.sim.policies.qlearning_policy import (
    QLearningPolicy,
)
from src.sim.policies.random_policy import (
    RandomPolicy,
)
from src.sim.policies.double_qlearning_policy import (
    DoubleQLearningPolicy,
)
from src.sim.policies.IPPO.ppo import (
    IPPO
)

from src.sim.policies.map_elites_policy import (
    MAPElitesPolicy,
)

POLICY_REGISTRY = {
    "random": RandomPolicy,
    "q_learning": QLearningPolicy,
    "double_q": DoubleQLearningPolicy,
    "ippo": IPPO,
    "map_elites": MAPElitesPolicy,
}


def build_policies(config, n_actions: int, env):
    result = {}

    for index, spec in enumerate(config.agents):
        policy_class = POLICY_REGISTRY[
            spec.policy
        ]
        try:
            if spec.policy == "ippo":
                result[spec.id] = policy_class(
                    spec=spec,
                    n_actions=n_actions,
                    seed=config.world.seed + index + 1,
                    env=env
                )
            else:
                result[spec.id] = policy_class(
                    spec=spec,
                    n_actions=n_actions,
                    seed=config.world.seed + index + 1,
                    **spec.params
                )
        except Exception as e:
            raise RuntimeError(
                f"Error building policy for agent "
                f"{spec.id}: {e}"
            ) from e

    return result
