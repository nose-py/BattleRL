from src.sim.policies.base_policy import BasePolicy
from src.sim.policies.utlils import validate_checkpoint

class RandomPolicy(BasePolicy):
    """
    This policy does not learn from experience and always chooses actions randomly.
    """
    def act(self, observation, training=False):
        return self.rng.randrange(self.n_actions)

    def save_state(self):
        return {
            "kind": "random",
            "n_actions": self.n_actions,
        }

    def load_state(self, data):
        validate_checkpoint(data, "random", self.n_actions)