from src.sim.policies.utlils import state_key, validate_checkpoint, validate_table
from src.sim.policies.base_policy import BasePolicy


class QLearningPolicy(BasePolicy):
    """
    Q-learning policy that updates its Q table based on experience.
    """
    kind = "q_learning"

    def __init__(self, spec, n_actions, seed):
        super().__init__(spec, n_actions, seed)

        self.q = {}
        self.epsilon = spec.epsilon

    def values(self, key):
        return self.q.setdefault(
            key, [0.0] * self.n_actions
        )

    def best_action(self, values):
        maximum = max(values)

        candidates = [
            i for i, value in enumerate(values)
            if value == maximum
        ]

        return self.rng.choice(candidates)

    def act(self, observation, training=False):
        if training and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)

        key = state_key(observation)
        return self.best_action(self.values(key))

    def learn(
        self,
        observation,
        action,
        reward,
        next_observation,
        done,
    ):
        current_key = state_key(observation)
        current_values = self.values(current_key)

        target = float(reward)

        if not done and next_observation is not None:
            next_key = state_key(next_observation)

            target += (
                self.spec.gamma
                * max(self.values(next_key))
            )

        old_value = current_values[action]

        current_values[action] += (
            self.spec.alpha * (target - old_value)
        )

    def end_episode(self):
        self.epsilon = max(
            self.spec.epsilon_min,
            self.epsilon * self.spec.epsilon_decay,
        )

    def save_state(self):
        return {
            "kind": self.kind,
            "n_actions": self.n_actions,
            "epsilon": self.epsilon,
            "q": self.q,
        }

    def load_state(self, data):
        validate_checkpoint(
            data, self.kind, self.n_actions
        )

        self.q = validate_table(
            data["q"], self.n_actions
        )
        self.epsilon = float(data["epsilon"])