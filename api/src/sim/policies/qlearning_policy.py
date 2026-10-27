from src.sim.policies.utils import state_key_str, validate_checkpoint, validate_table
from src.sim.policies.base_policy import BasePolicy

class QLearningPolicy(BasePolicy):
    """
    Q-learning policy that updates its Q table based on experience.
    """
    kind = "q_learning"

    def __init__(self, spec, n_actions, seed, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        super().__init__(spec, n_actions, seed)

        self.q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

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

        key = state_key_str(observation)
        return self.best_action(self.values(key))

    def learn(
        self,
        observation,
        action,
        reward,
        next_observation,
        done,
    ):
        current_key = state_key_str(observation)
        current_values = self.values(current_key)

        target = float(reward)

        if not done and next_observation is not None:
            next_key = state_key_str(next_observation)

            target += (
                self.gamma
                * max(self.values(next_key))
            )

        old_value = current_values[action]

        current_values[action] += (
            self.alpha * (target - old_value)
        )

    def end_episode(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
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