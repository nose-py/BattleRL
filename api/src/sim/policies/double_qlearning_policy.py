from src.sim.policies.utlils import state_key, validate_table
from src.sim.policies.qlearning_policy import QLearningPolicy


class DoubleQLearningPolicy(QLearningPolicy):
    """
    Double Q-learning policy that maintains two Q tables and updates them alternately.
    """
    kind = "double_q"

    def __init__(self, spec, n_actions, seed):
        super().__init__(spec, n_actions, seed)

        self.q2 = {}

    def values2(self, key):
        return self.q2.setdefault(
            key, [0.0] * self.n_actions
        )

    def act(self, observation, training=False):
        if training and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)

        key = state_key(observation)

        q1 = self.values(key)
        q2 = self.values2(key)

        combined = [
            a + b for a, b in zip(q1, q2)
        ]

        return self.best_action(combined)

    def learn(
        self,
        observation,
        action,
        reward,
        next_observation,
        done,
    ):
        key = state_key(observation)

        update_first = self.rng.random() < 0.5

        if update_first:
            table_a = self.values
            table_b = self.values2
        else:
            table_a = self.values2
            table_b = self.values

        current_values = table_a(key)
        target = float(reward)

        if not done and next_observation is not None:
            next_key = state_key(next_observation)

            best = self.best_action(
                table_a(next_key)
            )

            target += (
                self.spec.gamma
                * table_b(next_key)[best]
            )

        current_values[action] += (
            self.spec.alpha
            * (target - current_values[action])
        )

    def save_state(self):
        data = super().save_state()
        data["kind"] = self.kind
        data["q2"] = self.q2
        return data

    def load_state(self, data):
        super().load_state(data)

        self.q2 = validate_table(
            data["q2"], self.n_actions
        )