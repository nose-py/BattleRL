import math
import torch

if torch.cuda.is_available():
    import cupy as np
else:
    import numpy as np

from src.sim.policies.base_policy import BasePolicy

VIEW_CATEGORIES = 6
TD_ERROR_CLIP = 10.0


class MLP:
    """Two-hidden-layer ReLU network with manual backprop and Adam."""

    BETA1 = 0.9
    BETA2 = 0.999
    ADAM_EPSILON = 1e-8

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        n_actions: int,
        rng: np.random.Generator,
    ):
        sizes = (input_dim, hidden_size, hidden_size, n_actions)

        self.params: list[np.ndarray] = []

        for index, (fan_in, fan_out) in enumerate(
            zip(sizes[:-1], sizes[1:])
        ):
            std = (
                0.01
                if index == len(sizes) - 2
                else math.sqrt(2.0 / fan_in)
            )

            self.params.append(
                rng.normal(0.0, std, (fan_in, fan_out))
            )
            self.params.append(
                np.zeros(fan_out, dtype=np.float64)
            )

        self._reset_adam()

    def _reset_adam(self):
        self._adam_t = 0
        self._adam_m: list[np.ndarray] | None = None
        self._adam_v: list[np.ndarray] | None = None

    def forward(self, x: np.ndarray):
        last = len(self.params) - 2
        cache = []

        for index in range(0, len(self.params), 2):
            weights, bias = self.params[index:index + 2]

            activation = x @ weights + bias
            active = None

            if index != last:
                active = activation > 0
                activation = activation * active

            cache.append((x, active))
            x = activation

        return x, cache

    def backward(self, cache, d_out: np.ndarray) -> list[np.ndarray]:
        """Return d(sum of d_out * out)/d(params); rows are summed, not averaged."""
        grads: list[np.ndarray] = [None] * len(self.params)  # type: ignore[list-item]
        gradient = d_out

        for layer in range(len(cache) - 1, -1, -1):
            x, active = cache[layer]

            if active is not None:
                gradient = gradient * active

            grads[2 * layer] = x.T @ gradient
            grads[2 * layer + 1] = gradient.sum(axis=0)

            gradient = gradient @ self.params[2 * layer].T

        return grads

    def adam_step(self, grads: list[np.ndarray], lr: float):
        if self._adam_m is None or self._adam_v is None:
            self._adam_m = [np.zeros_like(p) for p in self.params]
            self._adam_v = [np.zeros_like(p) for p in self.params]

        self._adam_t += 1

        m_scale = 1.0 - self.BETA1 ** self._adam_t
        v_scale = 1.0 - self.BETA2 ** self._adam_t

        for param, grad, m, v in zip(
            self.params, grads, self._adam_m, self._adam_v
        ):
            m += (1.0 - self.BETA1) * (grad - m)
            v += (1.0 - self.BETA2) * (grad * grad - v)

            param -= lr * (m / m_scale) / (
                np.sqrt(v / v_scale) + self.ADAM_EPSILON
            )

    def copy_params_from(self, other: "MLP"):
        for mine, theirs in zip(self.params, other.params):
            np.copyto(mine, theirs)

        self._reset_adam()

    def to_dict(self) -> dict:
        return {
            "params": [param.tolist() for param in self.params],
        }

    def parse_dict(self, data) -> list[np.ndarray]:
        """Validate a serialized network without touching this one."""
        raw = data.get("params") if isinstance(data, dict) else None

        if not isinstance(raw, list) or len(raw) != len(self.params):
            raise ValueError("Invalid network checkpoint")

        params = []

        for value, expected in zip(raw, self.params):
            try:
                array = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Invalid network checkpoint"
                ) from exc

            if array.shape != expected.shape:
                raise ValueError(
                    "Network checkpoint shape mismatch"
                )

            if not np.all(np.isfinite(array)):
                raise ValueError(
                    "Non-finite network checkpoint"
                )

            params.append(array)

        return params

    def set_params(self, params: list[np.ndarray]):
        self.params = params
        self._reset_adam()

    def load_dict(self, data):
        self.set_params(self.parse_dict(data))


class VDNTeamLearner:
    """Joint Double DQN trainer shared by the VDN members of one team."""

    def __init__(
        self,
        team: str,
        members: list[str],
        hparams: dict,
        world,
        n_actions: int,
        seed: int,
    ):
        self.team = team
        self.members = list(members)
        self.hparams = dict(hparams)
        self.world = world
        self.n_actions = n_actions

        self.indices = {
            agent_id: index
            for index, agent_id in enumerate(self.members)
        }

        self.share_parameters = hparams["share_parameters"]

        n_agents = len(self.members)
        side = 2 * world.vision_radius + 1

        self.input_dim = (
            side * side * VIEW_CATEGORIES
            + 3
            + (n_agents if self.share_parameters else 0)
        )

        self.rng = np.random.default_rng(seed)

        n_networks = 1 if self.share_parameters else n_agents

        self.networks = [
            MLP(
                self.input_dim,
                hparams["hidden_size"],
                n_actions,
                self.rng,
            )
            for _ in range(n_networks)
        ]

        self.target_networks = [
            MLP(
                self.input_dim,
                hparams["hidden_size"],
                n_actions,
                self.rng,
            )
            for _ in range(n_networks)
        ]

        for target, network in zip(
            self.target_networks, self.networks
        ):
            target.copy_params_from(network)

        self.epsilon = hparams["epsilon"]
        self.gradient_steps = 0

        capacity = hparams["buffer_size"]
        shape = (capacity, n_agents, self.input_dim)

        self.buf_obs = np.zeros(shape, dtype=np.float32)
        self.buf_next = np.zeros(shape, dtype=np.float32)
        self.buf_act = np.zeros(
            (capacity, n_agents), dtype=np.int64
        )
        self.buf_mask = np.zeros(
            (capacity, n_agents), dtype=np.float32
        )
        self.buf_next_mask = np.zeros(
            (capacity, n_agents), dtype=np.float32
        )
        self.buf_reward = np.zeros(capacity, dtype=np.float32)

        self.buf_pos = 0
        self.buf_size = 0

        self.pending: dict[str, tuple] = {}
        self.phase = "act"
        self._ended: set[str] = set()

        self._view_encoding = np.eye(
            VIEW_CATEGORIES, dtype=np.float32
        )

    def features(self, agent_id: str, observation) -> np.ndarray:
        view = np.asarray(observation["view"], dtype=np.int64)

        if view.size and (
            view.min() < 0 or view.max() >= VIEW_CATEGORIES
        ):
            raise ValueError(
                "Observation view contains an unknown category"
            )

        x, y, health = observation["self"]
        world = self.world

        parts = [
            self._view_encoding[view].ravel(),
            np.array(
                [
                    x / (world.width - 1),
                    y / (world.height - 1),
                    health / world.max_health,
                ],
                dtype=np.float32,
            ),
        ]

        if self.share_parameters:
            identity = np.zeros(
                len(self.members), dtype=np.float32
            )
            identity[self.indices[agent_id]] = 1.0
            parts.append(identity)

        return np.concatenate(parts)

    def _network_index(self, agent_id: str) -> int:
        return (
            0
            if self.share_parameters
            else self.indices[agent_id]
        )

    def q_values(self, agent_id: str, features: np.ndarray) -> np.ndarray:
        network = self.networks[self._network_index(agent_id)]
        output, _ = network.forward(features[np.newaxis])

        return output[0]

    def on_act(self, training: bool):
        if training and self.phase == "learn":
            self._flush_step()

        self.phase = "act"

    def record(
        self,
        agent_id: str,
        features: np.ndarray,
        action: int,
        reward: float,
        next_features: np.ndarray | None,
        done: bool,
    ):
        # Steps made only of manual actions never reach on_act().
        if agent_id in self.pending:
            self._flush_step()

        self.pending[agent_id] = (
            features,
            int(action),
            float(reward),
            next_features,
            bool(done),
        )

        self.phase = "learn"

    def end_episode(self, agent_id: str):
        self._flush_step()
        self.phase = "act"

        self._ended.add(agent_id)

        if len(self._ended) == len(self.members):
            self._ended.clear()

            self.epsilon = max(
                self.hparams["epsilon_min"],
                self.epsilon * self.hparams["epsilon_decay"],
            )

    def _flush_step(self):
        if not self.pending:
            return

        slot = self.buf_pos

        # The slot may hold an older transition with other active agents.
        self.buf_obs[slot] = 0.0
        self.buf_next[slot] = 0.0
        self.buf_act[slot] = 0
        self.buf_mask[slot] = 0.0
        self.buf_next_mask[slot] = 0.0

        rewards = []

        for agent_id, (
            features,
            action,
            reward,
            next_features,
            done,
        ) in self.pending.items():
            index = self.indices[agent_id]

            self.buf_obs[slot, index] = features
            self.buf_act[slot, index] = action
            self.buf_mask[slot, index] = 1.0

            if not done and next_features is not None:
                self.buf_next[slot, index] = next_features
                self.buf_next_mask[slot, index] = 1.0

            rewards.append(reward)

        self.buf_reward[slot] = (
            sum(rewards) / len(rewards)
            if self.hparams["team_reward"] == "mean"
            else sum(rewards)
        )

        self.pending = {}

        self.buf_pos = (slot + 1) % self.hparams["buffer_size"]
        self.buf_size = min(
            self.buf_size + 1,
            self.hparams["buffer_size"],
        )

        if self.buf_size >= self.hparams["batch_size"]:
            self._train_step()

    def _forward_all(self, networks: list[MLP], x: np.ndarray):
        """Map [B, n, F] to Q-values [B, n, A] plus backprop caches."""
        batch, n_agents, _ = x.shape

        if self.share_parameters:
            output, cache = networks[0].forward(
                x.reshape(batch * n_agents, self.input_dim)
            )

            return (
                output.reshape(batch, n_agents, self.n_actions),
                [cache],
            )

        results = [
            network.forward(x[:, index])
            for index, network in enumerate(networks)
        ]

        return (
            np.stack([output for output, _ in results], axis=1),
            [cache for _, cache in results],
        )

    def _apply_gradients(self, caches: list, d_out: np.ndarray):
        lr = self.hparams["learning_rate"]

        if self.share_parameters:
            network = self.networks[0]

            network.adam_step(
                network.backward(
                    caches[0],
                    d_out.reshape(-1, self.n_actions),
                ),
                lr,
            )
            return

        for index, network in enumerate(self.networks):
            network.adam_step(
                network.backward(caches[index], d_out[:, index]),
                lr,
            )

    def _train_step(self):
        batch = self.hparams["batch_size"]
        n_agents = len(self.members)

        sample = self.rng.integers(0, self.buf_size, batch)

        actions = self.buf_act[sample]
        mask = self.buf_mask[sample]
        next_mask = self.buf_next_mask[sample]

        rows = np.arange(batch)[:, np.newaxis]
        columns = np.arange(n_agents)[np.newaxis, :]

        q_online, caches = self._forward_all(
            self.networks, self.buf_obs[sample]
        )

        next_observations = self.buf_next[sample]

        next_online, _ = self._forward_all(
            self.networks, next_observations
        )
        next_target, _ = self._forward_all(
            self.target_networks, next_observations
        )

        next_actions = next_online.argmax(axis=2)
        next_q = next_target[rows, columns, next_actions]

        q_team = (q_online[rows, columns, actions] * mask).sum(axis=1)

        target = (
            self.buf_reward[sample]
            + self.hparams["gamma"] * (next_q * next_mask).sum(axis=1)
        )

        td_error = np.clip(
            q_team - target,
            -TD_ERROR_CLIP,
            TD_ERROR_CLIP,
        )

        d_out = np.zeros_like(q_online)
        d_out[rows, columns, actions] = (
            2.0 * td_error[:, np.newaxis] * mask / batch
        )

        self._apply_gradients(caches, d_out)

        self.gradient_steps += 1

        if self.gradient_steps % self.hparams["target_update"] == 0:
            for target_network, network in zip(
                self.target_networks, self.networks
            ):
                target_network.copy_params_from(network)

    def state_dict(self) -> dict:
        return {
            "networks": [n.to_dict() for n in self.networks],
            "target_networks": [
                n.to_dict() for n in self.target_networks
            ],
            "epsilon": self.epsilon,
            "gradient_steps": self.gradient_steps,
        }

    def load_state_dict(self, data: dict):
        networks = data.get("networks")
        target_networks = data.get("target_networks")

        if (
            not isinstance(networks, list)
            or not isinstance(target_networks, list)
            or len(networks) != len(self.networks)
            or len(target_networks) != len(self.target_networks)
        ):
            raise ValueError("Checkpoint network count mismatch")

        epsilon = data.get("epsilon")
        gradient_steps = data.get("gradient_steps")

        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not 0.0 <= epsilon <= 1.0
        ):
            raise ValueError("Invalid epsilon in checkpoint")

        if type(gradient_steps) is not int or gradient_steps < 0:
            raise ValueError("Invalid gradient_steps in checkpoint")

        # Validate everything first so a bad checkpoint changes nothing.
        parsed = [
            (network, network.parse_dict(state))
            for network, state in zip(
                self.networks + self.target_networks,
                networks + target_networks,
            )
        ]

        for network, params in parsed:
            network.set_params(params)

        self.epsilon = float(epsilon)
        self.gradient_steps = gradient_steps

        self.pending = {}
        self.phase = "act"
        self._ended = set()

    def copy_weights_to(self, other: "VDNTeamLearner"):
        if other is self:
            return

        for mine, theirs in zip(
            self.networks + self.target_networks,
            other.networks + other.target_networks,
        ):
            theirs.copy_params_from(mine)

        other.epsilon = self.epsilon


def _validated_integer(name, value, low, high) -> int:
    if isinstance(value, float) and value.is_integer():
        value = int(value)

    if type(value) is not int or not low <= value <= high:
        raise ValueError(
            f"{name} must be an integer in [{low}, {high}]"
        )

    return value


def _validated_float(name, value, low, high, allow_low=True) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not low <= value <= high
        or (not allow_low and value == low)
    ):
        raise ValueError(f"Invalid {name}: {value!r}")

    return float(value)


class VDNPolicy(BasePolicy):
    """Value Decomposition Networks: independent execution, joint training."""

    kind = "vdn"

    def __init__(
        self,
        spec,
        n_actions: int,
        seed: int,
        share_parameters: bool = True,
        team_reward: str = "mean",
        hidden_size: int = 64,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 20000,
        batch_size: int = 32,
        target_update: int = 200,
    ):
        super().__init__(spec, n_actions, seed)

        if type(share_parameters) is not bool:
            raise ValueError(
                f"Invalid share_parameters: {share_parameters!r}"
            )

        if team_reward not in ("mean", "sum"):
            raise ValueError(
                f"Invalid team_reward: {team_reward!r}"
            )

        buffer_size = _validated_integer(
            "buffer_size", buffer_size, 1, 1_000_000
        )

        self.hparams = {
            "share_parameters": share_parameters,
            "team_reward": team_reward,
            "hidden_size": _validated_integer(
                "hidden_size", hidden_size, 1, 1024
            ),
            "learning_rate": _validated_float(
                "learning_rate", learning_rate, 0.0, 1.0, allow_low=False
            ),
            "gamma": _validated_float("gamma", gamma, 0.0, 1.0),
            "epsilon": _validated_float("epsilon", epsilon, 0.0, 1.0),
            "epsilon_min": _validated_float(
                "epsilon_min", epsilon_min, 0.0, 1.0
            ),
            "epsilon_decay": _validated_float(
                "epsilon_decay", epsilon_decay, 0.0, 1.0, allow_low=False
            ),
            "buffer_size": buffer_size,
            "batch_size": _validated_integer(
                "batch_size", batch_size, 1, buffer_size
            ),
            "target_update": _validated_integer(
                "target_update", target_update, 1, 1_000_000
            ),
        }

        self.learner: VDNTeamLearner | None = None

    @classmethod
    def link_policies(cls, policies: dict, config):
        teams: dict[str, list] = {}

        for spec in config.agents:
            teams.setdefault(spec.team, []).append(spec)

        for team, specs in teams.items():
            members = [
                policies[spec.id]
                for spec in specs
                if isinstance(policies[spec.id], cls)
            ]

            if not members:
                continue

            if len(members) != len(specs):
                others = sorted(
                    spec.id
                    for spec in specs
                    if not isinstance(policies[spec.id], cls)
                )

                raise ValueError(
                    f"Team {team!r} mixes VDN with other policies "
                    f"({', '.join(others)}); every member of a "
                    f"VDN team must use the vdn policy"
                )

            if any(
                member.hparams != members[0].hparams
                for member in members
            ):
                raise ValueError(
                    f"VDN team {team!r} has members with different "
                    f"hyperparameters; they must all be identical"
                )

            # world.seed + agent index, so stable across processes.
            learner = VDNTeamLearner(
                team=team,
                members=[member.spec.id for member in members],
                hparams=members[0].hparams,
                world=config.world,
                n_actions=members[0].n_actions,
                seed=members[0].seed,
            )

            for member in members:
                member.learner = learner

    def _linked_learner(self) -> VDNTeamLearner:
        if self.learner is None:
            raise RuntimeError(
                f"VDN policy for agent {self.spec.id} has no team "
                f"learner; it must be created via build_policies()"
            )

        return self.learner

    def q_values(self, observation) -> np.ndarray:
        learner = self._linked_learner()

        return learner.q_values(
            self.spec.id,
            learner.features(self.spec.id, observation),
        )

    def act(self, observation, training: bool = False) -> int:
        learner = self._linked_learner()

        learner.on_act(training)

        if training and self.rng.random() < learner.epsilon:
            return self.rng.randrange(self.n_actions)

        values = self.q_values(observation)

        return self.rng.choice(
            np.flatnonzero(values == values.max()).tolist()
        )

    def learn(
        self,
        observation,
        action,
        reward,
        next_observation,
        done,
    ):
        learner = self._linked_learner()

        value = float(reward)

        if not math.isfinite(value):
            raise ValueError("Non-finite training reward")

        learner.record(
            self.spec.id,
            learner.features(self.spec.id, observation),
            action,
            value,
            (
                None
                if next_observation is None
                else learner.features(self.spec.id, next_observation)
            ),
            done,
        )

    def end_episode(self):
        self._linked_learner().end_episode(self.spec.id)

    def save_state(self) -> dict:
        learner = self._linked_learner()

        return {
            "kind": self.kind,
            "n_actions": self.n_actions,
            "team": learner.team,
            "members": list(learner.members),
            "input_dim": learner.input_dim,
            "hparams": dict(self.hparams),
            **learner.state_dict(),
        }

    def load_state(self, data: dict):
        learner = self._linked_learner()

        if data.get("kind") != self.kind:
            raise ValueError("Checkpoint policy type mismatch")

        if data.get("n_actions") != self.n_actions:
            raise ValueError("Checkpoint action count mismatch")

        if data.get("members") != learner.members:
            raise ValueError("Checkpoint team members mismatch")

        if data.get("input_dim") != learner.input_dim:
            raise ValueError("Checkpoint observation size mismatch")

        hparams = data.get("hparams")

        if not isinstance(hparams, dict) or any(
            hparams.get(name) != self.hparams[name]
            for name in ("share_parameters", "hidden_size")
        ):
            raise ValueError("Checkpoint VDN architecture mismatch")

        learner.load_state_dict(data)

    def copy_to(self, other):
        learner = self._linked_learner()

        if not isinstance(other, VDNPolicy):
            raise ValueError("Incompatible VDN destination policy")

        destination = other._linked_learner()

        if (
            other.n_actions != self.n_actions
            or destination.members != learner.members
            or destination.input_dim != learner.input_dim
            or any(
                other.hparams[name] != self.hparams[name]
                for name in ("share_parameters", "hidden_size")
            )
        ):
            raise ValueError("Incompatible VDN destination policy")

        learner.copy_weights_to(destination)
