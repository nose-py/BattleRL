import numpy as np

def validate_checkpoint(data, kind, n_actions):
    if data.get("kind") != kind:
        raise ValueError("Policy kind incompatible")

    if data.get("n_actions") != n_actions:
        raise ValueError("Action space incompatible")


def validate_table(data, n_actions):
    if not isinstance(data, dict):
        raise ValueError("Invalid Q table")

    result = {}

    for key, values in data.items():
        if not isinstance(values, list):
            raise ValueError("Invalid table entry")

        if len(values) != n_actions:
            raise ValueError("Incompatible table dimension")

        result[key] = [float(v) for v in values]

    return result


def state_key(observation) -> str:
    view = np.asarray(
        observation["view"], dtype=np.int64
    ).ravel()

    own = np.asarray(
        observation["self"], dtype=np.int64
    ).ravel()

    values = np.concatenate([view, own])

    return ",".join(map(str, values.tolist()))