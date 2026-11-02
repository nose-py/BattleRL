from typing import Literal

PolicyName = Literal[
    "random",
    "q_learning",
    "double_q",
    "map_elites",
]

PolicyParams = {
    "random": {},
    "q_learning": {
        "alpha": ["float", 0.1],
        "gamma": ["float", 0.95],
        "epsilon": ["float", 1],
        "epsilon_min": ["float", 0.01],
        "epsilon_decay": ["float", 0.99],
    },
    "double_q": {
        "alpha": ["float", 0.1],
        "gamma": ["float", 0.95],
        "epsilon": ["float", 1],
        "epsilon_min": ["float", 0.01],
        "epsilon_decay": ["float", 0.99],
    },
    "map_elites": {
        "iterations": ["integer", 400],
        "evaluation_episodes": ["integer", 3],
        "hidden_size": ["integer", 16],
        "bins": ["integer", 8],
        "mutation_sigma": ["float", 0.2],
        "mutation_rate": ["float", 0.15],
        "random_injection": ["float", 0.10],
    }
}