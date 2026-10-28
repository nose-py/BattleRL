from typing import Literal

PolicyName = Literal[
    "random",
    "q_learning",
    "double_q",
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
}