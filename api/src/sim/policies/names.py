from typing import Literal

PolicyName = Literal[
    "random",
    "q_learning",
    "double_q",
    "map_elites",
    "vdn",
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
        "behavior_descriptor": [["entropy_coverage", "entropy_switching", "coverage_change", "switching_change"], "entropy_coverage"],
        "iterations": ["integer", 400],
        "evaluation_episodes": ["integer", 3],
        "hidden_size": ["integer", 16],
        "bins": ["integer", 8],
        "mutation_sigma": ["float", 0.2],
        "mutation_rate": ["float", 0.15],
        "random_injection": ["float", 0.10],
    },
    "vdn": {
        "share_parameters": ["boolean", True],
        "team_reward": [["mean", "sum"], "mean"],
        "hidden_size": ["integer", 64],
        "learning_rate": ["float", 0.001],
        "gamma": ["float", 0.95],
        "epsilon": ["float", 1.0],
        "epsilon_min": ["float", 0.05],
        "epsilon_decay": ["float", 0.995],
        "buffer_size": ["integer", 20000],
        "batch_size": ["integer", 32],
        "target_update": ["integer", 200],
    },
}