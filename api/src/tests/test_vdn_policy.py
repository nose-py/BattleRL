import json

import numpy as np
import pytest

from src.sim.config import SimulationConfig
from src.sim.policies.names import PolicyParams
from src.sim.runner import SimulationRunner


GOAL = [3, 2]


def vdn_params(**overrides):
    params = {
        name: spec[1]
        for name, spec in PolicyParams["vdn"].items()
    }

    return params | overrides


def vdn_agent(agent_id, position, **overrides):
    return {
        "id": agent_id,
        "team": "red",
        "position": position,
        "policy": "vdn",
        "params": vdn_params(**overrides),
    }


def opponent(policy="random", params=None, position=(4, 4)):
    return {
        "id": "blue_1",
        "team": "blue",
        "position": list(position),
        "policy": policy,
        "params": params or {},
    }


def make_config(
    red_params=None,
    blue=None,
    agents=None,
    rules=None,
    max_steps=20,
    seed=42,
):
    red_params = red_params or {}

    return SimulationConfig.model_validate({
        "world": {
            "width": 5,
            "height": 5,
            "max_steps": max_steps,
            "max_health": 1,
            "vision_radius": 1,
            "goal": GOAL,
            "seed": seed,
        },
        "agents": agents if agents is not None else [
            vdn_agent("red_1", [0, 0], **red_params),
            vdn_agent("red_2", [0, 1], **red_params),
            blue or opponent(),
        ],
        "rules": rules if rules is not None else [
            {"name": "movement"},
            {
                "name": "goal",
                "options": {"end_episode": True},
            },
        ],
        "rewards": [
            {
                "kind": "team_event",
                "values": {"goal": 10},
            },
        ],
    })


def trained_runner(episodes=3, **config_options):
    runner = SimulationRunner(make_config(**config_options))

    for episode in range(episodes):
        runner.run_episode(episode, training=True)

    return runner


def test_catalog_defaults_build_shared_team_learner():
    runner = SimulationRunner(make_config())

    red_1 = runner.policies["red_1"]
    red_2 = runner.policies["red_2"]

    assert red_1.learner is not None
    assert red_1.learner is red_2.learner
    assert len(red_1.learner.networks) == 1
    assert len(red_1.learner.target_networks) == 1


def test_separate_networks_per_agent():
    runner = SimulationRunner(
        make_config({"share_parameters": False})
    )

    learner = runner.policies["red_1"].learner

    assert learner is runner.policies["red_2"].learner
    assert len(learner.networks) == 2
    assert len(learner.target_networks) == 2


def test_team_mixing_vdn_with_other_policy_is_rejected():
    agents = [
        vdn_agent("red_1", [0, 0]),
        {
            "id": "red_2",
            "team": "red",
            "position": [0, 1],
            "policy": "random",
            "params": {},
        },
        opponent(),
    ]

    with pytest.raises(ValueError, match="mixes VDN"):
        SimulationRunner(make_config(agents=agents))


def test_team_with_different_hyperparameters_is_rejected():
    agents = [
        vdn_agent("red_1", [0, 0]),
        vdn_agent("red_2", [0, 1], gamma=0.5),
        opponent(),
    ]

    with pytest.raises(ValueError, match="different"):
        SimulationRunner(make_config(agents=agents))


def test_one_transition_per_team_step():
    runner = SimulationRunner(make_config())

    result = runner.run_episode(1, training=True)

    assert result["steps"] > 0
    assert runner.policies["red_1"].learner.buf_size == result["steps"]


def test_dead_agent_is_masked_out_of_joint_transition():
    runner = SimulationRunner(make_config(
        rules=[
            {"name": "movement"},
            {"name": "shooting"},
            {"name": "elimination"},
        ],
    ))

    learner = runner.policies["red_1"].learner

    for episode in range(100):
        runner.run_episode(episode, training=True)

        active = learner.buf_mask[:learner.buf_size].sum(axis=1)

        if np.any(active == 1):
            break

    assert np.any(active == 1)
    assert np.all(active >= 1)


def test_learns_to_reach_goal():
    runner = SimulationRunner(make_config({
        "share_parameters": False,
        "epsilon_decay": 0.97,
    }))

    for episode in range(150):
        runner.run_episode(episode, training=True)

    result = runner.run_episode(1000, training=False)

    assert result["winner"] == "red"
    assert result["steps"] <= 6


def test_checkpoint_round_trip_restores_q_values():
    runner = trained_runner(episodes=5)
    runner.reset(seed=1)
    observation = runner.observations["red_1"]

    restored = SimulationRunner(make_config())

    # A checkpoint is written per member and loaded once per member.
    for agent_id in ("red_1", "red_2"):
        state = json.loads(
            json.dumps(runner.policies[agent_id].save_state())
        )
        restored.policies[agent_id].load_state(state)

    source = runner.policies["red_1"]
    target = restored.policies["red_1"]

    np.testing.assert_allclose(
        target.q_values(observation),
        source.q_values(observation),
    )
    assert target.learner.epsilon == source.learner.epsilon
    assert (
        target.learner.gradient_steps
        == source.learner.gradient_steps
    )


def test_copy_to_copies_weights_exactly():
    source = trained_runner(episodes=5)
    destination = SimulationRunner(make_config())

    source.policies["red_1"].copy_to(destination.policies["red_1"])

    source_learner = source.policies["red_1"].learner
    destination_learner = destination.policies["red_1"].learner

    for mine, theirs in zip(
        source_learner.networks + source_learner.target_networks,
        destination_learner.networks + destination_learner.target_networks,
    ):
        for mine_param, their_param in zip(mine.params, theirs.params):
            np.testing.assert_array_equal(mine_param, their_param)
            assert mine_param is not their_param

    assert destination_learner.epsilon == source_learner.epsilon


def test_optimize_runs_against_map_elites_opponent():
    params = {
        name: spec[1]
        for name, spec in PolicyParams["map_elites"].items()
    } | {
        "iterations": 3,
        "evaluation_episodes": 1,
        "hidden_size": 4,
        "bins": 4,
    }

    runner = trained_runner(
        blue=opponent("map_elites", params),
    )

    results = runner.optimize_policies()

    assert set(results) == {"blue_1"}
