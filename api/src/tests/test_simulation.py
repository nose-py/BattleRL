from fastapi.testclient import TestClient

from src.api import app
from src.sim.config import SimulationConfig
from src.sim.environment import GridWorldEnv
from src.sim.runner import SimulationRunner


def make_config(
    agents=None,
    rules=None,
    rewards=None,
    max_health=1,
):
    if agents is None:
        agents = [
            {
                "id": "red_1",
                "team": "red",
                "position": [0, 0],
                "policy": "q_learning",
            },
            {
                "id": "blue_1",
                "team": "blue",
                "position": [4, 4],
                "policy": "random",
            },
        ]

    return SimulationConfig.model_validate({
        "world": {
            "width": 5,
            "height": 5,
            "max_steps": 20,
            "max_health": max_health,
            "vision_radius": 1,
            "goal": [1, 0],
            "seed": 42,
        },
        "agents": agents,
        "rules": rules if rules is not None else [
            {"name": "movement"},
            {
                "name": "goal",
                "options": {"end_episode": True},
            },
        ],
        "rewards": rewards if rewards is not None else [
            {
                "kind": "event",
                "values": {"goal": 10},
            },
        ],
    })


def test_goal_rule():
    config = make_config()
    env = GridWorldEnv(config)

    observations, infos = env.reset()

    assert len(observations) == 2

    obs, rewards, terminated, truncated, infos = (
        env.step({
            "red_1": 2,
            "blue_1": 0,
        })
    )

    assert env.world.finished
    assert env.world.winner == "red"
    assert rewards["red_1"] == 10.0
    assert terminated["red_1"]
    assert terminated["blue_1"]


def test_team_reward():
    agents = [
        {
            "id": "red_1",
            "team": "red",
            "position": [0, 0],
            "policy": "q_learning",
        },
        {
            "id": "red_2",
            "team": "red",
            "position": [0, 1],
            "policy": "q_learning",
        },
        {
            "id": "blue_1",
            "team": "blue",
            "position": [1, 0],
            "policy": "random",
        },
    ]

    config = make_config(
        agents=agents,
        rules=[{"name": "shooting"}],
        rewards=[
            {
                "kind": "team_event",
                "weight": 2.0,
                "values": {"kill": 1.0},
            },
        ],
        max_health=1,
    )

    env = GridWorldEnv(config)
    env.reset()

    obs, rewards, terminated, truncated, infos = (
        env.step({
            "red_1": 6,
            "red_2": 0,
            "blue_1": 0,
        })
    )

    assert not env.world.actors["blue_1"].alive
    assert rewards["red_1"] == 2.0
    assert rewards["red_2"] == 2.0
    assert rewards["blue_1"] == 0.0


def test_training_updates_q_table():
    config = make_config()

    runner = SimulationRunner(config)

    policy = runner.policies["red_1"]

    assert len(policy.q) == 0

    runner.run_episode(
        episode_number=1,
        training=True,
    )

    assert len(policy.q) > 0


def test_checkpoint_roundtrip():
    config = make_config()
    runner = SimulationRunner(config)

    runner.run_episode(1, training=True)

    original = runner.policies["red_1"]
    saved = original.save_state()

    new_runner = SimulationRunner(config)
    restored = new_runner.policies["red_1"]

    restored.load_state(saved)

    assert restored.save_state() == saved


def test_create_interactive_session():
    client = TestClient(app)

    config = make_config()

    response = client.post(
        "/sessions",
        json={
            "config": config.model_dump(mode="json")
        },
    )

    assert response.status_code == 200

    session_id = response.json()["session_id"]

    response = client.post(
        f"/sessions/{session_id}/step",
        json={"actions": {}},
    )

    assert response.status_code == 200
    assert response.json()["step"] == 1

    client.delete(f"/sessions/{session_id}")