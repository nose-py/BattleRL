"""Enabled rules alone decide the discrete action space."""

import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.sim.config import SimulationConfig
from src.sim.environment import GridWorldEnv
from src.sim.runner import SimulationRunner
from src.sim.rules.action_catalog import ActionCatalog
from src.sim.rules.base_rule import BaseRule


def config_for(*rule_names, policy="random"):
    return SimulationConfig.model_validate({
        "world": {"width": 5, "height": 5, "max_steps": 3, "max_health": 1, "goal": [1, 0]},
        "agents": [
            {"id": "red", "team": "r", "position": [0, 0], "policy": policy,
             "params": ({"alpha": 0.15, "gamma": 0.95, "epsilon": 0.5,
                         "epsilon_min": 0.05, "epsilon_decay": 0.99}
                        if policy == "q_learning" else {})},
            {"id": "blue", "team": "b", "position": [1, 0], "policy": "random"},
        ],
        "rules": [{"name": name} for name in rule_names],
        "rewards": [{"kind": "constant", "weight": -0.01}],
    })


def test_movement_only_uses_five_actions():
    env = GridWorldEnv(config_for("movement"))
    assert env.action_space("red").n == 5
    assert env.action_catalog.codes == (0, 1, 2, 3, 4)
    assert "shoot_up" not in env.action_catalog.names
    env.reset()
    with pytest.raises(ValueError):
        env.step({"red": 5, "blue": 0})


def test_shooting_only_reindexes_actions_and_executes_shot():
    env = GridWorldEnv(config_for("shooting"))
    assert env.action_space("red").n == 5
    assert env.action_catalog.codes == (0, 5, 6, 7, 8)
    assert env.action_catalog.index_for_code(6) == 2
    env.reset()
    env.step({"red": 2, "blue": 0})
    assert not env.world.actors["blue"].alive
    assert env.world.actors["red"].kills == 1


def test_action_union_is_deduplicated_and_order_independent():
    first = GridWorldEnv(config_for("movement", "shooting", "goal"))
    second = GridWorldEnv(config_for("shooting", "movement", "goal"))
    assert first.action_catalog.describe() == second.action_catalog.describe()
    assert first.action_catalog.codes == tuple(range(9))
    assert first.action_space("red").n == 9


def test_q_learning_uses_only_available_actions():
    runner = SimulationRunner(config_for("movement", policy="q_learning"))
    assert runner.policies["red"].n_actions == 5
    runner.run_episode(1)
    assert all(len(values) == 5 for values in runner.policies["red"].q.values())


def test_snapshot_exposes_actual_mapping():
    runner = SimulationRunner(config_for("shooting"))
    assert runner.snapshot()["action_space"][2] == {
        "index": 2, "code": 6, "name": "shoot_right"
    }
    result = runner.step(manual_actions={"red": 6, "blue": 0})

    assert result is not None
    assert result["action_space"][2]["name"] == "shoot_right"
    assert result["actions"]["red"] == 2 


def test_checkpoint_action_signature(tmp_path, monkeypatch):
    from src.sim import jobs
    monkeypatch.setattr(jobs, "DATA_ROOT", tmp_path)
    
    runner = SimulationRunner(config_for("movement", policy="q_learning"))
    runner.run_episode(1)
    
    training_id = str(uuid4())
    
    folder = jobs.safe_training_path(training_id)
    folder.mkdir()
    
    manifest = {"config": runner.config.model_dump(mode="json"),
                "action_space": runner.env.action_catalog.describe()}
    
    (folder / "manifest.json").write_text(json.dumps(manifest))
    
    for agent_id, policy in runner.policies.items():
        (folder / f"{agent_id}.json").write_text(json.dumps(policy.save_state()))
    
    assert jobs.load_trained_runner(training_id).env.action_catalog.n_actions == 5
    
    manifest["action_space"][1]["name"] = "shoot_up"
    
    (folder / "manifest.json").write_text(json.dumps(manifest))
    
    with pytest.raises(ValueError, match="Checkpoint action space differs"):
        jobs.load_trained_runner(training_id)
