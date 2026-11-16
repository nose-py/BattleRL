import json
import multiprocessing as mp
from multiprocessing.context import SpawnProcess
import os
import queue
import threading
from copy import deepcopy
from collections import deque
from pathlib import Path
from uuid import UUID, uuid4

from src.sim.config import SimulationConfig
from src.sim.runner import SimulationRunner


DATA_ROOT = Path(
    os.environ.get("SIM_DATA_DIR", "./data")
).resolve()

DATA_ROOT.mkdir(parents=True, exist_ok=True)


def safe_training_path(training_id: str) -> Path:
    normalized = str(UUID(training_id))

    return DATA_ROOT / normalized

# Default values used by older AgentSpec versions.
_LEGACY_Q_DEFAULTS = {
    "alpha": 0.15,
    "gamma": 0.95,
    "epsilon": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
}

def read_training_manifest(training_id: str) -> dict:
    path = safe_training_path(training_id) / "manifest.json"

    if not path.is_file():
        raise FileNotFoundError(
            "Training manifest not found"
        )

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Invalid training manifest JSON"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            "Training manifest must be a JSON object"
        )

    return data


def config_from_training_manifest(
    manifest: dict,
) -> SimulationConfig:
    saved = manifest.get("config")

    if not isinstance(saved, dict):
        raise ValueError(
            "Training manifest has no configuration"
        )

    data = deepcopy(saved)

    for agent in data.get("agents", []):
        if not isinstance(agent, dict):
            continue

        legacy = {
            name: agent.pop(name)
            for name in _LEGACY_Q_DEFAULTS
            if name in agent
        }

        if legacy and agent.get("policy") in (
            "q_learning",
            "double_q",
        ):
            params = agent.setdefault("params", {})

            if not isinstance(params, dict):
                raise ValueError(
                    "Agent params must be a JSON object"
                )

            for name, default in _LEGACY_Q_DEFAULTS.items():
                params.setdefault(
                    name,
                    legacy.get(name, default),
                )

    return SimulationConfig.model_validate(data)

def train_worker(
    config_data: dict,
    episodes: int,
    training_id: str,
    output_queue,
):
    try:
        config = SimulationConfig.model_validate(config_data)

        runner = SimulationRunner(config)
        folder = safe_training_path(training_id)

        folder.mkdir(parents=True, exist_ok=True)

        history_path = folder / "episodes.jsonl"

        with history_path.open("w", encoding="utf-8") as history:
            for episode in range(1, episodes + 1):
                result = runner.run_episode(
                    episode_number=episode,
                    training=True,
                )

                serialized = json.dumps(result)
                
                history.write(serialized + "\n")
                if episode % 50 == 0:
                    history.flush()

                output_queue.put({
                    "type": "episode",
                    "data": result,
                })

        runner.optimize_policies(
            on_event=lambda event_type, data: (
                output_queue.put({
                    "type": event_type,
                    "data": data,
                })
            )
        )

        for agent_id, policy in runner.policies.items():
            checkpoint = folder / f"{agent_id}.json"

            checkpoint.write_text(
                json.dumps(policy.save_state()),
                encoding="utf-8",
            )

            policy.save_artifacts(folder)

        manifest = {
            "training_id": training_id,
            "episodes": episodes,
            "config": config.model_dump(mode="json"),
            "action_space": runner.env.action_catalog.describe(),
        }

        (folder / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        output_queue.put({
            "type": "completed",
            "data": {
                "training_id": training_id,
                "episodes": episodes,
            },
        })

    except Exception as exc:
        output_queue.put({
            "type": "error",
            "data": {
                "message": f"{type(exc).__name__}: {exc}",
            },
        })


class TrainingJob:
    """
    Class representing a training job, including its status, progress, and events.
    """
    def __init__(self, training_id, episodes):
        self.training_id = training_id
        self.episodes = episodes

        self.status = "starting"
        self.completed_episodes = 0
        self.last_episode = None
        self.error = None

        self.events = deque(maxlen=2000)
        self.sequence = 0
        self.lock = threading.Lock()

        self.process: SpawnProcess | None = None

    def add_event(self, message):
        with self.lock:
            self.sequence += 1

            event = {
                "id": self.sequence,
                "type": message["type"],
                "data": message["data"],
            }

            self.events.append(event)

            if message["type"] == "episode":
                self.status = "running"
                self.completed_episodes += 1
                self.last_episode = message["data"]

            elif message["type"] == "optimization_started":
                self.status = "optimizing"

            elif message["type"] == "completed":
                self.status = "completed"

            elif message["type"] == "error":
                self.status = "error"
                self.error = message["data"]["message"]

    def events_since(self, cursor):
        with self.lock:
            events = [
                dict(event)
                for event in self.events
                if event["id"] > cursor
            ]

            finished = self.status in ("completed", "error")

            return events, finished

    def snapshot(self):
        with self.lock:
            return {
                "training_id": self.training_id,
                "status": self.status,
                "episodes": self.episodes,
                "completed_episodes": self.completed_episodes,
                "last_episode": self.last_episode,
                "error": self.error,
            }


class TrainingManager:
    """
    Class responsible for managing multiple training jobs, including starting new jobs,
    monitoring their progress, and handling concurrency.
    """
    def __init__(self, max_jobs=2):
        self.jobs = {}
        self.max_jobs = max_jobs
        self.lock = threading.Lock()

        self.context = mp.get_context("spawn")

    def start(self, config: SimulationConfig, episodes: int):
        with self.lock:
            active = sum(
                job.snapshot()["status"] not in (
                    "completed", "error"
                )
                for job in self.jobs.values()
            )

            if active >= self.max_jobs:
                raise RuntimeError(
                    "Maximum number of concurrent training jobs reached."
                )

            training_id = str(uuid4())
            job = TrainingJob(training_id, episodes)

            output_queue = self.context.Queue(maxsize=256)

            process = self.context.Process(
                target=train_worker,
                args=(
                    config.model_dump(mode="json"),
                    episodes,
                    training_id,
                    output_queue,
                ),
                daemon=False,
            )

            job.process = process
            self.jobs[training_id] = job

            try:
                process.start()
            except Exception:
                del self.jobs[training_id]
                raise

        thread = threading.Thread(
            target=self._monitor,
            args=(job, output_queue),
            daemon=True,
        )
        thread.start()

        return job

    def _monitor(self, job, output_queue):
        while True:
            try:
                message = output_queue.get(timeout=0.5)

            except queue.Empty:
                if not job.process.is_alive():
                    status = job.snapshot()["status"]

                    if status not in ("completed", "error"):
                        job.add_event({
                            "type": "error",
                            "data": {
                                "message": (
                                    "Process ended without "
                                    "terminal result"
                                ),
                            },
                        })
                    break

                continue

            job.add_event(message)

            if message["type"] in ("completed", "error"):
                break

        job.process.join(timeout=2)
        output_queue.close()

    def get(self, training_id):
        return self.jobs.get(training_id)


def load_trained_runner(training_id: str):
    try:
        folder = safe_training_path(training_id)
    except ValueError:
        raise FileNotFoundError("Invalid training ID")

    manifest = read_training_manifest(training_id)
    config = config_from_training_manifest(manifest)

    runner = SimulationRunner(config)

    saved_actions = manifest.get("action_space")
    current_actions = runner.env.action_catalog.describe()

    if saved_actions is None:
        if list(runner.env.action_catalog.codes) != list(range(9)):
            raise ValueError(
                "Legacy checkpoint has no compatible "
                "action-space signature"
            )

    elif saved_actions != current_actions:
        raise ValueError(
            "Checkpoint action space differs "
            "from the configured rules"
        )

    for agent_id, policy in runner.policies.items():
        checkpoint_path = folder / f"{agent_id}.json"

        state = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        )

        policy.load_state(state)
        policy.load_artifacts(folder)

    return runner