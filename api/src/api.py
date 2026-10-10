import asyncio
import json
import threading
from dataclasses import dataclass
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, model_validator
from sse_starlette.sse import EventSourceResponse

from src.sim.runner import SimulationRunner
from src.sim.config import SimulationConfig
from src.sim.jobs import (
    TrainingManager,
    load_trained_runner,
    safe_training_path,
)
from src.sim.policies.registry import POLICY_REGISTRY
from src.sim.rewards.registry import REWARD_REGISTRY
from src.sim.rules.actions import ACTION_NAMES
from src.sim.rules.registry import RULE_REGISTRY


app = FastAPI(
    title="Modular Multi-Agent Simulation API",
    version="1.0.0",
)

training_manager = TrainingManager()


class TrainingRequest(BaseModel):
    config: SimulationConfig
    episodes: int = Field(default=1000, ge=1, le=100000)


class SessionRequest(BaseModel):
    config: SimulationConfig | None = None
    training_id: str | None = None

    @model_validator(mode="after")
    def validate_source(self):
        if (self.config is None) == (self.training_id is None):
            raise ValueError(
                "Indicate either config or training_id, but not both"
            )

        return self


class StepRequest(BaseModel):
    actions: dict[str, int] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    seed: int | None = None


@dataclass
class Session:
    runner: SimulationRunner
    lock: threading.Lock


sessions: dict[str, Session] = {}
sessions_lock = threading.Lock()


def get_session(session_id: str) -> Session:
    with sessions_lock:
        session = sessions.get(session_id)

    if session is None:
        raise HTTPException(404, "Session not found")

    return session


@app.get("/catalog")
def catalog():
    return {
        "policies": list(POLICY_REGISTRY),
        "rules": list(RULE_REGISTRY),
        "rewards": list(REWARD_REGISTRY),
        "actions": ACTION_NAMES,
    }


@app.post("/trainings", status_code=202)
def create_training(request: TrainingRequest):
    try:
        job = training_manager.start(
            config=request.config,
            episodes=request.episodes,
        )
    except RuntimeError as exc:
        raise HTTPException(429, str(exc))

    return {
        "training_id": job.training_id,
        "status": "starting",
        "status_url": f"/trainings/{job.training_id}",
        "events_url": f"/trainings/{job.training_id}/events",
    }


@app.get("/trainings/{training_id}")
def get_training(training_id: str):
    job = training_manager.get(training_id)

    if job is not None:
        return job.snapshot()

    # Allows retrieving completed trainings
    # after the server has been restarted.
    try:
        folder = safe_training_path(training_id)
    except ValueError:
        raise HTTPException(404, "Training not found")

    manifest_path = folder / "manifest.json"

    if not manifest_path.is_file():
        raise HTTPException(404, "Training not found")

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    return {
        "training_id": training_id,
        "status": "completed",
        "episodes": manifest["episodes"],
        "completed_episodes": manifest["episodes"],
    }


@app.get("/trainings/{training_id}/events")
async def training_events(
    training_id: str,
    request: Request,
):
    job = training_manager.get(training_id)

    if job is None:
        raise HTTPException(
            404,
            "Stream not available for this training",
        )

    try:
        cursor = max(
            0,
            int(request.headers.get("last-event-id", "0")),
        )
    except ValueError:
        raise HTTPException(
            400, "Invalid Last-Event-ID"
        )

    async def generate():
        nonlocal cursor

        while True:
            events, finished = job.events_since(cursor)

            for event in events:
                cursor = event["id"]

                yield {
                    "id": str(cursor),
                    "event": event["type"],
                    "data": json.dumps(event["data"]),
                }

            if finished:
                break

            if await request.is_disconnected():
                break

            await asyncio.sleep(0.25)

    return EventSourceResponse(generate())


@app.post("/sessions")
def create_session(request: SessionRequest):
    if request.training_id is not None:
        try:
            runner = load_trained_runner(
                request.training_id
            )
        except (FileNotFoundError, ValueError):
            raise HTTPException(
                404,
                "Training not found or incomplete",
            )
    else:
        runner = SimulationRunner(request.config) # pyright: ignore[reportArgumentType]

    session_id = str(uuid4())

    with sessions_lock:
        if len(sessions) >= 100:
            raise HTTPException(
                429, "Session limit reached"
            )

        sessions[session_id] = Session(
            runner=runner,
            lock=threading.Lock(),
        )

    return {
        "session_id": session_id,
        "state": runner.snapshot(),
    }


@app.get("/sessions/{session_id}")
def get_session_state(session_id: str):
    session = get_session(session_id)

    with session.lock:
        return session.runner.snapshot()


@app.post("/sessions/{session_id}/step")
def step_session(
    session_id: str,
    request: StepRequest,
):
    session = get_session(session_id)

    with session.lock:
        try:
            return session.runner.step(
                manual_actions=request.actions,
                training=False,
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc))


@app.post("/sessions/{session_id}/reset")
def reset_session(
    session_id: str,
    request: ResetRequest,
):
    session = get_session(session_id)

    with session.lock:
        return session.runner.reset(
            seed=request.seed
        )


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    with sessions_lock:
        session = sessions.pop(session_id, None)

    if session is None:
        raise HTTPException(404, "Sesión no encontrada")

    session.runner.env.close()

    return {"deleted": True}