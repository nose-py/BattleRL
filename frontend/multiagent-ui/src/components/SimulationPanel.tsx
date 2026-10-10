import {
  Pause,
  Play,
  Plus,
  RotateCcw,
  SkipForward,
  Trash2,
  Trophy,
} from "lucide-react";

import {
  teamColor,
} from "../lib/config";

import type {
  SessionResponse,
} from "../types/simulation";


const EVENT_LABELS: Record<string, string> = {
  move: "Movement",
  invalid_move: "Invalid move",
  miss: "Missed shot",
  hit: "Hit",
  kill: "Elimination",
  goal: "Goal reached",
  first_goal: "First goal",
};

function describeEvent(event: {
  type: string;
  actor: string;
  target: string | null;
  step: number;
}): string {
  const name = EVENT_LABELS[event.type] ?? event.type;

  return [
    `[${event.step}]`,
    name,
    `· ${event.actor}`,
    event.target ? `→ ${event.target}` : "",
  ].filter(Boolean).join(" ");
}

interface SimulationPanelProps {
  session: SessionResponse | null;

  maxSteps: number;

  busy: boolean;
  stepping: boolean;
  playing: boolean;

  speed: number;
  onSpeed: (value: number) => void;

  useTrained: boolean;
  canUseTrained: boolean;
  onUseTrained: (value: boolean) => void;

  onCreate: () => void;
  onStep: () => void;
  onTogglePlay: () => void;
  onReset: () => void;
  onDelete: () => void;
}

export default function SimulationPanel({
  session,
  maxSteps,
  busy,
  stepping,
  playing,
  speed,
  onSpeed,
  useTrained,
  canUseTrained,
  onUseTrained,
  onCreate,
  onStep,
  onTogglePlay,
  onReset,
  onDelete,
}: SimulationPanelProps) {
  const state = session?.state;

  const teams = state
    ? [...new Set(state.actors.map((a) => a.team))]
    : [];

  const finished = state?.finished ?? false;

  return (
    <div className="panel-card">
      <div className="card-heading">
        <div>
          <div className="eyebrow">SIMULATION ENGINE</div>
          <h3><Play size={17} /> Session</h3>
        </div>

        <span className="session-state">
          {state
            ? finished
              ? "Finished"
              : "Active"
            : "Not started"}
        </span>
      </div>

      <label className="check-row">
        <input
          type="checkbox"
          checked={useTrained}
          disabled={!canUseTrained || busy}
          onChange={(e) => onUseTrained(e.target.checked)}
        />

        <span>
          Use completed training
          {!canUseTrained && (
            <small>
              Train first to enable this option.
            </small>
          )}
        </span>
      </label>

      <button
        className="primary-button full"
        disabled={busy}
        onClick={onCreate}
      >
        <Plus size={16} />
        {session ? "Crear otra sesión" : "Crear sesión"}
      </button>

      {state && (
        <>
          <div className="session-divider" />

          <div className="session-stats">
            <div>
              <span>CURRENT STEP</span>
              <strong>{state.step}</strong>
            </div>

            <div>
              <span>ALIVE AGENTS</span>
              <strong>
                {state.actors.filter((a) => a.alive).length}
                <small> / {state.actors.length}</small>
              </strong>
            </div>
          </div>

          <div className="stat-line">
            <span>Configured limit</span>
            <strong>{maxSteps} steps</strong>
          </div>

          <div className="playback-controls">
            <button
              title="Reset episode"
              className="control-button"
              disabled={busy}
              onClick={onReset}
            >
              <RotateCcw size={17} />
            </button>

            <button
              title="Execute one step"
              className="control-button"
              disabled={busy || stepping || playing || finished}
              onClick={onStep}
            >
              <SkipForward size={18} />
            </button>

            <button
              title={playing ? "Pause" : "Play"}
              className="play-button"
              disabled={busy || finished}
              onClick={onTogglePlay}
            >
              {playing
                ? <Pause size={20} fill="currentColor" />
                : <Play size={20} fill="currentColor" />}
            </button>
          </div>

          <label className="speed-control">
            <div className="stat-line">
              <span>Interval between steps</span>
              <strong>{speed} ms</strong>
            </div>

            <input
              type="range"
              min={250}
              max={2000}
              step={50}
              value={speed}
              onChange={(e) => onSpeed(Number(e.target.value))}
            />
          </label>

          {finished && (
            <div className="winner-card">
              <Trophy size={19} />

              <div>
                <strong>
                  {state.winner
                    ? `Winner: ${state.winner}`
                    : "Simulation finished"}
                </strong>

                <small>
                  {state.finish_reason ?? "Episode finished"}
                </small>
              </div>
            </div>
          )}

          <div className="mini-heading team-heading">
            Teams
          </div>

          {teams.map((team) => {
            const actors = state.actors.filter(
              (a) => a.team === team
            );

            return (
              <div className="team-card" key={team}>
                <div className="team-card-title">
                  <span
                    className="team-indicator"
                    style={{
                      background: teamColor(team),
                    }}
                  />

                  <strong>{team}</strong>

                  <span className="muted">
                    {actors.filter((a) => a.alive).length}
                    /{actors.length} alive
                  </span>
                </div>

                <div className="team-metrics">
                  <span>
                    Kills: {actors.reduce(
                      (sum, a) => sum + a.kills, 0
                    )}
                  </span>

                  <span>
                    Goals: {actors.reduce(
                      (sum, a) => sum + a.goals, 0
                    )}
                  </span>
                </div>
              </div>
            );
          })}

          <div className="mini-heading team-heading">
            State events
          </div>

          <div className="event-list">
            {(state.events ?? []).length ? (
              state.events.slice(-10).map((event, index) => (
                <div className="event-entry" key={index}>
                  <span className="event-bullet" />

                  <code title={JSON.stringify(event)}>
                    {describeEvent(event)}
                  </code>
                </div>
              ))
            ) : (
              <div className="empty-description">
                No events have been recorded yet.
              </div>
            )}
          </div>

          <button
            className="delete-button"
            disabled={busy}
            onClick={onDelete}
          >
            <Trash2 size={15} />
            Delete session
          </button>
        </>
      )}
    </div>
  );
}