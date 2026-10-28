import { useMemo, useState } from "react";

import {
  BrainCircuit,
  Play,
  Radio,
} from "lucide-react";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  isTrainingFinished,
  isTrainingSuccessful,
} from "../hooks/useTraining";

import type {
  EpisodeEvent,
  TrainingStatus,
  AgentOptimization
} from "../types/simulation";

interface TrainingPanelProps {
  status: TrainingStatus | null;
  episodes: EpisodeEvent[];

  optimizations: Record<string, AgentOptimization>;
  optimizationStarted: boolean;

  starting: boolean;
  streamConnected: boolean;

  onStart: (episodes: number) => void;
}

export default function TrainingPanel({
  status,
  episodes,

  optimizations,
  optimizationStarted,

  starting,
  streamConnected,
  onStart,
}: TrainingPanelProps) {
  const [count, setCount] = useState(250);

  const running = Boolean(
    status && !isTrainingFinished(status.status)
  );

  const completed = status?.completed_episodes ?? 0;
  const total = status?.episodes ?? 0;

  const percentage = total > 0
    ? Math.min(100, Math.max(0, completed / total * 100))
    : 0;

  const optimizationItems = Object.values(
    optimizations
  );

  const optimizedAgents = optimizationItems.filter(
    (item) => item.status === "completed"
  ).length;

  const showOptimization =
    optimizationStarted ||
    (
      status !== null &&
      total > 0 &&
      completed >= total &&
      !isTrainingFinished(status.status)
    );

  const displayedStatus =
    isTrainingSuccessful(status?.status)
      ? "Completed"
      : optimizationStarted
        ? "Optimizing agents"
        : status?.status ?? "Not started";

  const chartData = useMemo(
    () => episodes.map((episode) => ({
      episode: episode.episode,

      reward: Object.values(
        episode.team_rewards ?? {}
      ).reduce((a, b) => a + b, 0),
    })),
    [episodes]
  );

  return (
    <div className="panel-card">
      <div className="card-heading">
        <div>
          <div className="eyebrow">REINFORCEMENT LEARNING</div>
          <h3><BrainCircuit size={18} /> Training</h3>
        </div>

        {running && (
          <span className="live-indicator">
            <Radio size={13} />
            LIVE
          </span>
        )}
      </div>

      <div className="training-start">
        <label className="field">
          <span>Episodes</span>
          <input
            type="number"
            min={1}
            step={1}
            value={count}
            disabled={running || starting}
            onChange={(e) => setCount(Number(e.target.value))}
          />
        </label>

        <button
          className="primary-button"
          disabled={
            starting ||
            running ||
            !Number.isInteger(count) ||
            count < 1
          }
          onClick={() => onStart(count)}
        >
          <Play size={15} />
          {starting ? "Starting..." : "Train"}
        </button>
      </div>

      {status && (
        <>
          <div className="stat-line">
            <span>Status</span>
            <strong>{displayedStatus}</strong>
          </div>

          <div className="stat-line">
            <span>Episodes completed</span>
            <strong>{completed} / {total}</strong>
          </div>

          <div className="progress-track">
            <div
              className="progress-fill"
              style={{ width: `${percentage}%` }}
            />
          </div>

          <div className="progress-caption">
            <span>{percentage.toFixed(1)}%</span>
            <span>
              {streamConnected
                ? "SSE connected"
                : "HTTP update"}
            </span>
          </div>

          {showOptimization && (
            <div className="optimization-section">
              <div className="optimization-heading">
                <div>
                  <div className="mini-heading">
                    POST-TRAINING OPTIMIZATION
                  </div>

                  <h4>Agent optimization</h4>
                </div>

                <span className="optimization-count">
                  {optimizedAgents}/{optimizationItems.length}
                </span>
              </div>

              <div className="optimization-list">
                {optimizationItems.map((item) => {
                  const done = item.status === "completed";

                  const hasTotal =
                    item.total_evaluations !== null &&
                    item.total_evaluations > 0;

                  const percentage = done
                    ? 100
                    : hasTotal
                      ? Math.min(
                          100,
                          Math.max(
                            0,
                            item.evaluations /
                              item.total_evaluations! * 100
                          )
                        )
                      : 0;

                  const statusLabel = done
                    ? "Completed"
                    : item.status === "running"
                      ? "Optimizing"
                      : "Pending";

                  return (
                    <div
                      className="optimization-agent"
                      key={item.agent_id}
                    >
                      <div className="optimization-agent-header">
                        <strong>{item.agent_id}</strong>

                        <span
                          className={
                            done
                              ? "optimization-status completed"
                              : "optimization-status"
                          }
                        >
                          {statusLabel}
                        </span>
                      </div>

                      <div className="progress-track">
                        <div
                          className={
                            done
                              ? "progress-fill optimization-done"
                              : "progress-fill"
                          }
                          style={{
                            width: `${percentage}%`,
                          }}
                        />
                      </div>

                      <div className="progress-caption">
                        <span>
                          {item.total_evaluations !== null
                            ? `${item.evaluations} / ${item.total_evaluations} evaluations`
                            : `${item.evaluations} evaluations`}
                        </span>

                        <span>
                          {done
                            ? "100%"
                            : hasTotal
                              ? `${percentage.toFixed(1)}%`
                              : "—"}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {!isTrainingFinished(status.status) && (
                <div className="helper-text">
                  Training episodes are complete or optimization
                  has started. The training remains active until
                  the backend reports a terminal status.
                </div>
              )}
            </div>
          )}

          {chartData.length > 0 && (
            <div className="chart-container">
              <div className="mini-heading">
                Total reward per episode
              </div>

              <ResponsiveContainer width="100%" height={155}>
                <LineChart data={chartData.slice(-100)}>
                  <CartesianGrid
                    stroke="#243247"
                    vertical={false}
                  />

                  <XAxis
                    dataKey="episode"
                    tick={{ fill: "#8699ad", fontSize: 10 }}
                    axisLine={false}
                    tickLine={false}
                  />

                  <YAxis
                    tick={{ fill: "#8699ad", fontSize: 10 }}
                    width={36}
                    axisLine={false}
                    tickLine={false}
                  />

                  <Tooltip
                    contentStyle={{
                      background: "#152438",
                      color: "#e7f0f8",
                      border: "1px solid #34475c",
                      borderRadius: 10,
                    }}
                  />

                  <Line
                    type="monotone"
                    dataKey="reward"
                    stroke="#55e3c1"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {episodes.length > 0 && (
            <div className="recent-episodes">
              <div className="mini-heading">
                Recent episodes
              </div>

              {episodes.slice(-4).reverse().map((episode) => (
                <div
                  className="episode-row"
                  key={episode.episode}
                >
                  <strong>#{episode.episode}</strong>
                  <span>{episode.steps} steps</span>
                  <span>
                    {episode.winner ?? "No winner"}
                  </span>
                </div>
              ))}
            </div>
          )}

          {status.error && (
            <div className="error-inline">
              {status.error}
            </div>
          )}
        </>
      )}

      {!status && (
        <div className="empty-description">
          Configure your agents and start training.
          Episodes and their rewards will appear here.
        </div>
      )}
    </div>
  );
}