import { useMemo, useState } from "react";

import {
  Activity,
  Coins,
  TrendingUp,
} from "lucide-react";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { teamColor } from "../lib/config";

import type {
  ActorState,
  StepSnapshot,
} from "../types/simulation";

type ViewMode = "teams" | "agents";
type MetricMode = "step" | "total";

interface RewardsPanelProps {
  history: StepSnapshot[];
  actors: ActorState[];

  actionsCatalog: Record<string, string>;

  currentStep: number;
  sessionActive: boolean;
}

const COLORS = [
  "#55e3c1",
  "#ffad70",
  "#80aaff",
  "#f187cb",
  "#d7bd6e",
  "#a58aff",
];

function formatReward(value: number | null | undefined) {
  if (value == null) return "—";

  return value.toLocaleString("es-ES", {
    maximumFractionDigits: 3,
  });
}

function rewardClass(value: number | null | undefined) {
  if (value == null) return "reward-neutral";
  if (value > 0) return "reward-positive";
  if (value < 0) return "reward-negative";
  return "reward-neutral";
}

function actionLabel(
  actionId: number | undefined,
  catalog: Record<string, string>
): string {
  if (actionId === undefined) {
    return "—";
  }

  return catalog[String(actionId)] ?? `Acción #${actionId}`;
}

function statusLabel(
  actor: ActorState,
  snapshot?: StepSnapshot
): string {
  if (!snapshot) return "Sin ejecutar";

  if (snapshot.truncations[actor.id] === true) {
    return "Límite de tiempo";
  }

  if (!actor.alive) {
    return "Eliminado";
  }

  if (snapshot.terminations[actor.id] === true) {
    return "Episodio terminado";
  }

  return "Activo";
}

/**
 * Calculates the sum of the rewards assigned
 * to the members of each team.
 *
 * Does not modify or redefine the reward
 * function of the simulator.
 */
function aggregateTeamReward(
  rewards: Record<string, number>,
  actors: ActorState[],
  team: string
): number | null {
  const values = actors
    .filter((a) => a.team === team)
    .map((a) => rewards[a.id])
    .filter((v): v is number => v !== undefined);

  if (!values.length) return null;

  return values.reduce((sum, value) => sum + value, 0);
}

export default function RewardsPanel({
  history,
  actors,
  actionsCatalog,
  currentStep,
  sessionActive,
}: RewardsPanelProps) {
  const [viewMode, setViewMode] =
    useState<ViewMode>("teams");

  const [metricMode, setMetricMode] =
    useState<MetricMode>("step");

  const orderedHistory = useMemo(
    () => [...history].sort((a, b) => a.step - b.step),
    [history]
  );

  const latest = orderedHistory.find(
    (item) => item.step === currentStep
  );

  const teams = useMemo(
    () => [...new Set(actors.map((a) => a.team))].sort(),
    [actors]
  );

  const agentIds = useMemo(
    () => actors.map((a) => a.id).sort(),
    [actors]
  );

  const selectedIds =
    viewMode === "teams" ? teams : agentIds;

  /**
   * Value of a team/agent in a snapshot:
   *
   * step  -> rewards
   * total -> total_rewards
   */
  function valueFor(
    snapshot: StepSnapshot,
    id: string
  ): number | null {
    const values = metricMode === "step"
      ? snapshot.rewards
      : snapshot.total_rewards;

    if (viewMode === "agents") {
      return values[id] ?? null;
    }

    return aggregateTeamReward(values, actors, id);
  }

  const chartData = orderedHistory.map((snapshot) => {
    const row: Record<string, number | null> = {
      step: snapshot.step,
    };

    for (const id of selectedIds) {
      row[`value_${id}`] = valueFor(snapshot, id);
    }

    return row;
  });

  const teamRows = teams.map((team) => ({
    id: team,

    last: latest
      ? aggregateTeamReward(latest.rewards, actors, team)
      : null,

    total: latest
      ? aggregateTeamReward(
          latest.total_rewards,
          actors,
          team
        )
      : null,

    alive: actors.filter(
      (a) => a.team === team && a.alive
    ).length,

    count: actors.filter(
      (a) => a.team === team
    ).length,
  }));

  const agentRows = actors.map((actor) => ({
    ...actor,

    last: latest?.rewards[actor.id] ?? null,
    total: latest?.total_rewards[actor.id] ?? null,

    action: actionLabel(
      latest?.actions[actor.id],
      actionsCatalog
    ),

    status: statusLabel(actor, latest),
  }));

  const chartColor = (id: string, index: number) => {
    if (viewMode === "teams") {
      return teamColor(id);
    }

    return COLORS[index % COLORS.length];
  };

  return (
    <section className="rewards-panel">
      <div className="rewards-heading">
        <div>
          <div className="eyebrow">
            SIMULATION ANALYTICS
          </div>

          <h3>
            <Coins size={18} />
            Rewards and actions
          </h3>
        </div>

        <span className="rewards-step-label">
          <Activity size={14} />
          Paso {currentStep}
        </span>
      </div>

      {!sessionActive && (
        <div className="empty-description">
          Create a session to view the rewards
          and actions of the agents.
        </div>
      )}

      {sessionActive && history.length === 0 && (
        <div className="empty-description">
          Execute a step to start recording
          the rewards and actions.
        </div>
      )}

      {sessionActive && history.length > 0 && (
        <>
          {/* Team aggregated rewards */}
          <div className="rewards-subheading">
            Rewards by team
          </div>

          <div className="reward-table-scroll">
            <table className="reward-history-table">
              <thead>
                <tr>
                  <th>Team</th>
                  <th>Alive agents</th>
                  <th>Last step</th>
                  <th>Total</th>
                </tr>
              </thead>

              <tbody>
                {teamRows.map((team) => (
                  <tr key={team.id}>
                    <td>
                      <span className="reward-name">
                        <span
                          className="team-indicator"
                          style={{
                            background: teamColor(team.id),
                          }}
                        />
                        {team.id}
                      </span>
                    </td>

                    <td>
                      {team.alive}/{team.count}
                    </td>

                    <td className={rewardClass(team.last)}>
                      {formatReward(team.last)}
                    </td>

                    <td className={rewardClass(team.total)}>
                      {formatReward(team.total)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Rewards and actions by agent */}
          <div className="rewards-subheading">
            Rewards and actions by agent
          </div>

          <div className="reward-table-scroll">
            <table className="reward-history-table">
              <thead>
                <tr>
                  <th>Agent</th>
                  <th>Team</th>
                  <th>Action</th>
                  <th>Last step</th>
                  <th>Total</th>
                  <th>Status</th>
                </tr>
              </thead>

              <tbody>
                {agentRows.map((agent) => (
                  <tr key={agent.id}>
                    <td>{agent.id}</td>

                    <td>
                      <span className="reward-name">
                        <span
                          className="team-indicator"
                          style={{
                            background: teamColor(agent.team),
                          }}
                        />
                        {agent.team}
                      </span>
                    </td>

                    <td title={agent.action}>
                      {agent.action}
                    </td>

                    <td className={rewardClass(agent.last)}>
                      {formatReward(agent.last)}
                    </td>

                    <td className={rewardClass(agent.total)}>
                      {formatReward(agent.total)}
                    </td>

                    <td>{agent.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Charts */}
          <div className="reward-chart-heading">
            <div>
              <div className="mini-heading">
                EVOLUTION
              </div>

              <h3>
                <TrendingUp size={17} />
                Reward history
              </h3>
            </div>

            <div className="reward-chart-controls">
              <div className="segmented">
                <button
                  className={
                    viewMode === "teams" ? "active" : ""
                  }
                  onClick={() => setViewMode("teams")}
                >
                  Teams
                </button>

                <button
                  className={
                    viewMode === "agents" ? "active" : ""
                  }
                  onClick={() => setViewMode("agents")}
                >
                  Agents
                </button>
              </div>

              <div className="segmented">
                <button
                  className={
                    metricMode === "step" ? "active" : ""
                  }
                  onClick={() => setMetricMode("step")}
                >
                  Per step
                </button>

                <button
                  className={
                    metricMode === "total" ? "active" : ""
                  }
                  onClick={() => setMetricMode("total")}
                >
                  Accumulated
                </button>
              </div>
            </div>
          </div>

          <div className="reward-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={chartData}
                margin={{
                  top: 10,
                  right: 15,
                  bottom: 5,
                  left: 0,
                }}
              >
                <CartesianGrid
                  stroke="#283c50"
                  strokeDasharray="3 5"
                  vertical={false}
                />

                <XAxis
                  dataKey="step"
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  allowDecimals={false}
                  tick={{
                    fill: "#8fa7ba",
                    fontSize: 11,
                  }}
                  axisLine={false}
                  tickLine={false}
                />

                <YAxis
                  width={48}
                  tick={{
                    fill: "#8fa7ba",
                    fontSize: 11,
                  }}
                  axisLine={false}
                  tickLine={false}
                />

                <Tooltip
                  contentStyle={{
                    background: "#16283c",
                    color: "#e8f0f8",
                    border: "1px solid #3a5267",
                    borderRadius: 9,
                  }}
                />

                <Legend />

                {selectedIds.map((id, index) => (
                  <Line
                    key={id}
                    name={id}
                    type="linear"
                    dataKey={`value_${id}`}
                    stroke={chartColor(id, index)}
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 5 }}
                    isAnimationActive={false}
                    connectNulls={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </section>
  );
}