import {
  BrainCircuit,
  Grid3X3,
  Shield,
  Target,
  Users,
  Zap,
} from "lucide-react";

import { teamColor } from "../lib/config";

import type {
  SimulationConfig,
  TrainingStatus,
  WorldState,
} from "../types/simulation";

interface TrainingOverviewPanelProps {
  training: TrainingStatus | null;
  state: WorldState | null;
}

function ParametersList({
  values,
}: {
  values: Record<string, string | number | boolean>;
}) {
  const entries = Object.entries(values);

  if (!entries.length) {
    return (
      <div className="helper-text">
        No additional parameters.
      </div>
    );
  }

  return (
    <div className="training-parameters">
      {entries.map(([name, value]) => (
        <div className="stat-line" key={name}>
          <span>{name}</span>
          <strong>{String(value)}</strong>
        </div>
      ))}
    </div>
  );
}

function OriginalConfiguration({
  config,
}: {
  config: SimulationConfig;
}) {
  return (
    <>
      <details className="config-section" open>
        <summary>
          <span>
            <Grid3X3 size={17} />
            World
          </span>
        </summary>

        <div className="section-body">
          {[
            ["Width", config.world.width],
            ["Height", config.world.height],
            ["Max steps", config.world.max_steps],
            ["Max health", config.world.max_health],
            ["Vision radius", config.world.vision_radius],
            ["Seed", config.world.seed],
            [
              "Goal",
              config.world.goal?.join(", ") ?? "None",
            ],
            [
              "Obstacles",
              config.world.obstacles.length,
            ],
          ].map(([label, value]) => (
            <div className="stat-line" key={label}>
              <span>{label}</span>
              <strong>{value}</strong>
            </div>
          ))}
        </div>
      </details>

      <details className="config-section" open>
        <summary>
          <span>
            <Users size={17} />
            Agents
          </span>

          <span className="section-meta">
            {config.agents.length}
          </span>
        </summary>

        <div className="section-body">
          {config.agents.map((agent) => (
            <div className="training-overview-item" key={agent.id}>
              <div className="training-overview-title">
                <span
                  className="team-indicator"
                  style={{
                    background: teamColor(agent.team),
                  }}
                />

                <strong>{agent.id}</strong>
              </div>

              <div className="stat-line">
                <span>Team</span>
                <strong>{agent.team}</strong>
              </div>

              <div className="stat-line">
                <span>Position</span>
                <strong>{agent.position.join(", ")}</strong>
              </div>

              <div className="stat-line">
                <span>Policy</span>
                <strong>{agent.policy}</strong>
              </div>

              <ParametersList values={agent.params} />
            </div>
          ))}
        </div>
      </details>

      <details className="config-section">
        <summary>
          <span>
            <Zap size={17} />
            Rules
          </span>

          <span className="section-meta">
            {config.rules.length}
          </span>
        </summary>

        <div className="section-body">
          {config.rules.map((rule, index) => (
            <div className="training-overview-item" key={index}>
              <strong>{rule.name}</strong>

              <ParametersList values={rule.options} />
            </div>
          ))}
        </div>
      </details>

      <details className="config-section">
        <summary>
          <span>
            <Target size={17} />
            Rewards
          </span>

          <span className="section-meta">
            {config.rewards.length}
          </span>
        </summary>

        <div className="section-body">
          {config.rewards.map((reward, index) => (
            <div className="training-overview-item" key={index}>
              <strong>{reward.kind}</strong>

              <div className="stat-line">
                <span>Weight</span>
                <strong>{reward.weight}</strong>
              </div>

              <ParametersList values={reward.values} />
            </div>
          ))}
        </div>
      </details>
    </>
  );
}

export default function TrainingOverviewPanel({
  training,
  state,
}: TrainingOverviewPanelProps) {
  const config = training?.config ?? null;

  return (
    <div className="config-panel">
      <div className="panel-heading">
        <div>
          <div className="eyebrow">
            TRAINED CONFIGURATION
          </div>

          <h2>Training overview</h2>
        </div>

        <Shield size={18} color="#55e3c1" />
      </div>

      {!training && (
        <div className="notice">
          Enter a training ID to load an existing
          training from the API.
        </div>
      )}

      {training && !config && !state && (
        <div className="notice">
          The training was found, but the API does
          not expose its original configuration.

          Create a trained session to load the actual
          world and agent states.
        </div>
      )}

      {config && (
        <>
          <div className="notice">
            This is the original training configuration.
            It is displayed read-only to preserve
            compatibility with the trained policies.
          </div>

          <OriginalConfiguration config={config} />
        </>
      )}

      {state && (
        <>
          {!config && (
            <div className="notice">
              This information comes from the session
              state. Original policies, rules, rewards,
              and training parameters are unavailable.
            </div>
          )}

          <details className="config-section" open>
            <summary>
              <span>
                <Grid3X3 size={17} />
                Current world
              </span>

              <span className="section-meta">
                {state.width} × {state.height}
              </span>
            </summary>

            <div className="section-body">
              <div className="stat-line">
                <span>Step</span>
                <strong>{state.step}</strong>
              </div>

              <div className="stat-line">
                <span>Goal</span>

                <strong>
                  {state.goal?.join(", ") ?? "None"}
                </strong>
              </div>

              <div className="stat-line">
                <span>Obstacles</span>
                <strong>{state.obstacles.length}</strong>
              </div>

              <div className="stat-line">
                <span>Agents</span>
                <strong>{state.actors.length}</strong>
              </div>
            </div>
          </details>

          <details className="config-section" open>
            <summary>
              <span>
                <Users size={17} />
                Current agents
              </span>
            </summary>

            <div className="section-body">
              {state.actors.map((actor) => (
                <div
                  className="training-overview-item"
                  key={actor.id}
                >
                  <div className="training-overview-title">
                    <span
                      className="team-indicator"
                      style={{
                        background: teamColor(actor.team),
                      }}
                    />

                    <strong>{actor.id}</strong>
                  </div>

                  <div className="stat-line">
                    <span>Team</span>
                    <strong>{actor.team}</strong>
                  </div>

                  <div className="stat-line">
                    <span>Position</span>

                    <strong>
                      {actor.position.join(", ")}
                    </strong>
                  </div>

                  <div className="stat-line">
                    <span>Health</span>
                    <strong>{actor.health}</strong>
                  </div>

                  <div className="stat-line">
                    <span>Status</span>

                    <strong>
                      {actor.alive ? "Alive" : "Eliminated"}
                    </strong>
                  </div>
                </div>
              ))}
            </div>
          </details>

          <details className="config-section">
            <summary>
              <span>
                <BrainCircuit size={17} />
                Available actions
              </span>
            </summary>

            <div className="section-body">
              {state.action_space.map((action) => (
                <div
                  className="stat-line"
                  key={action.index}
                >
                  <span>
                    {action.index} · {action.name}
                  </span>

                  <strong>Code {action.code}</strong>
                </div>
              ))}
            </div>
          </details>
        </>
      )}
    </div>
  );
}