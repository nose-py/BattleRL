import { useState } from "react";

import {
  Crosshair,
  Map,
  MousePointer2,
  Plus,
  Target,
  Trash2,
  Users,
  Zap,
} from "lucide-react";

import {
  newAgent,
  teamColor,
} from "../lib/config";

import type {
  Catalog,
  EditTool,
  Point,
  SimulationConfig,
} from "../types/simulation";

interface ConfigPanelProps {
  config: SimulationConfig;
  catalog: Catalog;

  onChange: (config: SimulationConfig) => void;

  selectedAgentId: string;
  onSelectAgent: (id: string) => void;

  editTool: EditTool;
  onEditTool: (tool: EditTool) => void;

  sceneLocked: boolean;
}

type OptionValue = string | number | boolean;

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number | "any";
}) {
  return (
    <label className="field">
      <span>{label}</span>

      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

function DictionaryEditor({
  values,
  onChange,
  numbersOnly = false,
}: {
  values: Record<string, OptionValue>;
  onChange: (values: Record<string, OptionValue>) => void;
  numbersOnly?: boolean;
}) {
  const entries = Object.entries(values);

  const updateValue = (
    key: string,
    value: OptionValue
  ) => {
    onChange({
      ...values,
      [key]: value,
    });
  };

  const rename = (oldKey: string, newKey: string) => {
    if (!newKey || (newKey !== oldKey && newKey in values)) {
      return;
    }

    onChange(
      Object.fromEntries(
        entries.map(([key, value]) => [
          key === oldKey ? newKey : key,
          value,
        ])
      )
    );
  };

  const add = () => {
    let index = 1;
    let key = `param_${index}`;

    while (key in values) {
      index++;
      key = `param_${index}`;
    }

    onChange({
      ...values,
      [key]: numbersOnly ? 0 : true,
    });
  };

  return (
    <div className="dict-editor">
      {entries.map(([key, value]) => {
        const valueType = typeof value;

        return (
          <div className="dict-row" key={key}>
            <input
              aria-label="Nombre del parámetro"
              value={key}
              onChange={(e) => rename(key, e.target.value)}
            />

            {!numbersOnly && (
              <select
                aria-label="Tipo"
                value={valueType}
                onChange={(e) => {
                  const type = e.target.value;

                  updateValue(
                    key,
                    type === "boolean"
                      ? true
                      : type === "number"
                        ? 0
                        : ""
                  );
                }}
              >
                <option value="boolean">Bool</option>
                <option value="number">Num</option>
                <option value="string">Text</option>
              </select>
            )}

            {typeof value === "boolean" && !numbersOnly ? (
              <select
                aria-label="Valor"
                value={String(value)}
                onChange={(e) =>
                  updateValue(key, e.target.value === "true")
                }
              >
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            ) : (
              <input
                aria-label="Valor"
                type={numbersOnly || valueType === "number"
                  ? "number"
                  : "text"}
                step="any"
                value={String(value)}
                onChange={(e) =>
                  updateValue(
                    key,
                    numbersOnly || valueType === "number"
                      ? Number(e.target.value)
                      : e.target.value
                  )
                }
              />
            )}

            <button
              className="icon-button danger"
              title="Delete parameter"
              onClick={() => {
                const next = { ...values };
                delete next[key];
                onChange(next);
              }}
            >
              <Trash2 size={14} />
            </button>
          </div>
        );
      })}

      <button className="subtle-button" onClick={add}>
        <Plus size={14} />
        Add parameter
      </button>
    </div>
  );
}

export default function ConfigPanel({
  config,
  catalog,
  onChange,
  selectedAgentId,
  onSelectAgent,
  editTool,
  onEditTool,
  sceneLocked,
}: ConfigPanelProps) {
  const [message, setMessage] = useState("");

  function update(
    callback: (draft: SimulationConfig) => void
  ) {
    const draft = structuredClone(config);
    callback(draft);
    onChange(draft);
  }

  function changeWorld(
    key: keyof SimulationConfig["world"],
    value: number
  ) {
    update((draft) => {
      (draft.world as unknown as Record<string, unknown>)[key] =
        value;
    });
  }

  function addAgent() {
    const occupied = new Set([
      ...config.agents.map((a) => a.position.join(",")),
      ...config.world.obstacles.map((p) => p.join(",")),
    ]);

    let position: Point | null = null;

    for (let y = 0; y < config.world.height; y++) {
      for (let x = 0; x < config.world.width; x++) {
        if (!occupied.has(`${x},${y}`)) {
          position = [x, y];
          break;
        }
      }

      if (position) break;
    }

    if (!position) {
      setMessage("No available cells.");
      return;
    }

    let number = config.agents.length + 1;
    let id = `agent_${number}`;

    while (config.agents.some((a) => a.id === id)) {
      number++;
      id = `agent_${number}`;
    }

    const team = config.agents[0]?.team ?? "Alpha";
    const policy = catalog.policies[0] ?? "";

    update((draft) => {
      draft.agents.push(
        newAgent(id, team, policy, position!)
      );
    });

    onSelectAgent(id);
    setMessage("");
  }

  function addRule() {
    const available = catalog.rules.find(
      (name) => !config.rules.some((r) => r.name === name)
    );

    if (!available) return;

    update((draft) => {
      draft.rules.push({
        name: available,
        options: {},
      });
    });
  }

  function addReward() {
    const name = catalog.rewards[0];
    if (!name) return;

    update((draft) => {
      draft.rewards.push({
        kind: name,
        weight: 1,
        values: {},
      });
    });
  }

  const tools: {
    id: EditTool;
    label: string;
    icon: typeof MousePointer2;
  }[] = [
    { id: "inspect", label: "Inspeccionar", icon: MousePointer2 },
    { id: "obstacle", label: "Obstáculo", icon: Map },
    { id: "goal", label: "Objetivo", icon: Target },
    { id: "agent", label: "Ubicar agente", icon: Crosshair },
  ];

  return (
    <div className="config-panel">
      <div className="panel-heading">
        <div>
          <div className="eyebrow">EDITOR</div>
          <h2>Configuración</h2>
        </div>

        <span className="panel-count">WORLD</span>
      </div>

      {sceneLocked && (
        <div className="notice">
          There is an active session. Draft changes
          will be applied when creating a new session without training.
          To edit the board, delete the current session.
        </div>
      )}

      <details className="config-section" open>
        <summary>
          <span><Map size={17} /> Mundo</span>
          <span className="section-meta">
            {config.world.width} × {config.world.height}
          </span>
        </summary>

        <div className="section-body">
          <div className="two-fields">
            <NumberField
              label="Width"
              min={3}
              value={config.world.width}
              onChange={(v) => changeWorld("width", v)}
            />

            <NumberField
              label="Height"
              min={3}
              value={config.world.height}
              onChange={(v) => changeWorld("height", v)}
            />
          </div>

          <div className="two-fields">
            <NumberField
              label="Maximum Steps"
              min={1}
              max={10000}
              value={config.world.max_steps}
              onChange={(v) => changeWorld("max_steps", v)}
            />

            <NumberField
              label="Maximum Health"
              min={1}
              value={config.world.max_health}
              onChange={(v) => changeWorld("max_health", v)}
            />
          </div>

          <div className="two-fields">
            <NumberField
              label="Vision Radius"
              min={1}
              value={config.world.vision_radius}
              onChange={(v) => changeWorld("vision_radius", v)}
            />

            <NumberField
              label="Seed"
              value={config.world.seed}
              onChange={(v) => changeWorld("seed", v)}
            />
          </div>

          <div className="field">
            <span>Goal (X, Y)</span>

            {config.world.goal ? (
              <div className="two-fields">
                <NumberField
                  label="X"
                  value={config.world.goal[0]}
                  min={0}
                  onChange={(v) => update((d) => {
                    if (d.world.goal) d.world.goal[0] = v;
                  })}
                />

                <NumberField
                  label="Y"
                  value={config.world.goal[1]}
                  min={0}
                  onChange={(v) => update((d) => {
                    if (d.world.goal) d.world.goal[1] = v;
                  })}
                />
              </div>
            ) : (
              <span className="muted">No goal</span>
            )}

            <button
              className="subtle-button"
              onClick={() => update((d) => {
                d.world.goal = d.world.goal
                  ? null
                  : [
                      Math.floor(d.world.width / 2),
                      Math.floor(d.world.height / 2),
                    ];
              })}
            >
              {config.world.goal
                ? "Remove goal"
                : "Create goal"}
            </button>
          </div>

          <div className="field">
            <span>Edit the 3D board</span>

            <div className="tool-grid">
              {tools.map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  className={
                    editTool === id
                      ? "tool-button active"
                      : "tool-button"
                  }
                  disabled={sceneLocked && id !== "inspect"}
                  onClick={() => onEditTool(id)}
                  title={label}
                >
                  <Icon size={17} />
                  <span>{label}</span>
                </button>
              ))}
            </div>

            <div className="helper-text">
              Choose a tool and click on a cell.
              Obstacle and goal also work as toggles.
            </div>
          </div>
        </div>
      </details>

      <details className="config-section" open>
        <summary>
          <span><Users size={17} /> Agents</span>
          <span className="section-meta">
            {config.agents.length}
          </span>
        </summary>

        <div className="section-body">
          <div className="agent-selector">
  <div className="mini-heading">
    Selected Agent Information
  </div>

  <div className="agent-selector-list">
              {config.agents.map((agent) => (
                <button
                  key={agent.id}
                  type="button"
                  className={
                    selectedAgentId === agent.id
                      ? "agent-selector-item selected"
                      : "agent-selector-item"
                  }
                  onClick={() => {
                    onSelectAgent(agent.id);

                    if (!sceneLocked) {
                      onEditTool("agent");
                    }
                  }}
                >
                  <span
                    className="team-indicator"
                    style={{
                      background: teamColor(agent.team),
                    }}
                  />

                  <span className="agent-selector-info">
                    <strong>{agent.id}</strong>
                    <small>
                      {agent.team} · {agent.policy}
                    </small>
                  </span>

                  {selectedAgentId === agent.id && (
                    <span className="selected-tag">
                      ACTIVE
                    </span>
                  )}
                </button>
              ))}
            </div>

            <div className="helper-text">
              {sceneLocked
                ? "Active session: selection available for inspection."
                : `Selected: ${selectedAgentId}. Click on a free cell to move it.`}
            </div>
          </div>
          {config.agents.map((agent, index) => (
            <details
              className="agent-editor"
              key={`${index}-${agent.id}`}
            >
              <summary>
                <span
                  className="team-indicator"
                  style={{ background: teamColor(agent.team) }}
                />

                <span className="agent-title">
                  <strong>{agent.id}</strong>
                  <small>
                    {agent.team} · {agent.policy}
                  </small>
                </span>

                <span className="section-meta">
                  {agent.position.join(", ")}
                </span>
              </summary>

              <div className="agent-fields">
                <label className="field">
                  <span>ID</span>

                  <input
                    value={agent.id}
                    maxLength={40}
                    onChange={(e) => {
                      const id = e.target.value;

                      update((d) => {
                        d.agents[index].id = id;
                      });

                      if (selectedAgentId === agent.id) {
                        onSelectAgent(id);
                      }
                    }}
                  />
                </label>

                <label className="field">
                  <span>Team</span>

                  <input
                    value={agent.team}
                    maxLength={40}
                    onChange={(e) => update((d) => {
                      d.agents[index].team = e.target.value;
                    })}
                  />
                </label>

                <label className="field">
                  <span>Algorithm / Policy</span>

                  <select
                    value={agent.policy}
                    onChange={(e) => update((d) => {
                      d.agents[index].policy = e.target.value;
                    })}
                  >
                    {catalog.policies.map((policy) => (
                      <option key={policy} value={policy}>
                        {policy}
                      </option>
                    ))}
                  </select>
                </label>

                <div className="two-fields">
                  <NumberField
                    label="X Position"
                    min={0}
                    value={agent.position[0]}
                    onChange={(v) => update((d) => {
                      d.agents[index].position[0] = v;
                    })}
                  />

                  <NumberField
                    label="Y Position"
                    min={0}
                    value={agent.position[1]}
                    onChange={(v) => update((d) => {
                      d.agents[index].position[1] = v;
                    })}
                  />
                </div>

                <div className="two-fields">
                  <NumberField
                    label="Alpha"
                    min={0}
                    max={1}
                    step={0.01}
                    value={agent.alpha}
                    onChange={(v) => update((d) => {
                      d.agents[index].alpha = v;
                    })}
                  />

                  <NumberField
                    label="Gamma"
                    min={0}
                    max={1}
                    step={0.01}
                    value={agent.gamma}
                    onChange={(v) => update((d) => {
                      d.agents[index].gamma = v;
                    })}
                  />
                </div>

                <NumberField
                  label="Epsilon"
                  min={0}
                  max={1}
                  step={0.001}
                  value={agent.epsilon}
                  onChange={(v) => update((d) => {
                    d.agents[index].epsilon = v;
                  })}
                />

                <div className="two-fields">
                  <NumberField
                    label="Minimum Epsilon"
                    min={0}
                    max={1}
                    step={0.001}
                    value={agent.epsilon_min}
                    onChange={(v) => update((d) => {
                      d.agents[index].epsilon_min = v;
                    })}
                  />

                  <NumberField
                    label="Epsilon Decay"
                    min={0}
                    max={1}
                    step={0.001}
                    value={agent.epsilon_decay}
                    onChange={(v) => update((d) => {
                      d.agents[index].epsilon_decay = v;
                    })}
                  />
                </div>

                <div className="agent-actions">
                  <button
                    className="subtle-button"
                    disabled={sceneLocked}
                    onClick={() => {
                      onSelectAgent(agent.id);
                      onEditTool("agent");
                    }}
                  >
                    <Crosshair size={14} />
                    Locate in the field
                  </button>

                  <button
                    className="subtle-button danger-text"
                    disabled={config.agents.length <= 2}
                    onClick={() => update((d) => {
                      d.agents.splice(index, 1);
                    })}
                  >
                    <Trash2 size={14} />
                    Delete
                  </button>
                </div>
              </div>
            </details>
          ))}

          <button
            className="add-button"
            onClick={addAgent}
          >
            <Plus size={16} />
            Add agent
          </button>

          {message && (
            <div className="helper-text">{message}</div>
          )}
        </div>
      </details>

      <details className="config-section">
        <summary>
          <span><Zap size={17} /> Rules</span>
          <span className="section-meta">
            {config.rules.length}
          </span>
        </summary>

        <div className="section-body">
          {config.rules.map((rule, index) => (
            <div className="rule-editor" key={index}>
              <div className="editor-heading">
                <select
                  value={rule.name}
                  onChange={(e) => update((d) => {
                    d.rules[index].name = e.target.value;
                    d.rules[index].options = {};
                  })}
                >
                  {catalog.rules.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>

                <button
                  className="icon-button danger"
                  disabled={config.rules.length <= 1}
                  onClick={() => update((d) => {
                    d.rules.splice(index, 1);
                  })}
                >
                  <Trash2 size={15} />
                </button>
              </div>

              <div className="mini-heading">Options</div>

              <DictionaryEditor
                values={rule.options}
                onChange={(values) => update((d) => {
                  d.rules[index].options = values;
                })}
              />
            </div>
          ))}

          <button
            className="add-button"
            disabled={
              config.rules.length >= catalog.rules.length
            }
            onClick={addRule}
          >
            <Plus size={16} />
            Add rule
          </button>
        </div>
      </details>

      <details className="config-section">
        <summary>
          <span><Target size={17} /> Rewards</span>
          <span className="section-meta">
            {config.rewards.length}
          </span>
        </summary>

        <div className="section-body">
          {config.rewards.map((reward, index) => (
            <div className="rule-editor" key={index}>
              <div className="editor-heading">
                <select
                  value={reward.kind}
                  onChange={(e) => update((d) => {
                    d.rewards[index].kind = e.target.value;
                    d.rewards[index].values = {};
                  })}
                >
                  {catalog.rewards.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>

                <button
                  className="icon-button danger"
                  disabled={config.rewards.length <= 1}
                  onClick={() => update((d) => {
                    d.rewards.splice(index, 1);
                  })}
                >
                  <Trash2 size={15} />
                </button>
              </div>

              <NumberField
                label="Peso"
                min={-10000}
                max={10000}
                step="any"
                value={reward.weight}
                onChange={(v) => update((d) => {
                  d.rewards[index].weight = v;
                })}
              />

              <div className="mini-heading">Values</div>

              <DictionaryEditor
                values={reward.values}
                numbersOnly
                onChange={(values) => update((d) => {
                  d.rewards[index].values =
                    values as Record<string, number>;
                })}
              />
            </div>
          ))}

          <button
            className="add-button"
            onClick={addReward}
          >
            <Plus size={16} />
            Add reward
          </button>
        </div>
      </details>

      <div className="panel-footer">
        The available algorithms, rules, and rewards
        are automatically queried from <code>/catalog</code>.
      </div>
    </div>
  );
}