import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  Activity,
  AlertCircle,
  Box,
  CheckCircle2,
  Crosshair,
  Database,
  Layers3,
  RefreshCw,
  X,
} from "lucide-react";

import type {
  StepSnapshot,
} from "./types/simulation";

import ExistingTrainingPanel, {
  type WorkspaceMode,
} from "./components/ExistingTrainingPanel";

import TrainingOverviewPanel from "./components/TrainingOverviewPanel";

import RewardsPanel from "./components/RewardsPanel";

import ConfigPanel from "./components/ConfigPanel";
import SimulationPanel from "./components/SimulationPanel";
import TrainingPanel from "./components/TrainingPanel";
import WorldScene from "./components/WorldScene";

import { api } from "./lib/api";

import {
  initialConfig,
  previewState,
  samePoint,
  validateConfig,
  teamColor,
} from "./lib/config";

import { deriveEffects } from "./lib/effects";

import {
  isTrainingSuccessful,
  useTraining,
} from "./hooks/useTraining";

import type {
  Catalog,
  EditTool,
  Point,
  SessionResponse,
  SimulationConfig,
  VisualEffect,
  TrainingStatus,
} from "./types/simulation";

type BusyAction =
  | "create"
  | "reset"
  | "delete"
  | null;

export default function App() {
  const [stepHistory, setStepHistory] =
    useState<StepSnapshot[]>([]);

  const [catalog, setCatalog] = useState<Catalog | null>(null);

  const [config, setConfig] =
    useState<SimulationConfig | null>(null);

  const [catalogLoading, setCatalogLoading] = useState(true);
  const [catalogError, setCatalogError] = useState("");

  const [session, setSession] =
    useState<SessionResponse | null>(null);

  const sessionRef = useRef<SessionResponse | null>(null);
  const stepInFlight = useRef(false);

  const [effects, setEffects] = useState<VisualEffect[]>([]);

  const [selectedCell, setSelectedCell] =
    useState<Point | null>(null);

  const [selectedAgentId, setSelectedAgentId] =
    useState("agent_a1");

  const [editTool, setEditTool] =
    useState<EditTool>("inspect");

  const [busyAction, setBusyAction] =
    useState<BusyAction>(null);

  const [stepping, setStepping] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(650);

  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const [useTrained, setUseTrained] = useState(false);

  const [trainingConfig, setTrainingConfig] =
    useState<SimulationConfig | null>(null);

  const [sessionWorldLimits, setSessionWorldLimits] =
    useState<{
      maxHealth: number;
      maxSteps: number | null;
    }>({
      maxHealth: 3,
      maxSteps: null,
    });

  const [workspaceMode, setWorkspaceMode] =
  useState<WorkspaceMode>("new");

  const [existingTrainingId, setExistingTrainingId] =
    useState("");

  const [existingTraining, setExistingTraining] =
    useState<TrainingStatus | null>(null);

  const [loadingExistingTraining, setLoadingExistingTraining] =
    useState(false);

  const existingTrainingReady = Boolean(
    existingTraining &&
    isTrainingSuccessful(existingTraining.status)
  );

  const training = useTraining();

  // ----------------------------------------------
  // Catalog
  // ----------------------------------------------

  const loadCatalog = useCallback(async () => {
    setCatalogLoading(true);
    setCatalogError("");

    try {
      const result = await api.catalog();

      setCatalog(result);

      setConfig((previous) =>
        previous ?? initialConfig(result)
      );
    } catch (e) {
      setCatalogError(
        e instanceof Error ? e.message : String(e)
      );
    } finally {
      setCatalogLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadCatalog();
  }, [loadCatalog]);

  // ----------------------------------------------
  // Validation and preview state
  // ----------------------------------------------

  const validationErrors = useMemo(() => {
    if (!config || !catalog) return [];

    return validateConfig(config, catalog);
  }, [config, catalog]);

  const preview = useMemo(
    () => config ? previewState(config) : null,
    [config]
  );

  const worldState =
    session?.state ??
    (
      workspaceMode === "existing"
        ? existingTraining?.config
          ? previewState(existingTraining.config)
          : null
        : preview
    );

  const trainingReady = Boolean(
    training.created &&
    isTrainingSuccessful(training.status?.status)
  );

  const activeConfigChanged = Boolean(
    trainingConfig &&
    config &&
    JSON.stringify(trainingConfig) !== JSON.stringify(config)
  );

  const working = busyAction !== null;

  // ----------------------------------------------
  // Edit cells
  // ----------------------------------------------

  const handleCellClick = (point: Point) => {
    setSelectedCell(point);

    if (
      !config ||
      session ||
      workspaceMode !== "new" ||
      editTool === "inspect"
    ) {
      return;
    }

    const draft = structuredClone(config);
    const world = draft.world;

    const occupied = draft.agents.some(
      (actor) => samePoint(actor.position, point)
    );

    const obstacleIndex = world.obstacles.findIndex(
      (p) => samePoint(p, point)
    );

    setNotice("");

    if (editTool === "obstacle") {
      if (obstacleIndex >= 0) {
        world.obstacles.splice(obstacleIndex, 1);
      } else {
        if (occupied || samePoint(world.goal, point)) {
          setNotice(
            "You cannot place an obstacle on an agent or the goal."
          );
          return;
        }

        world.obstacles.push(point);
      }
    }

    if (editTool === "goal") {
      if (samePoint(world.goal, point)) {
        world.goal = null;
      } else {
        if (obstacleIndex >= 0) {
          setNotice(
            "First remove the obstacle from that cell."
          );
          return;
        }

        world.goal = point;
      }
    }

    if (editTool === "agent") {
      const agent = draft.agents.find(
        (a) => a.id === selectedAgentId
      );

      if (!agent) {
        setNotice("Select an agent first.");
        return;
      }

      const occupiedByAnother = draft.agents.some(
        (a) =>
          a.id !== agent.id &&
          samePoint(a.position, point)
      );

      if (occupiedByAnother || obstacleIndex >= 0) {
        setNotice("That cell is already occupied.");
        return;
      }

      agent.position = point;
    }

    setConfig(draft);
  };

  // ----------------------------------------------
  // Training
  // ----------------------------------------------

  const handleStartTraining = async (episodes: number) => {
    if (!config || !catalog) return;

    if (validationErrors.length) {
      setError(validationErrors.join(" "));
      return;
    }

    setError("");
    setNotice("");
    setUseTrained(false);

    try {
      const snapshot = structuredClone(config);

      const result = await training.start(snapshot, episodes);

      setTrainingConfig(snapshot);

      setNotice(
        `Training started: ${result.training_id}`
      );
    } catch (e) {
      setError(
        e instanceof Error ? e.message : String(e)
      );
    }
  };

  const handleCopyTrainingConfig = () => {
    const source = existingTraining?.config;

    if (!source || sessionRef.current) {
      return;
    }

    const copied = structuredClone(source);

    setConfig(copied);

    setSelectedAgentId(
      copied.agents[0]?.id ?? ""
    );

    setWorkspaceMode("new");
    setUseTrained(false);
    setEditTool("inspect");

    setNotice(
      "The training configuration was copied to a new editable draft."
    );
  };

  const handleWorkspaceModeChange = (
    mode: WorkspaceMode
  ) => {
    if (sessionRef.current) {
      setError(
        "Delete the active session before changing configuration source."
      );
      return;
    }

    setPlaying(false);
    setEffects([]);
    setSelectedCell(null);

    setWorkspaceMode(mode);

    if (mode === "existing") {
      setEditTool("inspect");
    }

    setError("");
  };

  const handleLoadExistingTraining = async () => {
    const id = existingTrainingId.trim();

    if (!id) {
      setError("Enter a training ID.");
      return;
    }

    if (loadingExistingTraining) return;

    setLoadingExistingTraining(true);
    setExistingTraining(null);

    setError("");
    setNotice("");

    try {
      const result = await api.training(id);

      if (result.training_id !== id) {
        throw new Error(
          "The API returned a different training ID."
        );
      }

      setExistingTraining(result);

      if (isTrainingSuccessful(result.status)) {
        setNotice(
          "Training loaded successfully. You can create a trained session."
        );
      } else {
        setNotice(
          `Training found with status: ${result.status}.`
        );
      }
    } catch (error) {
      setError(
        error instanceof Error
          ? error.message
          : String(error)
      );
    } finally {
      setLoadingExistingTraining(false);
    }
  };

  // ----------------------------------------------
  // Create session
  // ----------------------------------------------

  const handleCreateSession = async () => {
    if (
      !config ||
      !catalog ||
      working ||
      stepInFlight.current
    ) {
      return;
    }

    const usingExistingTraining =
      workspaceMode === "existing";

    let trainingId: string | null = null;

    let sourceConfig: SimulationConfig | null = null;

    if (usingExistingTraining) {
      if (!existingTrainingReady || !existingTraining) {
        setError(
          "Load a completed training before creating a session."
        );
        return;
      }

      trainingId = existingTraining.training_id;

      sourceConfig = existingTraining.config ?? null;
    } else if (useTrained) {
      if (!trainingReady || !training.created) {
        setError(
          "The selected training is not ready."
        );
        return;
      }

      trainingId = training.created.training_id;

      sourceConfig = trainingConfig;
    } else {
      if (validationErrors.length > 0) {
        setError(validationErrors.join(" "));
        return;
      }

      sourceConfig = config;
    }

    setPlaying(false);
    setBusyAction("create");

    setError("");
    setNotice("");

    try {
      const previousId =
        sessionRef.current?.session_id;

      const response = trainingId
        ? await api.createSession(
            undefined,
            trainingId
          )
        : await api.createSession(
            config
          );

      sessionRef.current = response;
      setSession(response);

      setSelectedAgentId(
        response.state.actors[0]?.id ?? ""
      );

      setEffects([]);
      setSelectedCell(null);
      setEditTool("inspect");

      setStepHistory([]);

      setSessionWorldLimits({
        maxHealth:
          sourceConfig?.world.max_health ??
          Math.max(
            1,
            ...response.state.actors.map(
              (actor) => actor.health
            )
          ),

        maxSteps:
          sourceConfig?.world.max_steps ?? null,
      });

      if (trainingId) {
        setNotice(
          `Trained session created using ${trainingId}.`
        );
      } else {
        setNotice(
          "Untrained session created successfully."
        );
      }

      if (
        previousId &&
        previousId !== response.session_id
      ) {
        try {
          await api.deleteSession(previousId);
        } catch {
        }
      }
    } catch (error) {
      setError(
        error instanceof Error
          ? error.message
          : String(error)
      );
    } finally {
      setBusyAction(null);
    }
  };

  // ----------------------------------------------
  // Perform exactly one step
  // ----------------------------------------------

  const performStep = useCallback(async () => {
    const previous = sessionRef.current;

    if (
      !previous ||
      previous.state.finished ||
      stepInFlight.current
    ) {
      return;
    }

    stepInFlight.current = true;
    setStepping(true);

    try {
      const next = await api.step(previous.session_id);

      // Ignore responses from sessions that are no longer active.
      if (
        sessionRef.current?.session_id !== previous.session_id
      ) {
        return;
      }

      setEffects(
        deriveEffects(previous.state, next.state)
      );

      const snapshot: StepSnapshot = {
        step: next.state.step,

        rewards: { ...next.rewards },
        total_rewards: { ...next.total_rewards },

        actions: { ...next.actions },

        action_space: next.state.action_space.map(
          (entry) => ({ ...entry })
        ),

        terminations: { ...next.terminations },
        truncations: { ...next.truncations },
      };

      setStepHistory((history) => {
        const withoutCurrent = history.filter(
          (item) => item.step !== snapshot.step
        );

        return [...withoutCurrent, snapshot]
          .sort((a, b) => a.step - b.step);
      });

      sessionRef.current = next;
      setSession(next);

      if (next.state.finished) {
        setPlaying(false);
      }
    } catch (e) {
      setPlaying(false);

      setError(
        e instanceof Error ? e.message : String(e)
      );
    } finally {
      stepInFlight.current = false;
      setStepping(false);
    }
  }, []);

  // ----------------------------------------------
  // Automatic playback
  // ----------------------------------------------

  useEffect(() => {
    if (!playing || !session || session.state.finished) {
      return;
    }

    const timer = window.setInterval(() => {
      void performStep();
    }, speed);

    return () => clearInterval(timer);
  }, [
    playing,
    session?.session_id,
    session?.state.finished,
    speed,
    performStep,
  ]);

  // ----------------------------------------------
  // Reset session
  // ----------------------------------------------

  const handleReset = async () => {
    if (!session || working || stepInFlight.current) return;

    setPlaying(false);
    setBusyAction("reset");
    setError("");

    try {
      const next = await api.reset(session.session_id);

      sessionRef.current = next;
      setSession(next);
      setStepHistory([]);
      setEffects([]);
      setSelectedCell(null);

      setNotice("Episode reset.");
    } catch (e) {
      setError(
        e instanceof Error ? e.message : String(e)
      );
    } finally {
      setBusyAction(null);
    }
  };

  // ----------------------------------------------
  // Delete session
  // ----------------------------------------------

  const handleDelete = async () => {
    if (!session || working || stepInFlight.current) return;

    setPlaying(false);
    setBusyAction("delete");
    setError("");

    try {
      await api.deleteSession(session.session_id);

      sessionRef.current = null;

      setSession(null);
      setStepHistory([]);
      setEffects([]);
      setSelectedCell(null);

      setNotice("Session deleted.");
    } catch (e) {
      setError(
        e instanceof Error ? e.message : String(e)
      );
    } finally {
      setBusyAction(null);
    }
  };

  // ----------------------------------------------
  // Selected cell information
  // ----------------------------------------------

  const selectedActor = worldState?.actors.find(
    (a) => samePoint(a.position, selectedCell)
  );

  const selectedIsObstacle = Boolean(
    worldState?.obstacles.some(
      (p) => samePoint(p, selectedCell)
    )
  );

  const selectedIsGoal = samePoint(
    worldState?.goal,
    selectedCell
  );

  // Hide notifications after 5 seconds.
  useEffect(() => {
    if (!notice) return;

    const timer = window.setTimeout(() => {
      setNotice("");
    }, 5000);

    return () => window.clearTimeout(timer);
  }, [notice]);

  // Hide errors after 8 seconds.
  useEffect(() => {
    if (!error) return;

    const timer = window.setTimeout(() => {
      setError("");
    }, 8000);

    return () => window.clearTimeout(timer);
  }, [error]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <Layers3 size={21} />
          </div>

          <div>
            <strong>AgentLab</strong>
            <span>MULTI-AGENT SIMULATION STUDIO</span>
          </div>
        </div>

        <div className="topbar-right">
          <span className="header-tag">
            <Box size={14} />
            3D WORLD
          </span>

          <span className="header-tag">
            <Activity size={14} />
            RL ENGINE
          </span>

          <span className={
            catalog ? "api-badge connected" : "api-badge"
          }>
            <span className="status-dot" />

            {catalog ? "API connected" : "API disconnected"}
          </span>

          <button
            className="header-refresh"
            title="Reload catalog"
            onClick={() => void loadCatalog()}
          >
            <RefreshCw size={17} />
          </button>
        </div>
      </header>

      {error && (
        <div className="global-message global-error">
          <AlertCircle size={17} />
          <span>{error}</span>

          <button onClick={() => setError("")}>
            <X size={17} />
          </button>
        </div>
      )}

      {notice && (
        <div className="global-message global-notice">
          <CheckCircle2 size={17} />
          <span>{notice}</span>

          <button onClick={() => setNotice("")}>
            <X size={17} />
          </button>
        </div>
      )}

      {catalogLoading && !catalog && (
        <div className="loading-screen">
          <RefreshCw size={30} className="spinning" />
          <h2>Connecting to the simulator</h2>
          <p>Querying algorithms, rules, and rewards...</p>
        </div>
      )}

      {catalogError && !catalog && !catalogLoading && (
        <div className="loading-screen">
          <AlertCircle size={30} />
          <h2>Could not connect to FastAPI</h2>
          <p>{catalogError}</p>

          <button
            className="primary-button"
            onClick={() => void loadCatalog()}
          >
            Retry
          </button>
        </div>
      )}

      {catalog && config && (
        <main className="workspace">
          <aside className="workspace-left">
            {workspaceMode === "new" ? (
              <ConfigPanel
                config={config}
                catalog={catalog}
                onChange={setConfig}

                selectedAgentId={selectedAgentId}
                onSelectAgent={setSelectedAgentId}

                editTool={editTool}
                onEditTool={setEditTool}

                sceneLocked={Boolean(session)}
              />
            ) : (
              <TrainingOverviewPanel
                training={existingTraining}
                state={session?.state ?? null}
              />
            )}
          </aside>

          <section className="workspace-center">
            {worldState ? (
              <>
                <div className="viewport-header">
                  <div>
                    <div className="eyebrow">
                      LIVE ENVIRONMENT
                    </div>

                    <h2>Simulation Field</h2>
                  </div>

                  <div className="viewport-chips">
                    <span>
                      <span className="small-dot teal" />
                      {worldState.actors.length} agents
                    </span>

                    <span>
                      {worldState.width} x {worldState.height}
                    </span>

                    <span>
                      Step {worldState.step}
                    </span>
                  </div>
                </div>

                <div className="viewport">
                  <WorldScene
                    state={worldState}
                    maxHealth={
                      session
                        ? sessionWorldLimits.maxHealth
                        : config.world.max_health
                    }
                    selectedCell={selectedCell}
                    onCellClick={handleCellClick}
                    effects={effects}

                    selectedAgentId={selectedAgentId}
                    onSelectAgent={(id) => {
                      setSelectedAgentId(id);

                      if (!session) {
                        setEditTool("agent");
                      }
                    }}
                  />

                  <div className="viewport-overlay top-left">
                    <span className="view-label">
                      ISOMETRIC VIEW
                    </span>

                    <span className="view-subtitle">
                      Drag to rotate · Scroll to zoom
                    </span>
                  </div>

                  <div className="viewport-overlay bottom-left">
                    <Crosshair size={15} />

                    <span>
                      {selectedCell
                        ? `Cell [${selectedCell.join(", ")}]`
                        : "Select a cell"}
                    </span>
                  </div>

                  <div className="viewport-overlay bottom-right">
                    <span className="render-indicator" />
                    WEBGL ACTIVE
                  </div>
                </div>

                <div className="viewport-footer">
                  <div className="cell-info">
                    <div className="eyebrow">
                      INSPECTOR
                    </div>

                    {selectedCell ? (
                      <div className="selected-details">
                        <strong>
                          [{selectedCell[0]}, {selectedCell[1]}]
                        </strong>

                        <span>
                          {selectedActor
                            ? `${selectedActor.id} · ${selectedActor.team} · HP ${selectedActor.health}`
                            : selectedIsObstacle
                              ? "Obstacle"
                              : selectedIsGoal
                                ? "Goal"
                                : "Free cell"}
                        </span>
                      </div>
                    ) : (
                      <p className="muted">
                        Select a cell to inspect it.
                      </p>
                    )}
                  </div>

                  <div className="legend">
                    {[
                      ...new Set(
                        worldState.actors.map((a) => a.team)
                      ),
                    ].map((team) => (
                      <span key={team}>
                        <i
                          style={{
                            background:
                              (
                                teamColor(team)
                              ),
                          }}
                        />
                        {team}
                      </span>
                    ))}

                    <span>
                      <i className="legend-goal" />
                      Goal
                    </span>
                  </div>
                </div>

                <RewardsPanel
                  history={stepHistory}
                  actors={worldState.actors}
                  actionsCatalog={catalog.actions}
                  currentStep={worldState.step}
                  sessionActive={Boolean(session)}
                />

                {workspaceMode === "new" &&
                  validationErrors.length > 0 && (
                  <div className="validation-panel">
                    <AlertCircle size={17} />

                    <div>
                      <strong>
                        Configuration pending correction
                      </strong>

                      {validationErrors.slice(0, 4).map((msg, i) => (
                        <p key={i}>{msg}</p>
                      ))}

                      {validationErrors.length > 4 && (
                        <p>
                          And {validationErrors.length - 4} more errors.
                        </p>
                      )}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="training-world-placeholder">
                <Database size={38} />

                <h2>World preview unavailable</h2>

                <p>
                  The training API does not expose the original
                  simulation configuration.
                </p>

                <p>
                  Load a completed training and create a session
                  to retrieve its world, agents, and action space.
                </p>
              </div>
            )}
          </section>

          <aside className="workspace-right">
            <ExistingTrainingPanel
              mode={workspaceMode}
              onModeChange={handleWorkspaceModeChange}

              trainingId={existingTrainingId}

              onTrainingIdChange={(value) => {
                setExistingTrainingId(value);
                setExistingTraining(null);
              }}

              training={existingTraining}
              loading={loadingExistingTraining}

              sessionActive={Boolean(session)}

              onLoad={() => {
                void handleLoadExistingTraining();
              }}

              onCopyConfig={handleCopyTrainingConfig}
            />

            {workspaceMode === "new" && (
              <TrainingPanel
                status={training.status}
                episodes={training.episodes}

                optimizations={training.optimizations}
                optimizationStarted={training.optimizationStarted}

                starting={training.starting}
                streamConnected={training.streamConnected}

                onStart={(count) => {
                  void handleStartTraining(count);
                }}
              />
            )}

            <SimulationPanel
              session={session}

              maxSteps={
                session
                  ? sessionWorldLimits.maxSteps
                  : workspaceMode === "existing"
                    ? existingTraining?.config?.world.max_steps ?? null
                    : config.world.max_steps
              }

              busy={working || stepping}
              stepping={stepping}
              playing={playing}

              speed={speed}
              onSpeed={setSpeed}

              useTrained={
                workspaceMode === "existing"
                  ? true
                  : useTrained
              }

              canUseTrained={
                workspaceMode === "existing"
                  ? existingTrainingReady
                  : trainingReady
              }

              fixedTrainingId={
                workspaceMode === "existing"
                  ? existingTraining?.training_id ?? null
                  : undefined
              }

              canCreateSession={
                workspaceMode === "existing"
                  ? existingTrainingReady
                  : true
              }

              onUseTrained={setUseTrained}

              onCreate={() => {
                void handleCreateSession();
              }}

              onStep={() => {
                void performStep();
              }}

              onTogglePlay={() => {
                setPlaying((value) => !value);
              }}

              onReset={() => {
                void handleReset();
              }}

              onDelete={() => {
                void handleDelete();
              }}
            />

            {workspaceMode === "new" && trainingReady && (
              <div className="notice">
                <strong>Trained model available</strong>

                <p>
                  You can create a session using the completed
                  training from this application.
                </p>

                {activeConfigChanged && (
                  <p>
                    The draft configuration has changed since
                    training started. The trained session will
                    use the original training configuration.
                  </p>
                )}
              </div>
            )}
          </aside>
        </main>
      )}
    </div>
  );
}