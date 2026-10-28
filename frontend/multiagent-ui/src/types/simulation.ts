export type Point = [number, number];

export type ParameterType =
  | "integer"
  | "float"
  | "string"
  | "boolean";

export type ParameterValue =
  | number
  | string
  | boolean;

export type ParameterDefinition =
  | ["integer", number]
  | ["float", number]
  | ["string", string]
  | ["boolean", boolean];

export type ParameterSchema = Record<
  string,
  ParameterDefinition
>;

export interface AgentSpec {
  id: string;
  team: string;
  position: Point;

  policy: string;

  params: Record<string, ParameterValue>;
}

export interface RuleSpec {
  name: string;
  options: Record<string, ParameterValue>;
}

export interface RewardSpec {
  kind: string;
  weight: number;
  values: Record<string, number>;
}

export interface WorldConfig {
  width: number;
  height: number;

  max_steps: number;
  max_health: number;
  vision_radius: number;

  goal: Point | null;
  obstacles: Point[];

  seed: number;
}

export interface SimulationConfig {
  world: WorldConfig;
  agents: AgentSpec[];
  rules: RuleSpec[];
  rewards: RewardSpec[];
}

export type SimulationEventType =
  | "goal"
  | "first_goal"
  | "invalid_move"
  | "move"
  | "miss"
  | "hit"
  | "kill"
  | (string & {});

export interface SimulationEvent {
  type: SimulationEventType;
  actor: string;
  target: string | null;
  step: number;
}

export type OptimizationStatus =
  | "pending"
  | "running"
  | "completed";

export interface AgentOptimization {
  agent_id: string;

  status: OptimizationStatus;

  evaluations: number;
  total_evaluations: number | null;
}

export type OptimizationEvent =
  | {
      type: "optimization_started";
      agent_id: string;
    }
  | {
      type: "optimization_progress";
      agent_id: string;
      evaluations: number;
      total_evaluations: number;
    }
  | {
      type: "optimization_completed";
      agent_id: string;
    };


export interface Catalog {
  policies: string[];

  policy_params: Record<
    string,
    ParameterSchema
  >;

  rules: string[];

  rule_params: Record<
    string,
    ParameterSchema
  >;

  rewards: string[];

  actions: Record<string, string>;
}

export interface ActorState {
  id: string;
  team: string;
  position: Point;

  health: number;
  alive: boolean;

  kills: number;
  goals: number;
}

export interface ActionSpaceEntry {
  index: number;
  code: number;
  name: string;
}

export interface WorldState {
  step: number;

  width: number;
  height: number;

  goal: Point | null;
  obstacles: Point[];

  actors: ActorState[];

  action_space: ActionSpaceEntry[];

  events: SimulationEvent[];

  finished: boolean;
  winner: string | null;
  finish_reason: string | null;
}

export interface SessionResponse {
  session_id: string;
  state: WorldState;
}

export interface TrainingCreated {
  training_id: string;
  status: string;
  status_url: string;
  events_url: string;
}

export interface TrainingStatus {
  training_id: string;
  status: string;

  episodes: number;
  completed_episodes: number;

  last_episode?: number;
  error?: string | null;
}

export interface EpisodeEvent {
  episode: number;
  steps: number;

  winner: string | null;
  finish_reason: string | null;

  agent_rewards: Record<string, number>;
  team_rewards: Record<string, number>;

  kills: Record<string, number>;
  goals: Record<string, number>;
}

export interface StepMetrics {
  total_rewards: Record<string, number>;
  actions: Record<string, number>;
  rewards: Record<string, number>;
  terminations: Record<string, boolean>;
  truncations: Record<string, boolean>;
}

export interface StepResponse
  extends SessionResponse, StepMetrics {}

export interface StepSnapshot extends StepMetrics {
  step: number;
  action_space: ActionSpaceEntry[];
}

export type EditTool =
  | "inspect"
  | "obstacle"
  | "goal"
  | "agent";

export type EffectKind =
  | "move"
  | "shot"
  | "hit"
  | "kill"
  | "miss"
  | "invalid_move"
  | "goal"
  | "first_goal"
  | "heal";

export interface VisualEffect {
  id: string;
  kind: EffectKind;

  from?: Point;
  to: Point;

  team?: string;
}