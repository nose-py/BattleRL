import type {
  AgentSpec,
  Catalog,
  Point,
  SimulationConfig,
  WorldState,
} from "../types/simulation";

const TEAM_PALETTE = [
  "#55e3c1",
  "#ffad70",
  "#80aaff",
  "#f187cb",
  "#d7bd6e",
  "#a58aff",
];

export function teamColor(team: string): string {
  let hash = 0;

  for (let i = 0; i < team.length; i++) {
    hash = (hash * 31 + team.charCodeAt(i)) | 0;
  }

  return TEAM_PALETTE[
    (hash >>> 0) % TEAM_PALETTE.length
  ];
}

export function samePoint(
  a: Point | null | undefined,
  b: Point | null | undefined
): boolean {
  return Boolean(
    a && b && a[0] === b[0] && a[1] === b[1]
  );
}

export function inside(
  point: Point,
  width: number,
  height: number
): boolean {
  return (
    Number.isInteger(point[0]) &&
    Number.isInteger(point[1]) &&
    point[0] >= 0 &&
    point[0] < width &&
    point[1] >= 0 &&
    point[1] < height
  );
}

export function newAgent(
  id: string,
  team: string,
  policy: string,
  position: Point
): AgentSpec {
  return {
    id,
    team,
    position,

    policy,

    alpha: 0.15,
    gamma: 0.95,

    epsilon: 1,
    epsilon_min: 0.05,
    epsilon_decay: 0.995,
  };
}

function matching(
  values: string[],
  regex: RegExp
): string | undefined {
  return values.find((x) => regex.test(x));
}

export function initialConfig(
  catalog: Catalog
): SimulationConfig {
  const move = matching(
    catalog.rules,
    /mov|walk|step|desplaz/i
  );

  const shoot = matching(
    catalog.rules,
    /shoot|shot|fire|attack|dispar/i
  );

  const selectedRules = [...new Set(
    [move, shoot].filter((x): x is string => Boolean(x))
  )];

  if (selectedRules.length === 0 && catalog.rules.length) {
    selectedRules.push(catalog.rules[0]);
  }

  const goalReward = matching(
    catalog.rewards,
    /goal|target|position|distance|meta/i
  );

  const reward = goalReward ?? catalog.rewards[0];

  return {
    world: {
      width: 7,
      height: 7,

      max_steps: 100,
      max_health: 3,
      vision_radius: 1,

      goal: [3, 3],
      obstacles: [],

      seed: 42,
    },

    agents: [
      newAgent(
        "agent_a1",
        "Alpha",
        catalog.policies[0] ?? "",
        [1, 1]
      ),
      newAgent(
        "agent_b1",
        "Beta",
        catalog.policies[0] ?? "",
        [5, 5]
      ),
    ],

    rules: selectedRules.map((name) => ({
      name,
      options: {},
    })),

    rewards: reward
      ? [{
          kind: reward,
          weight: 1,
          values: {},
        }]
      : [],
  };
}

export function previewState(
  config: SimulationConfig
): WorldState {
  return {
    step: 0,

    width: config.world.width,
    height: config.world.height,

    goal: config.world.goal,
    obstacles: config.world.obstacles,

    actors: config.agents.map((agent) => ({
      id: agent.id,
      team: agent.team,
      position: agent.position,

      health: config.world.max_health,
      alive: true,

      kills: 0,
      goals: 0,
    })),

    events: [],

    finished: false,
    winner: null,
    finish_reason: null,
  };
}

export function validateConfig(
  config: SimulationConfig,
  catalog: Catalog
): string[] {
  const errors: string[] = [];
  const w = config.world;

  if (
    !Number.isInteger(w.width) ||
    !Number.isInteger(w.height) ||
    w.width < 3 ||
    w.height < 3
  ) {
    errors.push("The field must be at least 3 × 3.");
  }

  if (
    !Number.isInteger(w.max_steps) ||
    w.max_steps < 1 ||
    w.max_steps > 10000
  ) {
    errors.push("max_steps must be between 1 and 10000.");
  }

  if (!Number.isInteger(w.max_health) || w.max_health < 1) {
    errors.push("max_health must be a positive integer.");
  }

  if (
    !Number.isInteger(w.vision_radius) ||
    w.vision_radius < 1
  ) {
    errors.push("vision_radius must be a positive integer.");
  }

  if (!Number.isInteger(w.seed)) {
    errors.push("The seed must be an integer.");
  }

  if (config.agents.length < 2) {
    errors.push("You need at least two agents.");
  }

  if (config.rules.length < 1) {
    errors.push("Select at least one rule.");
  }

  if (config.rewards.length < 1) {
    errors.push("Select at least one reward.");
  }

  if (w.goal && !inside(w.goal, w.width, w.height)) {
    errors.push("The goal is outside the field.");
  }

  const obstacleKeys = w.obstacles.map((p) => p.join(","));

  if (new Set(obstacleKeys).size !== obstacleKeys.length) {
    errors.push("There are duplicate obstacles.");
  }

  for (const p of w.obstacles) {
    if (!inside(p, w.width, w.height)) {
      errors.push(`Obstacle outside the field: ${p}.`);
    }

    if (samePoint(p, w.goal)) {
      errors.push("An obstacle cannot occupy the goal.");
    }
  }

  const ids = new Set<string>();
  const positions = new Set<string>();

  for (const a of config.agents) {
    if (!/^[A-Za-z0-9_-]{1,40}$/.test(a.id)) {
      errors.push(`Invalid ID: ${a.id}`);
    }

    if (!/^[A-Za-z0-9_-]{1,40}$/.test(a.team)) {
      errors.push(`Invalid team: ${a.team}`);
    }

    if (ids.has(a.id)) {
      errors.push(`Duplicate agent ID: ${a.id}`);
    }

    ids.add(a.id);

    if (!catalog.policies.includes(a.policy)) {
      errors.push(`Invalid policy for ${a.id}.`);
    }

    if (!inside(a.position, w.width, w.height)) {
      errors.push(`${a.id} is outside the field.`);
    }

    const key = a.position.join(",");

    if (positions.has(key)) {
      errors.push(`There are two agents at ${key}.`);
    }

    positions.add(key);

    if (obstacleKeys.includes(key)) {
      errors.push(`${a.id} occupies an obstacle.`);
    }

    if (
      !(a.alpha > 0 && a.alpha <= 1) ||
      !(a.gamma >= 0 && a.gamma <= 1) ||
      !(a.epsilon >= 0 && a.epsilon <= 1) ||
      !(a.epsilon_min >= 0 && a.epsilon_min <= 1) ||
      !(a.epsilon_decay > 0 && a.epsilon_decay <= 1)
    ) {
      errors.push(`Invalid hyperparameters for ${a.id}.`);
    }
  }

  for (const rule of config.rules) {
    if (!catalog.rules.includes(rule.name)) {
      errors.push(`Unknown rule: ${rule.name}.`);
    }
  }

  for (const reward of config.rewards) {
    if (!catalog.rewards.includes(reward.kind)) {
      errors.push(`Unknown reward: ${reward.kind}.`);
    }

    if (
      !Number.isFinite(reward.weight) ||
      reward.weight < -10000 ||
      reward.weight > 10000
    ) {
      errors.push(`Invalid weight for ${reward.kind}.`);
    }

    if (
      Object.values(reward.values).some(
        (v) => typeof v !== "number" || !Number.isFinite(v)
      )
    ) {
      errors.push(`Invalid values for ${reward.kind}.`);
    }
  }

  return errors;
}