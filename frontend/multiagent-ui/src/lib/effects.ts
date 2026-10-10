import type {
  ActorState,
  Point,
  SimulationEvent,
  VisualEffect,
  WorldState,
} from "../types/simulation";

/* --------------------------------------------------
   Utils
-------------------------------------------------- */

function getActor(
  id: string | null,
  previous: WorldState,
  current: WorldState
): ActorState | undefined {
  if (!id) return undefined;

  return (
    current.actors.find((a) => a.id === id) ??
    previous.actors.find((a) => a.id === id)
  );
}

function samePosition(a: Point, b: Point): boolean {
  return a[0] === b[0] && a[1] === b[1];
}

/**
 * Supports two forms of backend implementation:
 *
 * 1. state.events contains only the events
 *    corresponding to the last step.
 *
 * 2. state.events contains all accumulated events
 *    since the beginning of the session.
 */
function eventsForStep(
  previous: WorldState,
  current: WorldState
): SimulationEvent[] {
  const oldEvents = previous.events ?? [];
  const newEvents = current.events ?? [];

  const accumulated =
    oldEvents.length > 0 &&
    newEvents.length >= oldEvents.length &&
    oldEvents.every(
      (event, index) =>
        JSON.stringify(event) ===
        JSON.stringify(newEvents[index])
    );

  if (accumulated) {
    return newEvents.slice(oldEvents.length);
  }

  return newEvents;
}

/* --------------------------------------------------
   Event Adapters
-------------------------------------------------- */

type EventAdapter = (
  event: SimulationEvent,
  previous: WorldState,
  current: WorldState,
  id: string
) => VisualEffect[];

/**
 * Shot with impact.
 *
 * event.actor  -> attacker
 * event.target -> victim
 *
 * Works for both "hit" and "kill".
 */
function attackAdapter(
  impactKind: "hit" | "kill"
): EventAdapter {
  return (event, previous, current, id) => {
    const attacker = getActor(
      event.actor,
      previous,
      current
    );

    const victim = getActor(
      event.target,
      previous,
      current
    );

    if (!attacker || !victim) return [];

    return [
      {
        id: `${id}-projectile`,
        kind: "shot",
        from: attacker.position,
        to: victim.position,
        team: attacker.team,
      },
      {
        id: `${id}-impact`,
        kind: impactKind,
        to: victim.position,
        team: victim.team,
      },
    ];
  };
}

/**
 * Local effects around the agent.
 *
 * Used for miss, invalid_move, goal, and first_goal.
 */
function actorPulseAdapter(
  kind: VisualEffect["kind"]
): EventAdapter {
  return (event, previous, current, id) => {
    const actor = getActor(
      event.actor,
      previous,
      current
    );

    if (!actor) return [];

    return [{
      id,
      kind,
      to: actor.position,
      team: actor.team,
    }];
  };
}

/**
 * Extensible registry of actions.
 *
 * To add a new visual rule, simply
 * register the new event type here.
 */
export const eventAdapters: Record<string, EventAdapter> = {
  hit: attackAdapter("hit"),
  kill: attackAdapter("kill"),

  miss: actorPulseAdapter("miss"),

  invalid_move: actorPulseAdapter("invalid_move"),

  goal: actorPulseAdapter("goal"),
  first_goal: actorPulseAdapter("first_goal"),

  // "move" is resolved by comparing positions.
};

/* --------------------------------------------------
   Movement
-------------------------------------------------- */

/**
 * The previous and new positions come from
 * the simulation states.
 *
 * This is more reliable than trying to obtain
 * the coordinates from a "move" event, as the
 * event structure does not contain positions.
 */
function movementEffects(
  previous: WorldState,
  current: WorldState
): VisualEffect[] {
  const effects: VisualEffect[] = [];

  for (const actor of current.actors) {
    const before = previous.actors.find(
      (a) => a.id === actor.id
    );

    if (!before) continue;

    if (!samePosition(before.position, actor.position)) {
      effects.push({
        id: `move-${current.step}-${actor.id}`,
        kind: "move",
        from: before.position,
        to: actor.position,
        team: actor.team,
      });
    }
  }

  return effects;
}

/* --------------------------------------------------
   Health changes without explicit events
-------------------------------------------------- */

function healthEffects(
  previous: WorldState,
  current: WorldState,
  events: SimulationEvent[]
): VisualEffect[] {
  const effects: VisualEffect[] = [];

  const impactedTargets = new Set(
    events
      .filter(
        (e) => e.type === "hit" || e.type === "kill"
      )
      .map((e) => e.target)
      .filter((id): id is string => id !== null)
  );

  for (const actor of current.actors) {
    const before = previous.actors.find(
      (a) => a.id === actor.id
    );

    if (!before) continue;

    // Healing, even if there is no heal event yet.
    if (actor.health > before.health) {
      effects.push({
        id: `heal-${current.step}-${actor.id}`,
        kind: "heal",
        to: actor.position,
        team: actor.team,
      });
    }

    // Damage without explicit event.
    if (
      actor.health < before.health &&
      !impactedTargets.has(actor.id)
    ) {
      effects.push({
        id: `damage-${current.step}-${actor.id}`,
        kind: "hit",
        to: actor.position,
        team: actor.team,
      });
    }
  }

  return effects;
}

/* --------------------------------------------------
   Filtering redundant events
-------------------------------------------------- */

/**
 * Some backends may generate:
 *
 * hit  -> A shoots B
 * kill -> A kills B
 *
 * in the same step.
 *
 * In that case, we show the kill effect,
 * avoiding two overlapping projectiles.
 *
 * Likewise, first_goal has priority
 * over goal for the same agent and step.
 */
function filterRedundantEvents(
  events: SimulationEvent[]
): SimulationEvent[] {
  return events.filter((event) => {
    if (event.type === "hit") {
      const hasKill = events.some(
        (other) =>
          other.type === "kill" &&
          other.actor === event.actor &&
          other.target === event.target &&
          other.step === event.step
      );

      if (hasKill) return false;
    }

    if (event.type === "goal") {
      const hasFirstGoal = events.some(
        (other) =>
          other.type === "first_goal" &&
          other.actor === event.actor &&
          other.step === event.step
      );

      if (hasFirstGoal) return false;
    }

    return true;
  });
}

/* --------------------------------------------------
   Main function
-------------------------------------------------- */

export function deriveEffects(
  previous: WorldState,
  current: WorldState
): VisualEffect[] {
  const effects: VisualEffect[] = [];

  const events = filterRedundantEvents(
    eventsForStep(previous, current)
  );

  // 1. Detect movements based on positions.
  effects.push(
    ...movementEffects(previous, current)
  );

  // 2. Interpret API events.
  for (const [index, event] of events.entries()) {
    const adapter = eventAdapters[event.type];

    if (!adapter) {
      // Eventos desconocidos no rompen la visualización.
      continue;
    }

    effects.push(
      ...adapter(
        event,
        previous,
        current,
        `${event.step}-${index}-${event.type}`
      )
    );
  }

  // 3. Health changes without their own event.
  effects.push(
    ...healthEffects(previous, current, events)
  );

  return effects;
}