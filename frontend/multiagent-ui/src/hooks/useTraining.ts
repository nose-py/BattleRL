import {
  useCallback,
  useEffect,
  useState,
} from "react";

import {
  api,
  trainingEventsUrl,
} from "../lib/api";

import type {
  AgentOptimization,
  EpisodeEvent,
  SimulationConfig,
  TrainingCreated,
  TrainingStatus,
} from "../types/simulation";

/* --------------------------------------------------
   Training status helpers
-------------------------------------------------- */

export function isTrainingFinished(
  status?: string
): boolean {
  return [
    "completed",
    "finished",
    "succeeded",
    "success",
    "failed",
    "error",
    "cancelled",
    "canceled",
  ].includes((status ?? "").toLowerCase());
}

export function isTrainingSuccessful(
  status?: string
): boolean {
  return [
    "completed",
    "finished",
    "succeeded",
    "success",
  ].includes((status ?? "").toLowerCase());
}

/* --------------------------------------------------
   SSE parsing
-------------------------------------------------- */

function isObject(
  value: unknown
): value is Record<string, unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value)
  );
}

function parsePayload(
  raw: string
): Record<string, unknown> | null {
  try {
    const parsed: unknown = JSON.parse(raw);

    if (!isObject(parsed)) {
      return null;
    }

    // Support both raw SSE data and wrapped event objects.
    if (isObject(parsed.data)) {
      return parsed.data;
    }

    if (typeof parsed.data === "string") {
      try {
        const nested: unknown = JSON.parse(parsed.data);

        if (isObject(nested)) {
          return nested;
        }
      } catch {
        // Continue with the original payload.
      }
    }

    return parsed;
  } catch {
    return null;
  }
}

function parseEpisode(
  payload: Record<string, unknown>
): EpisodeEvent | null {
  if (typeof payload.episode !== "number") {
    return null;
  }

  return {
    episode: payload.episode,

    steps:
      typeof payload.steps === "number"
        ? payload.steps
        : 0,

    winner:
      typeof payload.winner === "string"
        ? payload.winner
        : null,

    finish_reason:
      typeof payload.finish_reason === "string"
        ? payload.finish_reason
        : null,

    agent_rewards: isObject(payload.agent_rewards)
      ? payload.agent_rewards as Record<string, number>
      : {},

    team_rewards: isObject(payload.team_rewards)
      ? payload.team_rewards as Record<string, number>
      : {},

    kills: isObject(payload.kills)
      ? payload.kills as Record<string, number>
      : {},

    goals: isObject(payload.goals)
      ? payload.goals as Record<string, number>
      : {},
  };
}

function createPendingOptimizations(
  config: SimulationConfig
): Record<string, AgentOptimization> {
  return Object.fromEntries(
    config.agents.map((agent) => [
      agent.id,
      {
        agent_id: agent.id,
        status: "pending" as const,
        evaluations: 0,
        total_evaluations: null,
      },
    ])
  );
}

/* --------------------------------------------------
   Training hook
-------------------------------------------------- */

export function useTraining() {
  const [created, setCreated] =
    useState<TrainingCreated | null>(null);

  const [status, setStatus] =
    useState<TrainingStatus | null>(null);

  const [episodes, setEpisodes] =
    useState<EpisodeEvent[]>([]);

  const [optimizations, setOptimizations] =
    useState<Record<string, AgentOptimization>>({});

  const [optimizationStarted, setOptimizationStarted] =
    useState(false);

  const [starting, setStarting] = useState(false);

  const [streamConnected, setStreamConnected] =
    useState(false);

  /* --------------------------------------------------
     Start training
  -------------------------------------------------- */

  const start = useCallback(async (
    config: SimulationConfig,
    count: number
  ) => {
    setStarting(true);

    setCreated(null);
    setStatus(null);

    setEpisodes([]);

    setOptimizationStarted(false);

    setOptimizations(
      createPendingOptimizations(config)
    );

    setStreamConnected(false);

    try {
      const result = await api.startTraining(
        config,
        count
      );

      setCreated(result);

      setStatus({
        training_id: result.training_id,
        status: result.status,

        episodes: count,
        completed_episodes: 0,
      });

      return result;
    } finally {
      setStarting(false);
    }
  }, []);

  /* --------------------------------------------------
     Subscribe to training events
  -------------------------------------------------- */

  useEffect(() => {
    if (!created) return;

    let active = true;

    const source = new EventSource(
      trainingEventsUrl(
        created.training_id,
        created.events_url
      )
    );

    const handleOptimization = (
      type: string,
      payload: Record<string, unknown>
    ) => {
      const agentId = payload.agent_id;

      if (typeof agentId !== "string" || !agentId) {
        return;
      }

      setOptimizationStarted(true);

      setOptimizations((previous) => {
        const current: AgentOptimization =
          previous[agentId] ?? {
            agent_id: agentId,
            status: "pending",
            evaluations: 0,
            total_evaluations: null,
          };

        if (type === "optimization_started") {
          return {
            ...previous,

            [agentId]: {
              ...current,

              status:
                current.status === "completed"
                  ? "completed"
                  : "running",
            },
          };
        }

        if (type === "optimization_progress") {
          const evaluations = payload.evaluations;
          const total = payload.total_evaluations;

          if (
            typeof evaluations !== "number" ||
            !Number.isFinite(evaluations) ||
            typeof total !== "number" ||
            !Number.isFinite(total)
          ) {
            return previous;
          }

          return {
            ...previous,

            [agentId]: {
              ...current,

              status:
                current.status === "completed"
                  ? "completed"
                  : "running",

              evaluations: Math.max(
                current.evaluations,
                evaluations
              ),

              total_evaluations: Math.max(
                0,
                total
              ),
            },
          };
        }

        if (type === "optimization_completed") {
          return {
            ...previous,

            [agentId]: {
              ...current,
              status: "completed",
            },
          };
        }

        return previous;
      });
    };

    const handleEvent = (event: MessageEvent) => {
      if (!active) return;

      const payload = parsePayload(event.data);

      if (!payload) return;

      let type = event.type;

      // Unnamed SSE messages can include an event name
      // inside the JSON payload.
      if (type === "message") {
        const parsedRaw: unknown = (() => {
          try {
            return JSON.parse(event.data);
          } catch {
            return null;
          }
        })();

        if (
          isObject(parsedRaw) &&
          typeof parsedRaw.event === "string"
        ) {
          type = parsedRaw.event;
        } else if (typeof payload.type === "string") {
          type = payload.type;
        }
      }

      if (
        type === "optimization_started" ||
        type === "optimization_progress" ||
        type === "optimization_completed"
      ) {
        handleOptimization(type, payload);
        return;
      }

      // Handle episode summaries from both named events
      // and ordinary SSE messages.
      const episode = parseEpisode(payload);

      if (!episode) return;

      setEpisodes((previous) => {
        const next = new Map<number, EpisodeEvent>(
          previous.map((e) => [
            e.episode,
            e,
          ] as const)
        );

        next.set(episode.episode, episode);

        return [...next.values()].sort(
          (a, b) => a.episode - b.episode
        );
      });
    };

    /* --------------------------------------------------
       SSE listeners
    -------------------------------------------------- */

    source.onopen = () => {
      if (active) {
        setStreamConnected(true);
      }
    };

    source.onmessage = handleEvent;

    const eventNames = [
      "episode",
      "episode_completed",
      "episode_end",
      "training_episode",
      "progress",

      "optimization_started",
      "optimization_progress",
      "optimization_completed",

      "completed",
      "finished",
      "failed",
    ];

    for (const name of eventNames) {
      source.addEventListener(
        name,
        handleEvent as EventListener
      );
    }

    source.onerror = () => {
      if (active) {
        setStreamConnected(false);
      }

      // EventSource automatically attempts reconnection.
      // The HTTP polling remains available as a fallback.
    };

    /* --------------------------------------------------
       HTTP status polling
    -------------------------------------------------- */

    const poll = async () => {
      try {
        const next = await api.training(
          created.training_id
        );

        if (!active) return;

        setStatus(next);

        if (isTrainingFinished(next.status)) {
          source.close();
          setStreamConnected(false);

          window.clearInterval(timer);
        }
      } catch {
        // Retry during the next polling interval.
      }
    };

    const timer = window.setInterval(
      () => void poll(),
      1200
    );

    void poll();

    /* --------------------------------------------------
       Cleanup
    -------------------------------------------------- */

    return () => {
      active = false;

      source.close();
      window.clearInterval(timer);
    };
  }, [created]);

  return {
    created,
    status,
    episodes,

    optimizations,
    optimizationStarted,

    starting,
    streamConnected,

    start,
  };
}