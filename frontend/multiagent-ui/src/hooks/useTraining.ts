import {
  useCallback,
  useEffect,
  useState,
} from "react";

import { api, trainingEventsUrl } from "../lib/api";

import type {
  EpisodeEvent,
  SimulationConfig,
  TrainingCreated,
  TrainingStatus,
} from "../types/simulation";

export function isTrainingFinished(status?: string): boolean {
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

export function isTrainingSuccessful(status?: string): boolean {
  return [
    "completed",
    "finished",
    "succeeded",
    "success",
  ].includes((status ?? "").toLowerCase());
}

function parseEpisode(value: unknown): EpisodeEvent | null {
  if (!value || typeof value !== "object") return null;

  const data = value as Record<string, unknown>;

  const payload =
    data.data &&
    typeof data.data === "object" &&
    !Array.isArray(data.data)
      ? data.data as Record<string, unknown>
      : data;

  if (typeof payload.episode !== "number") {
    return null;
  }

  return {
    episode: payload.episode,
    steps: Number(payload.steps ?? 0),

    winner: typeof payload.winner === "string"
      ? payload.winner
      : null,

    finish_reason:
      typeof payload.finish_reason === "string"
        ? payload.finish_reason
        : null,

    agent_rewards:
      payload.agent_rewards as Record<string, number> ?? {},

    team_rewards:
      payload.team_rewards as Record<string, number> ?? {},

    kills:
      payload.kills as Record<string, number> ?? {},

    goals:
      payload.goals as Record<string, number> ?? {},
  };
}

export function useTraining() {
  const [created, setCreated] =
    useState<TrainingCreated | null>(null);

  const [status, setStatus] =
    useState<TrainingStatus | null>(null);

  const [episodes, setEpisodes] =
    useState<EpisodeEvent[]>([]);

  const [starting, setStarting] = useState(false);
  const [streamConnected, setStreamConnected] = useState(false);

  const start = useCallback(async (
    config: SimulationConfig,
    count: number
  ) => {
    setStarting(true);
    setCreated(null);
    setStatus(null);
    setEpisodes([]);
    setStreamConnected(false);

    try {
      const result = await api.startTraining(config, count);

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

  useEffect(() => {
    if (!created) return;

    let active = true;

    const source = new EventSource(
      trainingEventsUrl(
        created.training_id,
        created.events_url
      )
    );

    const onEvent = (event: MessageEvent) => {
      try {
        const payload: unknown = JSON.parse(event.data);
        const episode = parseEpisode(payload);

        if (episode && active) {
          setEpisodes((previous) => {
            const next = new Map<number, EpisodeEvent>(
                previous.map((e) => [e.episode, e] as const)
            );

            next.set(episode.episode, episode);

            return [...next.values()]
              .sort((a, b) => a.episode - b.episode);
          });
        }
      } catch {
        // Informational messages that are not valid JSON.
      }
    };

    source.onopen = () => {
      if (active) setStreamConnected(true);
    };

    source.onmessage = onEvent;

    // SSE with explicit event names.
    const eventNames = [
      "episode",
      "episode_completed",
      "episode_end",
      "training_episode",
      "progress",
      "completed",
      "finished",
      "failed",
      "error",
    ];

    for (const name of eventNames) {
      source.addEventListener(
        name,
        onEvent as EventListener
      );
    }

    source.onerror = () => {
      setStreamConnected(false);

      // Polling will continue to work.
      source.close();
    };

    const poll = async () => {
      try {
        const next = await api.training(created.training_id);

        if (!active) return;

        setStatus(next);

        if (isTrainingFinished(next.status)) {
          source.close();
          setStreamConnected(false);
          clearInterval(timer);
        }
      } catch {
        // Allow retrying on the next poll.
      }
    };

    const timer = window.setInterval(poll, 1200);

    void poll();

    return () => {
      active = false;
      source.close();
      clearInterval(timer);
    };
  }, [created]);

  return {
    created,
    status,
    episodes,
    starting,
    streamConnected,
    start,
  };
}