import type {
  Catalog,
  SimulationConfig,
  SessionResponse,
  WorldState,
  StepResponse,
  TrainingCreated,
  TrainingStatus,
} from "../types/simulation";

const BASE = (
  import.meta.env.VITE_API_BASE_URL || "/api"
).replace(/\/$/, "");

function url(path: string): string {
  return `${BASE}${path}`;
}

async function request<T>(
  path: string,
  method: string = "GET",
  body?: unknown
): Promise<T> {
  const response = await fetch(url(path), {
    method,
    headers: {
      ...(body !== undefined
        ? { "Content-Type": "application/json" }
        : {}),
    },
    ...(body !== undefined
      ? { body: JSON.stringify(body) }
      : {}),
  });

  if (!response.ok) {
    let message = `HTTP ${response.status}`;

    try {
      const error = await response.json();

      if (typeof error.detail === "string") {
        message = error.detail;
      } else if (Array.isArray(error.detail)) {
        message = error.detail
          .map((e: { msg?: string; loc?: unknown[] }) =>
            `${e.loc?.join(".") ?? "error"}: ${e.msg ?? "invalid"}`
          )
          .join("; ");
      } else {
        message = JSON.stringify(error);
      }
    } catch {
      // We keep the HTTP message.
    }

    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

function encoded(id: string): string {
  return encodeURIComponent(id);
}


function isObject(
  value: unknown
): value is Record<string, unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value)
  );
}

function isWorldState(
  value: unknown
): value is WorldState {
  return (
    isObject(value) &&
    typeof value.step === "number" &&
    typeof value.width === "number" &&
    typeof value.height === "number" &&
    Array.isArray(value.actors)
  );
}

function normalizeSessionResponse(
  response: unknown,
  sessionId: string
): SessionResponse {
  if (!isObject(response)) {
    throw new Error("Invalid session response.");
  }

  const state = isWorldState(response.state)
    ? response.state
    : isWorldState(response)
      ? response
      : null;

  if (!state) {
    throw new Error(
      "The response does not contain a valid simulation state."
    );
  }

  return {
    session_id:
      typeof response.session_id === "string"
        ? response.session_id
        : sessionId,
    state,
  };
}

function actionDictionary(
  value: unknown
): Record<string, number> {
  const values = numberDictionary(value, "actions");

  if (Object.values(values).some(
    (id) => !Number.isInteger(id)
  )) {
    throw new Error(
      "El backend devolvió un ID de acción no entero."
    );
  }

  return values;
}

function numberDictionary(
  value: unknown,
  name: string
): Record<string, number> {
  if (
    !isObject(value) ||
    Object.values(value).some(
      (v) => typeof v !== "number" || !Number.isFinite(v)
    )
  ) {
    throw new Error(`Field '${name}' invalid in /step.`);
  }

  return value as Record<string, number>;
}

function booleanDictionary(
  value: unknown,
  name: string
): Record<string, boolean> {
  if (
    !isObject(value) ||
    Object.values(value).some(
      (v) => typeof v !== "boolean"
    )
  ) {
    throw new Error(`Field '${name}' invalid in /step.`);
  }

  return value as Record<string, boolean>;
}

function normalizeStepResponse(
  response: unknown,
  sessionId: string
): StepResponse {
  const normalized = normalizeSessionResponse(
    response,
    sessionId
  );

  const raw = response as Record<string, unknown>;

  // The five fields correspond to the response
  // from /step, not the WorldState.
  return {
    ...normalized,

    total_rewards: numberDictionary(
      raw.total_rewards,
      "total_rewards"
    ),

    actions: actionDictionary(raw.actions),

    rewards: numberDictionary(
      raw.rewards,
      "rewards"
    ),

    terminations: booleanDictionary(
      raw.terminations,
      "terminations"
    ),

    truncations: booleanDictionary(
      raw.truncations,
      "truncations"
    ),
  };
}


export const api = {
  catalog(): Promise<Catalog> {
    return request("/catalog");
  },

  startTraining(
    config: SimulationConfig,
    episodes: number
  ): Promise<TrainingCreated> {
    return request("/trainings", "POST", {
      config,
      episodes,
    });
  },

  training(id: string): Promise<TrainingStatus> {
    return request(`/trainings/${encoded(id)}`);
  },

  createSession(
    config?: SimulationConfig,
    trainingId?: string
  ): Promise<SessionResponse> {
    return request("/sessions", "POST",
      trainingId
        ? { training_id: trainingId }
        : { config }
    );
  },

  session(id: string): Promise<SessionResponse> {
    return request(`/sessions/${encoded(id)}`);
  },

  async step(id: string): Promise<StepResponse> {
    const response = await request<unknown>(
      `/sessions/${encoded(id)}/step`,
      "POST",
      {}
    );

    return normalizeStepResponse(response, id);
  },

  async reset(
    id: string,
    seed: number | null = null
  ): Promise<SessionResponse> {
    const response = await request<
      SessionResponse | WorldState
    >(
      `/sessions/${encoded(id)}/reset`,
      "POST",
      { seed }
    );

    return normalizeSessionResponse(response, id);
  },

  deleteSession(id: string): Promise<{ deleted: boolean }> {
    return request(`/sessions/${encoded(id)}`, "DELETE");
  },
};

/**
 * Normalize URLs SSE returned by FastAPI.
 *
 * If we are using the Vite proxy and FastAPI returns
 * /trainings/... or http://localhost:8000/trainings/...,
 * we connect through /api/trainings/...
 */
export function trainingEventsUrl(
  trainingId: string,
  providedUrl?: string
): string {
  const fallback = `/trainings/${encoded(trainingId)}/events`;

  if (!providedUrl) {
    return url(fallback);
  }

  const resolved = new URL(
    providedUrl,
    window.location.origin
  );

  if (BASE.startsWith("/")) {
    const isLocal =
      resolved.origin === window.location.origin ||
      ["localhost", "127.0.0.1"].includes(resolved.hostname);

    if (isLocal) {
      const path = resolved.pathname.startsWith(`${BASE}/`)
        ? resolved.pathname.slice(BASE.length)
        : resolved.pathname;

      return url(path) + resolved.search;
    }
  }

  return resolved.toString();
}