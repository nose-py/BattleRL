import { type FormEvent } from "react";

import {
  CheckCircle2,
  Clock3,
  Copy,
  Database,
  LoaderCircle,
  Search,
} from "lucide-react";

import {
  isTrainingFinished,
  isTrainingSuccessful,
} from "../hooks/useTraining";

import type {
  TrainingStatus,
} from "../types/simulation";

export type WorkspaceMode = "new" | "existing";

interface ExistingTrainingPanelProps {
  mode: WorkspaceMode;

  onModeChange: (mode: WorkspaceMode) => void;

  trainingId: string;
  onTrainingIdChange: (value: string) => void;

  training: TrainingStatus | null;
  loading: boolean;

  sessionActive: boolean;

  onLoad: () => void;
  onCopyConfig: () => void;
}

export default function ExistingTrainingPanel({
  mode,
  onModeChange,
  trainingId,
  onTrainingIdChange,
  training,
  loading,
  sessionActive,
  onLoad,
  onCopyConfig,
}: ExistingTrainingPanelProps) {
  const successful = isTrainingSuccessful(
    training?.status
  );

  const finished = isTrainingFinished(
    training?.status
  );

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!loading && trainingId.trim()) {
      onLoad();
    }
  }

  return (
    <div className="panel-card">
      <div className="card-heading">
        <div>
          <div className="eyebrow">
            TRAINING SOURCE
          </div>

          <h3>
            <Database size={17} />
            Configuration source
          </h3>
        </div>
      </div>

      <div className="segmented training-source-switch">
        <button
          type="button"
          className={mode === "new" ? "active" : ""}
          disabled={sessionActive}
          onClick={() => onModeChange("new")}
        >
          New
        </button>

        <button
          type="button"
          className={mode === "existing" ? "active" : ""}
          disabled={sessionActive}
          onClick={() => onModeChange("existing")}
        >
          Existing training
        </button>
      </div>

      {mode === "existing" && (
        <div className="existing-training-content">
          <form
            className="existing-training-form"
            onSubmit={submit}
          >
            <label className="field">
              <span>Training ID</span>

              <input
                type="text"
                value={trainingId}
                placeholder="Enter an existing training ID"
                autoComplete="off"
                spellCheck={false}
                disabled={loading || sessionActive}
                onChange={(event) => {
                  onTrainingIdChange(event.target.value);
                }}
              />
            </label>

            <button
              type="submit"
              className="primary-button full"
              disabled={
                loading ||
                sessionActive ||
                !trainingId.trim()
              }
            >
              {loading ? (
                <LoaderCircle
                  size={16}
                  className="spinning"
                />
              ) : (
                <Search size={16} />
              )}

              {loading ? "Loading..." : "Load training"}
            </button>
          </form>

          {training && (
            <div className="loaded-training">
              <div className="loaded-training-header">
                {successful ? (
                  <CheckCircle2
                    size={19}
                    color="#55e3c1"
                  />
                ) : (
                  <Clock3
                    size={19}
                    color="#efbd83"
                  />
                )}

                <div>
                  <strong>
                    {successful
                      ? "Training ready"
                      : finished
                        ? "Training unavailable"
                        : "Training in progress"}
                  </strong>

                  <small>{training.status}</small>
                </div>
              </div>

              <div className="loaded-training-id">
                {training.training_id}
              </div>

              <div className="stat-line">
                <span>Completed episodes</span>

                <strong>
                  {training.completed_episodes}
                  /{training.episodes}
                </strong>
              </div>

              <div className="stat-line">
                <span>Original configuration</span>

                <strong>
                  {training.config
                    ? "Available"
                    : "Not exposed by API"}
                </strong>
              </div>

              {training.error && (
                <div className="error-inline">
                  {training.error}
                </div>
              )}

              {!finished && (
                <p className="helper-text">
                  Reload the training status to check
                  whether optimization has completed.
                </p>
              )}

              {training.config && (
                <button
                  type="button"
                  className="subtle-button full"
                  disabled={sessionActive}
                  onClick={onCopyConfig}
                >
                  <Copy size={14} />
                  Copy configuration to a new draft
                </button>
              )}

              {successful && (
                <p className="helper-text">
                  This training can be used to create
                  a new trained simulation session.
                </p>
              )}
            </div>
          )}

          <p className="helper-text">
            Training IDs are verified through the API.
            They do not depend on browser storage.
          </p>
        </div>
      )}
    </div>
  );
}