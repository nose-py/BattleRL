"""Start a training job from a JSON request file."""

import argparse
import json
from pathlib import Path

from src.api import TrainingRequest, create_training
from src.sim.jobs import SingleTrainingManager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-request-path",
        required=True,
        type=Path,
        help="Path to a JSON file containing a TrainingRequest",
    )
    args = parser.parse_args()

    with args.training_request_path.open(encoding="utf-8") as request_file:
        request_data = json.load(request_file)

    request = TrainingRequest.model_validate(request_data)
    job = training_manager.start(
        config=request.config,
        episodes=request.episodes,
    )


if __name__ == "__main__":
    main()
