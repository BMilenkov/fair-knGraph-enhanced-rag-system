"""Run a named experiment via the full pipeline.

Shorthand for:
    python scripts/run_full_pipeline.py --config configs/experiments/<name>.yaml

Usage:
    python scripts/run_experiment.py --experiment kg2rag_fair
    python scripts/run_experiment.py --experiment baseline_naive --split dev --only EVALUATE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a named experiment")
    parser.add_argument("--experiment", required=True, help="Experiment name under configs/experiments/")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--start-from", default=None)
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    config_path = Path("configs/experiments") / f"{args.experiment}.yaml"
    if not config_path.exists():
        print(f"Error: Experiment config not found: {config_path}")
        available = sorted(p.stem for p in Path("configs/experiments").glob("*.yaml"))
        print(f"Available: {', '.join(available)}")
        sys.exit(1)

    # Forward to run_full_pipeline
    forward_args = ["run_full_pipeline.py", "--config", str(config_path), "--split", args.split]
    if args.start_from:
        forward_args.extend(["--start-from", args.start_from])
    if args.only:
        forward_args.extend(["--only", args.only])

    sys.argv = forward_args
    from scripts.run_full_pipeline import main as pipeline_main
    pipeline_main()


if __name__ == "__main__":
    main()
