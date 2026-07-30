#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


def find_trainer_state(run_dir: Path) -> Path:
    candidates = sorted(
        run_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda path: int(path.parent.name.rsplit("-", 1)[-1]),
    )
    if not candidates:
        candidate = run_dir / "trainer_state.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No trainer_state.json found under {run_dir}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.05,
        help="Minimum relative first-to-last loss-window improvement for PASS.",
    )
    args = parser.parse_args()

    state_path = find_trainer_state(args.run_dir)
    state = json.loads(state_path.read_text())
    records = [
        entry
        for entry in state.get("log_history", [])
        if "loss" in entry and "step" in entry
    ]
    if not records:
        raise RuntimeError(f"No loss records found in {state_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "loss.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "loss", "learning_rate", "grad_norm"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "step": record["step"],
                    "loss": record["loss"],
                    "learning_rate": record.get("learning_rate", ""),
                    "grad_norm": record.get("grad_norm", ""),
                }
            )

    losses = [float(record["loss"]) for record in records]
    window = min(5, max(1, len(losses) // 2))
    initial = sum(losses[:window]) / window
    final = sum(losses[-window:]) / window
    improvement = (initial - final) / initial if initial else 0.0
    converging = final < initial and improvement >= args.min_improvement

    summary = "\n".join(
        [
            f"trainer_state={state_path}",
            f"loss_records={len(losses)}",
            f"first_window_mean={initial:.6f}",
            f"last_window_mean={final:.6f}",
            f"relative_improvement={improvement:.2%}",
            f"minimum_loss={min(losses):.6f}",
            f"pass_threshold={args.min_improvement:.2%}",
            f"convergence_check={'PASS' if converging else 'INCONCLUSIVE'}",
        ]
    )
    (args.output_dir / "summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
