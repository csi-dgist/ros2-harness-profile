#!/usr/bin/env python3
"""Train reproducible linear and MLP behavioral clones from expert Twist traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REVISION = "pit-portability-bc-v2-group-split"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_trace(path: Path, history: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    sequences: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        run = str(row["run_id"])
        sequences.setdefault(run, []).append(
            (float(row["linear_x"]), float(row["angular_z"]))
        )
    features, targets, groups = [], [], []
    for run in sorted(sequences):
        sequence = sequences[run]
        for index in range(history, len(sequence)):
            window = sequence[index - history : index]
            features.append([value for pair in window for value in pair])
            targets.append(sequence[index])
            groups.append(run)
    if len(features) < 100:
        raise ValueError(f"need at least 100 temporal samples, found {len(features)}")
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        np.asarray(groups),
    )


def standardize(train_x: np.ndarray, all_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return (all_x - mean) / scale, mean, scale


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(prediction - target)))


def group_split(
    groups: np.ndarray, seed: int, test_fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray, set[str], set[str]]:
    rng = np.random.default_rng(seed)
    run_ids = np.asarray(sorted(set(groups.tolist())))
    if len(run_ids) < 2:
        raise ValueError("need at least two expert runs for a group-held-out split")
    shuffled_runs = run_ids[rng.permutation(len(run_ids))]
    test_run_count = max(1, round(len(shuffled_runs) * test_fraction))
    test_runs = set(shuffled_runs[-test_run_count:].tolist())
    train_runs = set(shuffled_runs[:-test_run_count].tolist())
    train_indices = np.flatnonzero(np.isin(groups, list(train_runs)))
    test_indices = np.flatnonzero(np.isin(groups, list(test_runs)))
    return train_indices, test_indices, train_runs, test_runs


def train_linear(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([x, np.ones(len(x))])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[-1, -1] = 0.0
    solution = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return solution[:-1], solution[-1]


def train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    hidden: int,
    epochs: int,
    learning_rate: float,
) -> tuple[dict[str, np.ndarray], list[float]]:
    rng = np.random.default_rng(seed)
    params = {
        "w1": rng.normal(0.0, 0.15, size=(x.shape[1], hidden)),
        "b1": np.zeros(hidden),
        "w2": rng.normal(0.0, 0.15, size=(hidden, y.shape[1])),
        "b2": np.zeros(y.shape[1]),
    }
    first = {name: np.zeros_like(value) for name, value in params.items()}
    second = {name: np.zeros_like(value) for name, value in params.items()}
    losses: list[float] = []
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for epoch in range(1, epochs + 1):
        hidden_value = np.tanh(x @ params["w1"] + params["b1"])
        prediction = hidden_value @ params["w2"] + params["b2"]
        difference = prediction - y
        gradient_prediction = 2.0 * difference / len(x)
        gradients = {
            "w2": hidden_value.T @ gradient_prediction,
            "b2": gradient_prediction.sum(axis=0),
        }
        gradient_hidden = (gradient_prediction @ params["w2"].T) * (1.0 - hidden_value**2)
        gradients["w1"] = x.T @ gradient_hidden
        gradients["b1"] = gradient_hidden.sum(axis=0)
        for name in params:
            first[name] = beta1 * first[name] + (1.0 - beta1) * gradients[name]
            second[name] = beta2 * second[name] + (1.0 - beta2) * gradients[name] ** 2
            corrected_first = first[name] / (1.0 - beta1**epoch)
            corrected_second = second[name] / (1.0 - beta2**epoch)
            params[name] -= learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            losses.append(mse(prediction, y))
    return params, losses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x, y, groups = load_trace(args.trace, args.history)
    train_indices, test_indices, train_runs, test_runs = group_split(groups, args.seed)
    standardized, mean, scale = standardize(x[train_indices], x)
    train_x, test_x = standardized[train_indices], standardized[test_indices]
    train_y, test_y = y[train_indices], y[test_indices]

    linear_w, linear_b = train_linear(train_x, train_y, ridge=1e-4)
    linear_path = args.output_dir / "linear_bc.npz"
    np.savez(linear_path, w=linear_w, b=linear_b, mean=mean, scale=scale)
    linear_train = mse(train_x @ linear_w + linear_b, train_y)
    linear_test = mse(test_x @ linear_w + linear_b, test_y)

    mlp, losses = train_mlp(
        train_x,
        train_y,
        seed=args.seed,
        hidden=16,
        epochs=args.epochs,
        learning_rate=0.01,
    )
    mlp_path = args.output_dir / "mlp_bc.npz"
    np.savez(mlp_path, **mlp, mean=mean, scale=scale)
    mlp_train_prediction = np.tanh(train_x @ mlp["w1"] + mlp["b1"]) @ mlp["w2"] + mlp["b2"]
    mlp_test_prediction = np.tanh(test_x @ mlp["w1"] + mlp["b1"]) @ mlp["w2"] + mlp["b2"]

    manifest = {
        "revision": REVISION,
        "seed": args.seed,
        "history": args.history,
        "trace": str(args.trace),
        "trace_sha256": sha256(args.trace),
        "samples": len(x),
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "split_kind": "seeded_group_holdout_by_expert_run",
        "train_run_ids": sorted(train_runs),
        "test_run_ids": sorted(test_runs),
        "split_indices_sha256": hashlib.sha256(
            np.concatenate([train_indices, test_indices]).tobytes()
        ).hexdigest(),
        "models": {
            "linear_bc": {
                "path": str(linear_path),
                "sha256": sha256(linear_path),
                "train_mse": linear_train,
                "test_mse": linear_test,
            },
            "mlp_bc": {
                "path": str(mlp_path),
                "sha256": sha256(mlp_path),
                "train_mse": mse(mlp_train_prediction, train_y),
                "test_mse": mse(mlp_test_prediction, test_y),
                "loss_checkpoints": losses,
            },
        },
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
