#!/usr/bin/env python3
"""Evaluate Kumo Tabular on the existing purged training-period folds.

Requires NVIDIA's structured-data-models (SDM). This is a separate benchmark:
it does not alter model selection or the existing project results. The CPU
defaults deliberately use only a small, randomly sampled context and one
estimator; they are a resource-constrained experiment, not a full-data result.
The score is computed over every candidate for each completed decision date.

Examples:
    python benchmark_kumo.py --max-batches 1  # API/runtime smoke test; incomplete
    python benchmark_kumo.py                  # both purged folds, all candidates
    python benchmark_kumo.py --context-size 1024 --batch-size 256 --estimators 2
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from food_inspections import config as C
from food_inspections.evaluation import rank_order
from food_inspections.features import PANEL_FEATURE_VERSION
from food_inspections.models import check_purged, design_matrix


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_progress(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def positive_probability(output) -> np.ndarray:
    """Select the probability of the actual failure class by name, not index."""
    frame = output.to_pandas()
    matches = [c for c in frame.columns if str(c) == "1"]
    if len(matches) != 1:
        raise ValueError(f"Expected one failure class '1'; output columns: {list(frame.columns)}")
    score = frame[matches[0]].to_numpy(dtype=float)
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("Kumo returned non-finite or invalid class probabilities")
    return score


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context-size", type=int, default=512,
                        help="Uniform sample of training rows per fold; default 512")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Number of query rows per prediction call; default 256")
    parser.add_argument("--estimators", type=int, default=1,
                        help="Kumo randomized-view ensemble size; default 1")
    parser.add_argument("--size", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4,
                        help="PyTorch CPU thread limit; default 4")
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--fold", choices=("P1", "P2", "all"), default="all")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Stop early for a smoke test; incomplete dates have no precision score")
    parser.add_argument("--output", type=Path, default=C.RESULTS_DIR / "kumo_benchmark.json")
    args = parser.parse_args(argv)
    if min(args.context_size, args.batch_size, args.estimators, args.threads) < 1:
        parser.error("context-size, batch-size, estimators, and threads must be positive")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("max-batches must be positive")
    try:
        import torch
        import sdm
    except ImportError as exc:
        raise SystemExit("Install optional dependencies: pip install structured-data-models torch") from exc
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but PyTorch cannot access a CUDA device")
    torch.set_num_threads(args.threads)
    if not C.PANEL_FILE.exists():
        raise SystemExit("Panel cache is missing; first run python run_project.py")

    panel = pd.read_csv(C.PANEL_FILE, parse_dates=["decision_date"],
                        dtype={"license_": str, "zip": str})
    if "panel_feature_version" not in panel or not panel["panel_feature_version"].eq(PANEL_FEATURE_VERSION).all():
        raise SystemExit("Panel cache has an incompatible feature version; rebuild with run_project.py")
    train = panel[panel["period"] == "train"].copy()
    X = design_matrix(train).astype("float32")
    schemes = [(name, spec) for name, spec in C.VALIDATION_SCHEMES.items()
               if spec["purged"] and (args.fold == "all" or name.startswith(args.fold + ":"))]
    if not schemes:
        raise ValueError("No purged validation schemes matched")
    result = {
        "status": "running", "complete": False,
        "description": "Kumo benchmark on purged training-period folds; no held-out test data used",
        "panel_sha256": sha256(C.PANEL_FILE),
        "python": platform.python_version(), "torch": torch.__version__,
        "structured_data_models": importlib.metadata.version("structured-data-models"),
        "settings": {"size": args.size, "device": args.device, "context_size": args.context_size,
                     "context_sampling": "uniform without replacement within each purged fit fold",
                     "batch_size": args.batch_size, "estimators": args.estimators,
                     "threads": args.threads,
                     "seed": args.seed, "fold": args.fold, "max_batches": args.max_batches,
                     "k": C.K, "features": list(X.columns)},
        "folds": {},
    }
    save_progress(args.output, result)
    started, batches = time.monotonic(), 0
    try:
        model = sdm.models.KumoTabular(task="classification", size=args.size, device=args.device)
        all_purged_names = [name for name, spec in C.VALIDATION_SCHEMES.items() if spec["purged"]]
        for name, spec in schemes:
            fold_number = all_purged_names.index(name)
            if not check_purged(spec["fit"], spec["val"]):
                raise ValueError(f"Unpurged scheme rejected: {name}")
            fit = train[train["decision_date"].isin(spec["fit"])]
            val = train[train["decision_date"].isin(spec["val"])]
            if fit.empty or val.empty:
                raise ValueError(f"Empty fit or validation data for {name}")
            context = fit.sample(n=min(args.context_size, len(fit)), random_state=args.seed + fold_number)
            if set(context["label"].astype(int).unique()) != {0, 1}:
                raise ValueError("Sampled context must include both outcome classes")
            context_frame = X.loc[context.index].copy()
            context_frame["target"] = context["label"].astype(int)
            # All design-matrix features are numeric, including one-hot columns.
            # Infer nothing from query batches: distinct-value heuristics can
            # assign different semantic types to the same feature per batch.
            feature_stypes = {column: "numerical" for column in X.columns}
            context_table = sdm.TableTensor.from_pandas(
                df=context_frame, stypes={**feature_stypes, "target": "categorical"},
                device=args.device)
            fold_result = {"status": "running", "fit_rows": len(fit), "context_rows": len(context),
                           "context_index_sha256": hashlib.sha256(
                               context.index.to_numpy(dtype="int64").tobytes()).hexdigest(),
                           "validation_rows": len(val), "dates": {}}
            result["folds"][name] = fold_result
            save_progress(args.output, result)
            print(f"[{name}] fit context={len(context):,}, validation={len(val):,}", flush=True)
            with torch.inference_mode():
                model.fit(x=context_table.drop_columns("target"), y=context_table["target"],
                          num_estimators=args.estimators, estimator_batch_size=1,
                          generator=torch.Generator(device=args.device).manual_seed(args.seed + fold_number))
                for T, rows in val.groupby("decision_date", sort=True):
                    if len(rows) < C.K:
                        raise ValueError(f"Date {T.date()} has fewer than {C.K} candidates")
                    scores = []
                    for start in range(0, len(rows), args.batch_size):
                        query_frame = X.loc[rows.index[start:start + args.batch_size]]
                        query = sdm.TableTensor.from_pandas(df=query_frame, stypes=feature_stypes,
                                                           device=args.device)
                        scores.extend(positive_probability(model.predict(query)).tolist())
                        batches += 1
                        if args.max_batches is not None and batches >= args.max_batches and len(scores) < len(rows):
                            fold_result["partial_date"] = str(T.date())
                            fold_result["partial_rows_scored"] = len(scores)
                            result["status"] = "incomplete_smoke_test"
                            result["elapsed_seconds"] = time.monotonic() - started
                            save_progress(args.output, result)
                            print("Stopped at max-batches; no score reported for partial date.", flush=True)
                            return
                    if len(scores) != len(rows):
                        raise RuntimeError("Incomplete date; refusing to calculate Precision@K")
                    order = rank_order(rows, scores)
                    baseline_order = rank_order(rows, rows["hist_fail_rate_smooth"])
                    date_key = str(T.date())
                    fold_result["dates"][date_key] = {
                        "rows": len(rows),
                        "kumo_precision_at_k": float(rows["label"].to_numpy()[order[:C.K]].mean()),
                        "baseline_precision_at_k": float(rows["label"].to_numpy()[baseline_order[:C.K]].mean()),
                    }
                    save_progress(args.output, result)
                    print(f"  {date_key}: Precision@{C.K}={fold_result['dates'][date_key]['kumo_precision_at_k']:.3f}", flush=True)
                    if args.max_batches is not None and batches >= args.max_batches:
                        result["status"] = "incomplete_smoke_test"
                        result["elapsed_seconds"] = time.monotonic() - started
                        save_progress(args.output, result)
                        return
            model.clear()
            fold_result["status"] = "complete"
            fold_result["mean_precision_at_k"] = float(np.mean(
                [entry["kumo_precision_at_k"] for entry in fold_result["dates"].values()]))
            save_progress(args.output, result)
        result["complete"] = args.fold == "all" and len(schemes) == sum(
            bool(s["purged"]) for s in C.VALIDATION_SCHEMES.values())
        result["status"] = "complete" if result["complete"] else "incomplete_selected_fold_only"
        if result["complete"]:
            result["mean_purged_precision_at_k"] = float(np.mean(
                [fold["mean_precision_at_k"] for fold in result["folds"].values()]))
        result["elapsed_seconds"] = time.monotonic() - started
        save_progress(args.output, result)
    except BaseException as exc:
        result["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_seconds"] = time.monotonic() - started
        save_progress(args.output, result)
        raise


if __name__ == "__main__":
    main()
