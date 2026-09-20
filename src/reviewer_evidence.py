"""Generate reviewer-requested evidence from the frozen evaluation protocol.

This module does not tune on test labels. It retrains three already frozen
models on the original sampled training rows and evaluates binary attack
precision-recall curves on the identical held-out rows used in the benchmark.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from .config import ARTIFACTS, RESULTS, SEEDS
from .data import load_bundle
from .final_evaluation import make_model, probabilities, stratified_cap, temperature_fit


MODELS = ["AnytimeGraph-Guard", "LightGBM", "XGBoost"]
DATASETS = ["toniot", "edgeiiot", "apa_ddos"]
TRAIN_CAP = 60_000
RECALL_GRID = np.linspace(0.0, 1.0, 201)
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def temperature_scale(p: np.ndarray, temperature: float) -> np.ndarray:
    q = np.exp(np.log(np.clip(p, 1e-12, 1.0)) / temperature)
    return q / q.sum(axis=1, keepdims=True)


def interpolate_pr(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, float]:
    precision, recall, _ = precision_recall_curve(y, score)
    frame = pd.DataFrame({"recall": recall, "precision": precision})
    frame = frame.groupby("recall", as_index=False).precision.max().sort_values("recall")
    curve = np.interp(RECALL_GRID, frame.recall, frame.precision)
    return curve, float(average_precision_score(y, score))


def main():
    curves = []
    summaries = []
    for dataset in DATASETS:
        bundle = load_bundle(dataset)
        n_classes = len(bundle.classes)
        benign = bundle.classes.index({
            "toniot": "normal",
            "edgeiiot": "Normal",
            "apa_ddos": "Benign",
        }[dataset])
        for seed in SEEDS:
            idx = stratified_cap(bundle.y["train"], TRAIN_CAP, seed)
            for model_name in MODELS:
                graph = model_name == "AnytimeGraph-Guard"
                x = bundle.X_all if graph else bundle.X_base
                model = make_model(model_name, seed, n_classes)
                model.fit(x["train"][idx], bundle.y["train"][idx])
                p_val = probabilities(model, x["validation"], n_classes)
                p_test = probabilities(model, x["test"], n_classes)
                temperature = 1.0
                if model_name == "AnytimeGraph-Guard":
                    temperature = temperature_fit(bundle.y["validation"], p_val)
                    p_test = temperature_scale(p_test, temperature)
                attack_score = 1.0 - p_test[:, benign]
                precision, ap = interpolate_pr(bundle.y_binary["test"], attack_score)
                summaries.append({
                    "dataset": dataset,
                    "model": model_name,
                    "seed": seed,
                    "task": "binary",
                    "class": "attack",
                    "average_precision": ap,
                    "test_rows": len(attack_score),
                    "attack_prevalence": float(bundle.y_binary["test"].mean()),
                    "temperature": temperature,
                })
                for recall, value in zip(RECALL_GRID, precision):
                    curves.append({
                        "dataset": dataset,
                        "model": model_name,
                        "seed": seed,
                        "task": "binary",
                        "class": "attack",
                        "recall": recall,
                        "precision": float(value),
                    })

                class_curves = []
                class_aps = []
                for class_id, class_name in enumerate(bundle.classes):
                    binary_y = (bundle.y["test"] == class_id).astype(np.int8)
                    class_curve, class_ap = interpolate_pr(binary_y, p_test[:, class_id])
                    class_curves.append(class_curve)
                    class_aps.append(class_ap)
                    summaries.append({
                        "dataset": dataset,
                        "model": model_name,
                        "seed": seed,
                        "task": "one_vs_rest",
                        "class": class_name,
                        "average_precision": class_ap,
                        "test_rows": len(binary_y),
                        "attack_prevalence": float(binary_y.mean()),
                        "temperature": temperature,
                    })
                    for recall, value in zip(RECALL_GRID, class_curve):
                        curves.append({
                            "dataset": dataset,
                            "model": model_name,
                            "seed": seed,
                            "task": "one_vs_rest",
                            "class": class_name,
                            "recall": recall,
                            "precision": float(value),
                        })
                macro_curve = np.mean(class_curves, axis=0)
                summaries.append({
                    "dataset": dataset,
                    "model": model_name,
                    "seed": seed,
                    "task": "macro_ovr",
                    "class": "macro",
                    "average_precision": float(np.mean(class_aps)),
                    "test_rows": len(attack_score),
                    "attack_prevalence": np.nan,
                    "temperature": temperature,
                })
                for recall, value in zip(RECALL_GRID, macro_curve):
                    curves.append({
                        "dataset": dataset,
                        "model": model_name,
                        "seed": seed,
                        "task": "macro_ovr",
                        "class": "macro",
                        "recall": recall,
                        "precision": float(value),
                    })

    pd.DataFrame(curves).to_csv(RESULTS / "pr_curve_results.csv", index=False)
    pd.DataFrame(summaries).to_csv(RESULTS / "pr_curve_summary.csv", index=False)
    config = {
        "purpose": "reviewer evidence for imbalanced intrusion detection",
        "models": MODELS,
        "datasets": DATASETS,
        "seeds": SEEDS,
        "train_cap": TRAIN_CAP,
        "splits": "existing frozen manifests",
        "feature_rule": "proposed uses X_all; LightGBM and XGBoost use X_base as in the confirmatory benchmark",
        "selection": "none; models and hyperparameters copied from final_evaluation.py",
        "test_labels_used_for_tuning": False,
        "curve_tasks": ["binary benign versus attack", "one-vs-rest per class", "macro one-vs-rest"],
    }
    (ARTIFACTS / "protocol" / "reviewer_evidence_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
