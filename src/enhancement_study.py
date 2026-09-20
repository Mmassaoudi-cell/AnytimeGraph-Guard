"""Post-hoc, validation-selected enhancement and diagnostic studies.

The held-out data were already examined in the original study.  Consequently,
enhanced test results written here are explicitly exploratory.  Every ensemble
weight and temperature is nevertheless selected using validation labels only.
"""
from __future__ import annotations

import json
from itertools import product

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import ttest_rel, wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

from .config import ARTIFACTS, RESULTS, SEEDS
from .data import load_bundle
from .final_evaluation import probabilities, stratified_cap, temperature_fit
from .metrics import classification_metrics

DATASETS = ["toniot", "edgeiiot", "apa_ddos"]
TRAIN_CAP = 60_000
BASE_WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
STAGES = {1, 5, 10, *range(20, 201, 10)}


def base_model(seed: int, **overrides):
    params = dict(
        max_iter=200,
        max_leaf_nodes=63,
        learning_rate=0.06,
        l2_regularization=2.0,
        min_samples_leaf=30,
        class_weight="balanced",
        random_state=seed,
    )
    params.update(overrides)
    return HistGradientBoostingClassifier(**params)


def expert_model(seed: int):
    return LGBMClassifier(
        n_estimators=260,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
        verbosity=-1,
    )


def calibrate(p: np.ndarray, temperature: float) -> np.ndarray:
    q = np.exp(np.log(np.clip(p, 1e-12, 1.0)) / temperature)
    return q / q.sum(axis=1, keepdims=True)


def mix(p_base: np.ndarray, p_expert: np.ndarray, weight: float) -> np.ndarray:
    p = weight * p_base + (1.0 - weight) * p_expert
    return p / p.sum(axis=1, keepdims=True)


def choose_weight(y: np.ndarray, p_base: np.ndarray, p_expert: np.ndarray):
    rows = []
    for weight in BASE_WEIGHTS:
        p = mix(p_base, p_expert, weight)
        score = f1_score(y, p.argmax(1), average="macro", zero_division=0)
        rows.append((weight, float(score)))
    # Retain the original branch when validation scores tie.
    best = max(rows, key=lambda z: (z[1], z[0]))
    return best[0], rows


def bootstrap_delta(values: np.ndarray, seed: int = 17):
    rng = np.random.default_rng(seed)
    means = np.empty(10_000)
    for i in range(len(means)):
        means[i] = rng.choice(values, len(values), replace=True).mean()
    return np.quantile(means, [0.025, 0.975])


def main():
    enhancement = []
    confusions = []
    robustness = []
    convergence = []
    sensitivity = []

    for dataset in DATASETS:
        bundle = load_bundle(dataset)
        n_classes = len(bundle.classes)
        for seed in SEEDS:
            idx = stratified_cap(bundle.y["train"], TRAIN_CAP, seed)
            x_train = bundle.X_all["train"][idx]
            y_train = bundle.y["train"][idx]

            base = base_model(seed).fit(x_train, y_train)
            expert = expert_model(seed).fit(x_train, y_train)

            p_base_val_raw = probabilities(base, bundle.X_all["validation"], n_classes)
            p_expert_val_raw = probabilities(expert, bundle.X_all["validation"], n_classes)
            temp_base = temperature_fit(bundle.y["validation"], p_base_val_raw)
            temp_expert = temperature_fit(bundle.y["validation"], p_expert_val_raw)
            p_base_val = calibrate(p_base_val_raw, temp_base)
            p_expert_val = calibrate(p_expert_val_raw, temp_expert)
            weight, weight_rows = choose_weight(
                bundle.y["validation"], p_base_val, p_expert_val
            )

            for candidate_weight, score in weight_rows:
                sensitivity.append({
                    "dataset": dataset,
                    "seed": seed,
                    "factor": "base_blend_weight",
                    "value": candidate_weight,
                    "validation_macro_f1": score,
                    "train_cap": len(idx),
                })

            p_base_test = calibrate(
                probabilities(base, bundle.X_all["test"], n_classes), temp_base
            )
            p_expert_test = calibrate(
                probabilities(expert, bundle.X_all["test"], n_classes), temp_expert
            )
            p_enhanced_test = mix(p_base_test, p_expert_test, weight)

            for model_name, p_val, p_test in [
                ("Frozen base", p_base_val, p_base_test),
                (
                    "Validation-gated dual booster",
                    mix(p_base_val, p_expert_val, weight),
                    p_enhanced_test,
                ),
            ]:
                enhancement.append({
                    "dataset": dataset,
                    "seed": seed,
                    "model": model_name,
                    "selected_base_weight": weight,
                    "base_temperature": temp_base,
                    "expert_temperature": temp_expert,
                    "validation_macro_f1": f1_score(
                        bundle.y["validation"], p_val.argmax(1),
                        average="macro", zero_division=0,
                    ),
                    **classification_metrics(bundle.y["test"], p_test),
                })

            pred = p_enhanced_test.argmax(1)
            cm = confusion_matrix(
                bundle.y["test"], pred, labels=np.arange(n_classes)
            )
            for i, true_name in enumerate(bundle.classes):
                for j, pred_name in enumerate(bundle.classes):
                    confusions.append({
                        "dataset": dataset,
                        "seed": seed,
                        "true": true_name,
                        "predicted": pred_name,
                        "count": int(cm[i, j]),
                        "row_fraction": float(cm[i, j] / max(1, cm[i].sum())),
                    })

            # Curves describe the primary HGB branch.  The companion expert is
            # fit once and its validation-gated contribution is reported above.
            if seed in SEEDS[:3]:
                train_stages = base.staged_predict_proba(x_train)
                val_stages = base.staged_predict_proba(bundle.X_all["validation"])
                labels = np.arange(n_classes)
                for stage, (p_train, p_val) in enumerate(
                    zip(train_stages, val_stages), start=1
                ):
                    if stage not in STAGES:
                        continue
                    for split, y, p in [
                        ("training", y_train, p_train),
                        ("validation", bundle.y["validation"], p_val),
                    ]:
                        convergence.append({
                            "dataset": dataset,
                            "seed": seed,
                            "iteration": stage,
                            "split": split,
                            "loss": float(log_loss(y, p, labels=labels)),
                            "accuracy": float(accuracy_score(y, p.argmax(1))),
                        })

                rng = np.random.default_rng(seed)
                x_clean = bundle.X_all["test"]
                conditions = [("clean", "clean", 0.0, x_clean)]
                for rate in [0.10, 0.20, 0.30]:
                    x = x_clean.copy()
                    mask = rng.random(x[:, :-6].shape) < rate
                    x[:, :-6][mask] = 0
                    conditions.append(("missing", "missing features", rate, x))
                for sigma in [0.05, 0.10, 0.20]:
                    x = x_clean.copy()
                    x[:, :-6] += rng.normal(0, sigma, size=x[:, :-6].shape)
                    conditions.append(("noise", "feature noise", sigma, x))
                for rate in [0.10, 0.20, 0.40]:
                    x = x_clean.copy()
                    take = rng.random(len(x)) < rate
                    source = rng.permutation(len(x))[: take.sum()]
                    x[take, -6:] = x[source, -6:]
                    conditions.append(("context", "graph-context corruption", rate, x))
                x = x_clean.copy()
                x[:, -6:] = 0
                conditions.append(("topology", "topology removed", 1.0, x))

                for family, label, severity, x_condition in conditions:
                    pb = calibrate(probabilities(base, x_condition, n_classes), temp_base)
                    pe = calibrate(probabilities(expert, x_condition, n_classes), temp_expert)
                    p = mix(pb, pe, weight)
                    robustness.append({
                        "dataset": dataset,
                        "seed": seed,
                        "family": family,
                        "condition": label,
                        "severity": severity,
                        "selected_base_weight": weight,
                        **classification_metrics(bundle.y["test"], p),
                    })

        # One-factor-at-a-time sensitivity uses validation data and three seeds.
        variants = [
            ("max_iter", 100, {"max_iter": 100}),
            ("max_iter", 200, {}),
            ("max_iter", 300, {"max_iter": 300}),
            ("max_leaf_nodes", 31, {"max_leaf_nodes": 31}),
            ("max_leaf_nodes", 63, {}),
            ("max_leaf_nodes", 127, {"max_leaf_nodes": 127}),
            ("learning_rate", 0.03, {"learning_rate": 0.03}),
            ("learning_rate", 0.06, {}),
            ("learning_rate", 0.10, {"learning_rate": 0.10}),
            ("min_samples_leaf", 15, {"min_samples_leaf": 15}),
            ("min_samples_leaf", 30, {}),
            ("min_samples_leaf", 60, {"min_samples_leaf": 60}),
        ]
        for seed in SEEDS[:3]:
            idx = stratified_cap(bundle.y["train"], TRAIN_CAP, seed)
            cache = {}
            for factor, value, overrides in variants:
                key = tuple(sorted(overrides.items()))
                if key not in cache:
                    model = base_model(seed, **overrides).fit(
                        bundle.X_all["train"][idx], bundle.y["train"][idx]
                    )
                    p = probabilities(model, bundle.X_all["validation"], n_classes)
                    cache[key] = f1_score(
                        bundle.y["validation"], p.argmax(1),
                        average="macro", zero_division=0,
                    )
                sensitivity.append({
                    "dataset": dataset,
                    "seed": seed,
                    "factor": factor,
                    "value": value,
                    "validation_macro_f1": float(cache[key]),
                    "train_cap": len(idx),
                })

    enhancement_frame = pd.DataFrame(enhancement)
    enhancement_frame.to_csv(RESULTS / "enhancement_results.csv", index=False)
    pd.DataFrame(confusions).to_csv(
        RESULTS / "enhanced_confusion_matrices.csv", index=False
    )
    pd.DataFrame(robustness).to_csv(
        RESULTS / "enhanced_robustness_results.csv", index=False
    )
    pd.DataFrame(convergence).to_csv(
        RESULTS / "convergence_curves.csv", index=False
    )
    pd.DataFrame(sensitivity).to_csv(
        RESULTS / "sensitivity_results.csv", index=False
    )

    stats = []
    for dataset in DATASETS + ["all"]:
        frame = enhancement_frame if dataset == "all" else enhancement_frame[
            enhancement_frame.dataset == dataset
        ]
        pivot = frame.pivot(index=["dataset", "seed"], columns="model", values="macro_f1")
        delta = (
            pivot["Validation-gated dual booster"] - pivot["Frozen base"]
        ).to_numpy()
        ci_low, ci_high = bootstrap_delta(delta)
        try:
            wp = float(wilcoxon(delta).pvalue) if np.any(delta) else 1.0
        except ValueError:
            wp = 1.0
        stats.append({
            "dataset": dataset,
            "n_pairs": len(delta),
            "mean_delta_macro_f1": float(delta.mean()),
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
            "paired_t_p": float(ttest_rel(
                pivot["Validation-gated dual booster"], pivot["Frozen base"]
            ).pvalue) if np.std(delta) > 0 else 1.0,
            "wilcoxon_p": wp,
        })
    pd.DataFrame(stats).to_csv(RESULTS / "enhancement_statistics.csv", index=False)

    config = {
        "status": "post-hoc exploratory because the original held-out results were already examined",
        "selection_labels": "validation only",
        "base": "frozen graph-context HistGradientBoosting encoder",
        "expert": "LightGBM trained on the same train-only graph-context features",
        "base_weight_grid": BASE_WEIGHTS,
        "tie_break": "largest base weight",
        "temperatures": "fit separately on validation NLL",
        "test_labels_used_for_selection": False,
        "convergence_scope": "primary HGB branch; three seeds",
        "sensitivity_scope": "one-factor-at-a-time validation macro-F1; three seeds",
        "robustness_scope": "frozen validation-selected enhancement; three seeds",
    }
    path = ARTIFACTS / "protocol" / "enhancement_config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
