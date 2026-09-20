"""Freeze leakage-safe row manifests before candidate or benchmark evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import ARTIFACTS, PRIMARY_FILES, REPORTS, RESULTS


SPECS = {
    "toniot": {
        "label": "type", "binary": "label", "benign": "normal",
        "src": "src_ip", "dst": "dst_ip", "relation": ["proto", "service", "conn_state"],
        "split_basis": "directed source-destination pair",
    },
    "edgeiiot": {
        "label": "Attack_type", "binary": "Attack_label", "benign": "Normal",
        "src": "ip.src_host", "dst": "ip.dst_host",
        "relation": ["http.request.method", "mqtt.msgtype", "tcp.dstport", "udp.port", "mbtcp.unit_id"],
        "split_basis": "class-block capture segment (endpoint-disjoint split infeasible for several classes)",
    },
    "apa_ddos": {
        "label": "Label", "binary": None, "benign": "Benign",
        "src": "ip.src", "dst": "ip.dst", "relation": ["ip.proto", "tcp.srcport", "tcp.dstport"],
        "time": "frame.time", "split_basis": "source device",
    },
}


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [c.strip() for c in frame.columns]
    return frame


def make_groups(name: str, frame: pd.DataFrame, spec: dict) -> pd.Series:
    if name == "apa_ddos":
        return frame[spec["src"]].astype(str)
    if name == "toniot":
        return frame[spec["src"]].astype(str) + ">" + frame[spec["dst"]].astype(str)
    # Edge's selected extract is concatenated by class and has only 4--10
    # endpoint pairs for several classes. Ten contiguous groups per class
    # preserve local order without allowing exact-row overlap.
    groups = pd.Series(index=frame.index, dtype="object")
    for label, idx in frame.groupby(spec["label"], sort=False).groups.items():
        ordered = np.asarray(list(idx))
        chunks = np.array_split(ordered, 10)
        for j, rows in enumerate(chunks):
            groups.loc[rows] = f"{label}|segment{j:02d}"
    return groups


def assign_folds(y: pd.Series, groups: pd.Series) -> np.ndarray:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260731)
    folds = np.full(len(y), -1, dtype=np.int8)
    dummy = np.zeros((len(y), 1), dtype=np.float32)
    for fold, (_, idx) in enumerate(sgkf.split(dummy, y.astype(str), groups.astype(str))):
        folds[idx] = fold
    if (folds < 0).any():
        raise RuntimeError("unassigned rows")
    return folds


def main() -> None:
    manifest_dir = ARTIFACTS / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    checksums = {}
    for name, path in PRIMARY_FILES.items():
        frame = normalize_columns(pd.read_csv(path, low_memory=False))
        spec = SPECS[name]
        frame.insert(0, "original_row", np.arange(len(frame), dtype=np.int64))
        label_cols = [spec["label"]] + ([spec["binary"]] if spec.get("binary") else [])
        feature_cols = [c for c in frame.columns if c not in label_cols + ["original_row"]]
        # Conflicting duplicates were audited as zero. Drop identical complete
        # rows deterministically, retaining the first occurrence.
        duplicate = frame.duplicated(subset=feature_cols + label_cols, keep="first")
        clean = frame.loc[~duplicate].copy()
        groups = make_groups(name, clean, spec)
        folds = assign_folds(clean[spec["label"]], groups)
        split = np.where(folds == 0, "test", np.where(folds == 1, "validation", "train"))
        row_hash = pd.util.hash_pandas_object(clean[feature_cols], index=False).astype(str)
        manifest = pd.DataFrame({
            "original_row": clean["original_row"].to_numpy(),
            "row_hash": row_hash.to_numpy(),
            "group_id": groups.to_numpy(),
            "fold": folds,
            "split": split,
            "multiclass_label": clean[spec["label"]].astype(str).to_numpy(),
            "binary_label": (clean[spec["label"]].astype(str) != spec["benign"]).astype(int).to_numpy(),
        })
        out = manifest_dir / f"{name}_split_manifest.csv"
        manifest.to_csv(out, index=False)
        checksums[name] = hashlib.sha256(out.read_bytes()).hexdigest()
        for part in ["train", "validation", "test"]:
            sub = manifest[manifest["split"] == part]
            counts = sub["multiclass_label"].value_counts().to_dict()
            summaries.append({
                "dataset": name, "split": part, "rows": len(sub),
                "groups": sub["group_id"].nunique(), "class_counts": json.dumps(counts, sort_keys=True),
                "removed_exact_duplicates": int(duplicate.sum()), "split_basis": spec["split_basis"],
            })

    summary = pd.DataFrame(summaries)
    summary.to_csv(RESULTS / "split_summary.csv", index=False)
    protocol = {
        "status": "FROZEN_BEFORE_MODEL_EVALUATION",
        "frozen_date": "2026-07-31",
        "selection_rule": "validation median across seeds; test inaccessible to tuning scripts",
        "primary_metric": "multiclass macro-F1",
        "secondary_metrics": ["balanced_accuracy", "minority_recall", "macro_PR_AUC", "FPR", "FNR", "ECE"],
        "practical_equivalence_margin_macro_f1": 0.02,
        "seeds": [1103, 2207, 3313, 4421, 5527],
        "split_assignment": "five stratified group folds; fold 0 test, fold 1 validation, folds 2-4 train",
        "preprocessing": "fit imputation, scaling, categorical encoding, graph-frequency maps, feature selection, calibration, and thresholds on train/validation only",
        "duplicate_policy": "drop exact duplicates before splitting; reject conflicting-label duplicates",
        "test_gate": "src/final_evaluation.py requires this file and consumes frozen candidate/hyperparameter record",
        "manifest_sha256": checksums,
        "dataset_specs": SPECS,
    }
    pdir = ARTIFACTS / "protocol"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "frozen_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    lines = ["# Leakage-Safe Split Protocol", "",
             "The protocol was frozen before any candidate or benchmark test evaluation. Fold 0 is held out as test, fold 1 is validation, and folds 2--4 are training.", "",
             "| Dataset | Split | Rows | Groups | Exact duplicates removed | Grouping basis |",
             "|---|---|---:|---:|---:|---|"]
    for row in summaries:
        lines.append(f"| {row['dataset']} | {row['split']} | {row['rows']:,} | {row['groups']:,} | {row['removed_exact_duplicates']:,} | {row['split_basis']} |")
    lines += ["", "All imputers, encoders, scalers, graph-frequency maps, feature selectors, probability calibrators, decision thresholds, e-value betting fractions, uncertainty radii, and policies are fit without test labels. Edge-IIoT's selected extract cannot support endpoint-disjoint multiclass folds for all labels because several attacks occupy only 4--10 endpoint pairs; its capture-segment split is therefore a declared limitation rather than evidence of cross-topology generalization."]
    (REPORTS / "LEAKAGE_SAFE_PROTOCOL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
