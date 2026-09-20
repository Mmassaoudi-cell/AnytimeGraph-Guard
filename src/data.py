"""Leakage-safe loading, feature encoding, and graph-context construction."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import ARTIFACTS, PRIMARY_FILES
from .prepare_protocol import SPECS


@dataclass
class DataBundle:
    name: str
    X_base: dict[str, np.ndarray]
    X_graph: dict[str, np.ndarray]
    X_all: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    y_binary: dict[str, np.ndarray]
    relation: dict[str, np.ndarray]
    groups: dict[str, np.ndarray]
    row_ids: dict[str, np.ndarray]
    classes: list[str]
    feature_names_base: list[str]
    feature_names_graph: list[str]


def _safe_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col in frame:
        return frame[col].fillna("<NA>").astype(str).str.strip()
    return pd.Series("<NA>", index=frame.index)


def _fit_graph_maps(train: pd.DataFrame, spec: dict) -> dict:
    src = _safe_series(train, spec["src"])
    dst = _safe_series(train, spec["dst"])
    pair = src + ">" + dst
    rel_cols = [c for c in spec["relation"] if c in train]
    rel = _safe_series(train, rel_cols[0]) if rel_cols else pd.Series("default", index=train.index)
    return {
        "src_count": src.value_counts().to_dict(),
        "dst_count": dst.value_counts().to_dict(),
        "pair_count": pair.value_counts().to_dict(),
        "rel_count": rel.value_counts().to_dict(),
        "src_unique_dst": pd.DataFrame({"s": src, "d": dst}).groupby("s")["d"].nunique().to_dict(),
        "dst_unique_src": pd.DataFrame({"s": src, "d": dst}).groupby("d")["s"].nunique().to_dict(),
        "relations": {v: i + 1 for i, v in enumerate(sorted(rel.unique()))},
        "relation_col": rel_cols[0] if rel_cols else None,
    }


def _graph_matrix(frame: pd.DataFrame, spec: dict, maps: dict) -> tuple[np.ndarray, np.ndarray]:
    src = _safe_series(frame, spec["src"])
    dst = _safe_series(frame, spec["dst"])
    pair = src + ">" + dst
    rel = _safe_series(frame, maps["relation_col"]) if maps["relation_col"] else pd.Series("default", index=frame.index)
    cols = [
        src.map(maps["src_count"]).fillna(0), dst.map(maps["dst_count"]).fillna(0),
        pair.map(maps["pair_count"]).fillna(0), rel.map(maps["rel_count"]).fillna(0),
        src.map(maps["src_unique_dst"]).fillna(0), dst.map(maps["dst_unique_src"]).fillna(0),
    ]
    matrix = np.log1p(np.column_stack([c.to_numpy(float) for c in cols])).astype(np.float32)
    relation = rel.map(maps["relations"]).fillna(0).to_numpy(np.int64)
    return matrix, relation


def _fit_base_encoder(train: pd.DataFrame, excluded: set[str]) -> dict:
    numeric_cols, cat_cols = [], []
    for col in train.columns:
        if col in excluded:
            continue
        numeric = pd.to_numeric(train[col], errors="coerce")
        valid = float(numeric.notna().mean())
        if valid >= 0.95 and numeric.nunique(dropna=True) > 1:
            numeric_cols.append(col)
        else:
            nunique = train[col].fillna("<NA>").astype(str).nunique()
            if 1 < nunique <= 128:
                cat_cols.append(col)
    # Deterministic variance cap prevents payload-heavy datasets dominating.
    if len(numeric_cols) > 64:
        variances = {c: pd.to_numeric(train[c], errors="coerce").var() for c in numeric_cols}
        numeric_cols = sorted(numeric_cols, key=lambda c: (-float(variances[c] or 0), c))[:64]
    medians, iqrs = {}, {}
    for col in numeric_cols:
        s = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        medians[col] = float(s.median()) if s.notna().any() else 0.0
        q1, q3 = s.quantile([0.25, 0.75])
        iqrs[col] = float(q3 - q1) if np.isfinite(q3 - q1) and q3 > q1 else 1.0
    cat_freq = {c: train[c].fillna("<NA>").astype(str).str.strip().value_counts(normalize=True).to_dict() for c in cat_cols}
    return {"numeric_cols": numeric_cols, "cat_cols": cat_cols, "medians": medians, "iqrs": iqrs, "cat_freq": cat_freq}


def _base_matrix(frame: pd.DataFrame, enc: dict) -> np.ndarray:
    cols = []
    for col in enc["numeric_cols"]:
        s = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(enc["medians"][col])
        cols.append(((s.to_numpy(float) - enc["medians"][col]) / enc["iqrs"][col]).clip(-20, 20))
    for col in enc["cat_cols"]:
        s = frame[col].fillna("<NA>").astype(str).str.strip().map(enc["cat_freq"][col]).fillna(0)
        cols.append(np.log1p(s.to_numpy(float) * max(1, len(enc["cat_freq"][col]))))
    return np.column_stack(cols).astype(np.float32) if cols else np.empty((len(frame), 0), np.float32)


def load_bundle(name: str) -> DataBundle:
    path = PRIMARY_FILES[name]
    spec = SPECS[name]
    full = pd.read_csv(path, low_memory=False)
    full.columns = [c.strip() for c in full.columns]
    manifest = pd.read_csv(ARTIFACTS / "manifests" / f"{name}_split_manifest.csv")
    parts = {}
    part_manifest = {}
    for split in ["train", "validation", "test"]:
        m = manifest[manifest.split == split].copy()
        parts[split] = full.iloc[m.original_row.to_numpy()].reset_index(drop=True)
        part_manifest[split] = m.reset_index(drop=True)

    train = parts["train"]
    graph_maps = _fit_graph_maps(train, spec)
    excluded = {spec["label"], spec.get("binary"), spec["src"], spec["dst"], spec.get("time"), "frame.time", "no", "id"}
    excluded.discard(None)
    base_enc = _fit_base_encoder(train, excluded)
    label_enc = LabelEncoder().fit(train[spec["label"]].astype(str))

    X_base, X_graph, X_all, ys, yb, relations, groups, rows = {}, {}, {}, {}, {}, {}, {}, {}
    for split, frame in parts.items():
        xb = _base_matrix(frame, base_enc)
        xg, rel = _graph_matrix(frame, spec, graph_maps)
        X_base[split], X_graph[split] = xb, xg
        X_all[split] = np.column_stack([xb, xg]).astype(np.float32)
        ys[split] = label_enc.transform(frame[spec["label"]].astype(str))
        yb[split] = (frame[spec["label"]].astype(str) != spec["benign"]).to_numpy(np.int8)
        relations[split] = rel
        groups[split] = part_manifest[split].group_id.astype(str).to_numpy()
        rows[split] = part_manifest[split].original_row.to_numpy(np.int64)

    prep = {
        "dataset": name, "source": str(path), "classes": label_enc.classes_.tolist(),
        "base_encoder": base_enc,
        "graph_maps": {k: v for k, v in graph_maps.items() if k not in {"src_count", "dst_count", "pair_count", "src_unique_dst", "dst_unique_src"}},
        "graph_features": ["log_src_count", "log_dst_count", "log_pair_count", "log_relation_count", "log_src_unique_dst", "log_dst_unique_src"],
        "fit_scope": "training rows only",
    }
    out_dir = ARTIFACTS / "preprocessing"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_parameters.json").write_text(json.dumps(prep, indent=2, default=str), encoding="utf-8")
    return DataBundle(
        name, X_base, X_graph, X_all, ys, yb, relations, groups, rows,
        label_enc.classes_.tolist(), base_enc["numeric_cols"] + [f"freq:{c}" for c in base_enc["cat_cols"]],
        prep["graph_features"],
    )

