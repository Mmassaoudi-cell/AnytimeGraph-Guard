"""Streaming inventory and leakage-risk audit for every local IDS corpus.

Exact expensive diagnostics are performed for prospective primary files. Large
raw corpora receive exact file/row/schema/label counts and deterministic sampled
quality diagnostics; the scope is explicit in the output and is never presented
as a full exact duplicate audit.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import mmap
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ARTIFACTS, DATA_ROOT, PRIMARY_FILES, REPORTS, RESULTS


LABEL_HINTS = {
    "APA-DDoS-Dataset": ["Label"],
    "Bot_IoT": ["attack", "category", "subcategory"],
    "CIC-IDS- 2017": ["Label"],
    "CICIOT23": ["label"],
    "CIC_VPN2016": ["traffic_type"],
    "CS240_ISCXVPN2016": ["Label", "Category", "App_protocol", "Web_service"],
    "dataset": ["attack_cat", "label"],
    "Edge-IIoTset": ["Attack_label", "Attack_type"],
    "IDS_ISCX_2012_dataset": ["label"],
    "KDD": ["labels"],
    "RT_IOT": ["Attack_type"],
    "RT_IOT2022": ["Attack_type"],
    "TONIoT Network Dataset": ["label", "type"],
    "UNSW_NB15": ["attack_cat", "label"],
}

TIME_NAMES = {"frame.time", "timestamp", "flow_start", "stime", "ltime"}
SRC_NAMES = {"ip.src", "ip.src_host", "src_ip", "src ip", "saddr", "arp.src.proto_ipv4"}
DST_NAMES = {"ip.dst", "ip.dst_host", "dst_ip", "dst ip", "daddr", "arp.dst.proto_ipv4"}
REL_RE = re.compile(r"proto|service|port|state|method|mqtt|dns|http|modbus|mbtcp", re.I)
SESSION_RE = re.compile(r"flow.?id|session|capture|pkseqid|^id$", re.I)


def count_rows(path: Path) -> int:
    """Count data rows exactly using a memory map."""
    if path.stat().st_size == 0:
        return 0
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        lines = 0
        while mm.readline():
            lines += 1
    return max(0, lines - 1)


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return [x.strip() for x in next(csv.reader(handle))]


def file_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def label_counts(paths: Iterable[Path], columns: list[str]) -> dict[str, dict[str, int]]:
    out: dict[str, Counter] = {c: Counter() for c in columns}
    for path in paths:
        available = set(read_header(path))
        use = [c for c in columns if c in available]
        if not use:
            continue
        try:
            for chunk in pd.read_csv(
                path, usecols=lambda raw: raw.strip() in use,
                chunksize=500_000, low_memory=False
            ):
                chunk.columns = [c.strip() for c in chunk.columns]
                for col in use:
                    out[col].update(chunk[col].fillna("<NA>").astype(str).str.strip().tolist())
        except Exception as exc:
            out.setdefault("_errors", Counter())[f"{path.name}: {type(exc).__name__}: {exc}"] += 1
    return {k: dict(v) for k, v in out.items()}


def sampled_quality(paths: list[Path], cap_per_file: int = 20_000) -> dict:
    frames = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path, nrows=cap_per_file, low_memory=False))
        except Exception:
            continue
    if not frames:
        return {"scope": "unavailable"}
    frame = pd.concat(frames, ignore_index=True, sort=False)
    numeric = frame.select_dtypes(include=[np.number])
    missing = int(frame.isna().sum().sum())
    inf = int(np.isinf(numeric.to_numpy(dtype=float, copy=False)).sum()) if len(numeric.columns) else 0
    return {
        "scope": f"deterministic first {cap_per_file} rows per CSV",
        "sample_rows": int(len(frame)),
        "missing_cells": missing,
        "infinite_cells": inf,
        "exact_duplicate_rows": int(frame.duplicated().sum()),
    }


def exact_primary_quality(path: Path, label_cols: list[str]) -> dict:
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [c.strip() for c in frame.columns]
    numeric = frame.select_dtypes(include=[np.number])
    feature_cols = [c for c in frame.columns if c not in label_cols]
    exact_dups = int(frame.duplicated().sum())
    feature_hash = pd.util.hash_pandas_object(frame[feature_cols], index=False)
    labels = frame[label_cols].astype(str).agg("|".join, axis=1) if label_cols else pd.Series("", index=frame.index)
    conflict_frame = pd.DataFrame({"h": feature_hash, "y": labels})
    conflicts = int((conflict_frame.groupby("h", sort=False)["y"].nunique() > 1).sum())
    rounded = frame[feature_cols].copy()
    for col in rounded.select_dtypes(include=[np.number]).columns:
        rounded[col] = pd.to_numeric(rounded[col], errors="coerce").round(4)
    near_dups = int(pd.util.hash_pandas_object(rounded, index=False).duplicated().sum())
    return {
        "scope": "exact full selected file",
        "rows": int(len(frame)),
        "missing_cells": int(frame.isna().sum().sum()),
        "infinite_cells": int(np.isinf(numeric.to_numpy(dtype=float, copy=False)).sum()) if len(numeric.columns) else 0,
        "exact_duplicate_rows": exact_dups,
        "conflicting_label_feature_groups": conflicts,
        "near_duplicate_rows_round4": near_dups,
    }


def feasibility(columns: list[str]) -> dict:
    times = [c for c in columns if c.strip().lower() in TIME_NAMES]
    src = [c for c in columns if c.strip().lower() in SRC_NAMES]
    dst = [c for c in columns if c.strip().lower() in DST_NAMES]
    rel = [c for c in columns if REL_RE.search(c)]
    sessions = [c for c in columns if SESSION_RE.search(c)]
    graph = bool(src and dst)
    temporal = bool(times)
    domain = graph or bool(sessions)
    return {
        "timestamp_fields": times,
        "source_fields": src,
        "destination_fields": dst,
        "relation_fields": rel,
        "session_fields": sessions,
        "graph_feasible": graph,
        "temporal_feasible": temporal,
        "open_world_feasible": bool(rel),
        "domain_environment_feasible": domain,
        "sparse_stream_feasible": graph and temporal,
        "partial_observability_feasible": graph,
        "condensation_feasible": graph,
        "hardware_proxy_feasible": True,
        "equation_discovery_feasible": False,
        "leakage_safe_group_split_feasible": graph or bool(sessions),
    }


def main() -> None:
    csv_files = sorted(DATA_ROOT.rglob("*.csv"))
    by_dataset: dict[str, list[Path]] = defaultdict(list)
    for path in csv_files:
        by_dataset[path.relative_to(DATA_ROOT).parts[0]].append(path)

    records = []
    duplicate_files: dict[str, list[str]] = defaultdict(list)
    for dataset, paths in sorted(by_dataset.items()):
        schemas = defaultdict(list)
        rows_by_file = {}
        hashes = {}
        for path in paths:
            header = read_header(path)
            schemas[tuple(header)].append(path)
            rows_by_file[str(path)] = count_rows(path)
            # Full-file hashes are practical for the small duplicated corpora only.
            if path.stat().st_size < 100 * 1024 * 1024:
                hashes[str(path)] = file_fingerprint(path)
                duplicate_files[hashes[str(path)]].append(str(path))
        all_cols = sorted(set().union(*(set(s) for s in schemas)))
        labels = [c for c in LABEL_HINTS.get(dataset, []) if c in all_cols]
        exact_path = next((p for p in PRIMARY_FILES.values() if p in paths), None)
        quality = exact_primary_quality(exact_path, labels) if exact_path else sampled_quality(paths)
        f = feasibility(all_cols)
        total_rows = int(sum(rows_by_file.values()))
        rec = {
            "dataset": dataset,
            "full_paths": [str(p) for p in paths],
            "formats": sorted({p.suffix.lower() for p in DATA_ROOT.joinpath(dataset).rglob("*") if p.is_file()}),
            "file_count_all_formats": sum(1 for p in DATA_ROOT.joinpath(dataset).rglob("*") if p.is_file()),
            "csv_file_count": len(paths),
            "total_size_bytes": sum(p.stat().st_size for p in DATA_ROOT.joinpath(dataset).rglob("*") if p.is_file()),
            "rows": total_rows,
            "predictor_count_range": [min(len(s) for s in schemas) - len(labels), max(len(s) for s in schemas) - len(labels)],
            "schema_count": len(schemas),
            "label_columns": labels,
            "label_counts": label_counts(paths, labels),
            "quality": quality,
            "row_counts_by_file": rows_by_file,
            **f,
        }
        # Saturation risk is a documented risk flag, not a performance result.
        rec["saturation_risk"] = "high" if dataset in {"KDD", "APA-DDoS-Dataset", "CIC-IDS- 2017"} else "moderate"
        rec["computational_requirement"] = "high" if rec["total_size_bytes"] > 2 * 1024**3 else "moderate"
        records.append(rec)

    duplicates = [v for v in duplicate_files.values() if len(v) > 1]
    out = {
        "data_root": str(DATA_ROOT),
        "audit_scope": {
            "exact": "file inventory, byte sizes, CSV rows, schemas, labels; full quality audit for prospective primary files",
            "sampled": "missing/infinite/duplicate diagnostics for non-primary large raw corpora",
        },
        "datasets": records,
        "byte_identical_small_files": duplicates,
    }
    ARTIFACTS.joinpath("audit").mkdir(parents=True, exist_ok=True)
    with ARTIFACTS.joinpath("audit", "dataset_inventory.json").open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, ensure_ascii=False)

    flat = []
    for r in records:
        flat.append({
            "dataset": r["dataset"], "files": r["file_count_all_formats"],
            "csv_files": r["csv_file_count"], "size_gb": r["total_size_bytes"] / 1024**3,
            "rows": r["rows"], "predictors_min": r["predictor_count_range"][0],
            "predictors_max": r["predictor_count_range"][1],
            "labels": ";".join(r["label_columns"]), "graph": r["graph_feasible"],
            "temporal": r["temporal_feasible"], "group_split": r["leakage_safe_group_split_feasible"],
            "quality_scope": r["quality"].get("scope"), "exact_duplicates": r["quality"].get("exact_duplicate_rows"),
            "conflicting_groups": r["quality"].get("conflicting_label_feature_groups"),
        })
    pd.DataFrame(flat).to_csv(RESULTS / "dataset_inventory.csv", index=False)

    lines = ["# Complete Dataset Screening Report", "", 
             "The inventory below is generated from the files actually present locally. Exact full-row quality diagnostics are limited to the three prospective primary files; large raw archives are explicitly marked as sampled.", ""]
    lines.append("| Dataset | Files | Size (GiB) | Rows | Labels | Graph | Time/order | Quality scope |")
    lines.append("|---|---:|---:|---:|---|:---:|:---:|---|")
    for r in records:
        lines.append(f"| {r['dataset']} | {r['file_count_all_formats']} | {r['total_size_bytes']/1024**3:.3f} | {r['rows']:,} | {', '.join(r['label_columns']) or 'none detected'} | {'yes' if r['graph_feasible'] else 'no'} | {'yes' if r['temporal_feasible'] else 'no'} | {r['quality'].get('scope','')} |")
    lines += ["", "## Byte-identical local copies", ""]
    if duplicates:
        for group in duplicates:
            lines.append("- " + " = ".join(group))
    else:
        lines.append("- None among files under 100 MiB.")
    REPORTS.joinpath("DATASET_SCREENING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
