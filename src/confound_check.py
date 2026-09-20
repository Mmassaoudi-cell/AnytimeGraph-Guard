"""Diagnostic control: does APA-DDoS source-device identity alone predict the label?

This is not part of the frozen evaluation protocol; it is a post-hoc diagnostic
requested to test whether APA-DDoS's near-saturated multiclass performance
could be explained by device fingerprinting rather than attack-signature
generalization. It reuses exactly the mechanism the proposed method's own
graph features rely on (Eq. 1: a train-only frequency count mapped onto
held-out rows) but strips away every other feature, so any residual signal
on the disjoint validation/test devices can only come from that mechanism.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from .config import ARTIFACTS, PRIMARY_FILES, RESULTS
from .prepare_protocol import SPECS


def main() -> None:
    name = "apa_ddos"
    spec = SPECS[name]
    raw = pd.read_csv(PRIMARY_FILES[name], low_memory=False)
    raw.columns = [c.strip() for c in raw.columns]
    manifest = pd.read_csv(ARTIFACTS / "manifests" / f"{name}_split_manifest.csv")

    frames = {}
    for split in ["train", "validation", "test"]:
        m = manifest[manifest.split == split]
        frames[split] = raw.iloc[m.original_row.to_numpy()].reset_index(drop=True)

    # Train-only device frequency, exactly the mechanism behind the proposed
    # method's own log-count graph features. Devices unseen in training (true
    # for every validation/test device under the source-disjoint split) map
    # to a frequency of zero, i.e. a single constant feature value.
    device_freq = frames["train"][spec["src"]].astype(str).value_counts(normalize=True).to_dict()
    label_enc = LabelEncoder().fit(frames["train"][spec["label"]].astype(str))

    def feature(frame: pd.DataFrame) -> np.ndarray:
        device = frame[spec["src"]].astype(str)
        return np.log1p(device.map(device_freq).fillna(0.0).to_numpy())[:, None]

    x_train = feature(frames["train"])
    y_train = label_enc.transform(frames["train"][spec["label"]].astype(str))

    model = LogisticRegression(max_iter=500, class_weight="balanced").fit(x_train, y_train)
    majority = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)

    rows = []
    for split in ["train", "validation", "test"]:
        frame = frames[split]
        x = feature(frame)
        y = label_enc.transform(frame[spec["label"]].astype(str))
        pred = model.predict(x)
        pred_majority = majority.predict(x)
        n_devices = frame[spec["src"]].nunique()
        n_unseen_devices = int((~frame[spec["src"]].astype(str).isin(device_freq)).sum())
        rows.append({
            "split": split,
            "n_rows": len(y),
            "n_devices": int(n_devices),
            "rows_from_devices_unseen_in_train": n_unseen_devices,
            "device_identity_only_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "majority_class_baseline_macro_f1": float(f1_score(y, pred_majority, average="macro", zero_division=0)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "apa_device_identity_confound.csv", index=False)


if __name__ == "__main__":
    main()
