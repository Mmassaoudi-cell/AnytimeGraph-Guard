import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("ANYTIMEGRAPH_DATA_ROOT", ROOT / "data")).expanduser().resolve()
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"

PRIMARY_FILES = {
    "toniot": DATA_ROOT / "TONIoT Network Dataset" / "train_test_network.csv",
    "edgeiiot": DATA_ROOT / "Edge-IIoTset" / "Edge-IIoTset dataset" /
        "Selected dataset for ML and DL" / "ML-EdgeIIoT-dataset.csv",
    "apa_ddos": DATA_ROOT / "APA-DDoS-Dataset" / "APA-DDoS-Dataset.csv",
}

SEEDS = [1103, 2207, 3313, 4421, 5527]

for path in (ARTIFACTS, REPORTS, RESULTS, FIGURES, TABLES):
    path.mkdir(parents=True, exist_ok=True)
