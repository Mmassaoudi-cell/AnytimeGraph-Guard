# AnytimeGraph-RL: Reproducibility Package

This source-only repository contains the audit, selection, benchmarking, and
diagnostic code for a leakage-controlled graph intrusion-detection study.
Ten predeclared research directions were assessed for feasibility. Three were
implemented fully, five received partial mechanism tests, and two were rejected
because the local corpora did not support their assumptions. AnytimeGraph-RL
was selected using validation data only.

The final conclusion is deliberately conservative: the selected method is the
strongest validated candidate, not a universal state-of-the-art detector. It
has a clear sequential false-alarm advantage over fixed-threshold and
fixed-window rules, but it does not significantly beat the strongest boosted
tree on the two informative batch datasets.

## Repository scope

The repository intentionally does not track datasets, manuscript files,
figures, tables, reports, generated results, PDFs, images, or office documents.
Those files are excluded by `.gitignore`. It does include:

- all Python source under `src/`;
- the invariant tests under `tests/`;
- environment definitions;
- frozen split manifests and protocol records under `artifacts/`.

## Requirements

- Windows or Linux with Python 3.11+
- The ToN-IoT network, Edge-IIoT ML, and APA-DDoS CSV datasets. Dataset files
  are not redistributed by this repository.

Create the environment with either:

    conda env create -f environment.yml
    conda activate anytimegraph-rl

or:

    python -m pip install -r requirements.txt

Set the dataset root before running the pipeline. The default is the untracked
`data/` directory in the repository root.

PowerShell:

    $env:ANYTIMEGRAPH_DATA_ROOT = "D:\path\to\datasets"

Bash:

    export ANYTIMEGRAPH_DATA_ROOT=/path/to/datasets

The expected relative file layout is defined in `src/config.py`.

## Reproduction order

Run from the repository root:

    python -m src.dataset_audit
    python -m src.prepare_protocol
    python -m src.candidate_screening
    python -m src.tune_candidates
    python -m src.final_evaluation
    python -m src.graph_baselines
    python -m src.statistical_analysis
    python -m src.sequential_evaluation
    python -m src.secondary_experiments
    python -m src.enhancement_study
    python -m src.reviewer_evidence
    python -m src.make_figures
    python -m src.make_tables
    python -m unittest discover -s tests -v

The final benchmark is the slowest stage and took roughly 29 minutes on the
development machine. Scripts use fixed seeds and write generated outputs under
`artifacts`, `results`, `figures`, `tables`, and `reports`; generated outputs
remain local and are not committed. The final-test script is fail-closed: it
requires both the frozen protocol and frozen model-selection records.

## Integrity notes

- Exact duplicates are removed before splitting.
- ToN-IoT uses source-destination pair-disjoint partitions.
- APA-DDoS uses source-device-disjoint partitions.
- Edge-IIoT uses class-aware contiguous blocks because endpoint-disjoint class
  coverage is not possible in the available ML extract; this limitation is
  carried into every claim.
- APA-DDoS is a saturated control and is not used as novelty evidence.
