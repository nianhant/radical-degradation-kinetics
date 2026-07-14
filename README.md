# Radical Degradation Kinetics

This repository collects the pieces of a computational workflow for studying radical-driven degradation of pharmaceuticals in wastewater.

The specific use case behind it is oxidative electrochemical treatment, where reactive radical species can attack drug-like molecules through hydrogen abstraction, addition, or other bond-breaking pathways. The main descriptors of interest here are bond dissociation energies (BDEs) and frontier orbital properties such as HOMO energies, with the longer-term goal of relating those intrinsic properties to observed degradation kinetics.

At the moment the repository is set up to handle:

- molecule retrieval and standardization with RDKit
- homolytic bond fragmentation with ALFABET or an RDKit fallback
- 3D conformer generation, force-field pre-optimization, and optional ML pre-relaxation
- ORCA input generation and Slurm submission helpers for DFT
- ORCA/UMA NEB-TS endpoint and input generation for H abstraction and OH addition
- ML potential benchmarking across UMA, MACE, and ORB-style backends
- uncertainty-aware predictive modeling for degradation kinetics from computed and cheminformatics descriptors
- interpretable feature rankings and decision tables for follow-up experiments or higher-fidelity calculations
- a lightweight demo pipeline
- simple tests for core workflow utilities

## Project focus

- fragment generation for BDE calculations
- structure generation before DFT
- optional ML pre-relaxation before DFT
- UMA NEB-TS setup for radical H abstraction and OH addition
- end-to-end predictive modeling that connects scientific descriptors to decision-ready kinetic predictions

## Repository layout

```text
radical-degradation-kinetics/
├── README.md
├── pyproject.toml
├── .gitignore
├── demo/
│   ├── benchmark_ml_potentials.py
│   ├── demo_pipeline.py
│   └── sample_molecules.csv
├── src/radical_degradation/
│   ├── __init__.py
│   ├── benchmark.py
│   ├── fetch.py
│   ├── fragments.py
│   ├── modeling.py
│   ├── neb.py
│   ├── preopt.py
│   └── dft.py
├── scripts/
│   └── prepare_uma_neb_jobs.py
└── tests/
    └── test_pipeline.py
```

## Installation

```bash
cd radical-degradation-kinetics
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Optional extras:

- `pubchempy` for name-to-SMILES lookup from PubChem
- `alfabet` for model-based bond dissociation fragmentation
- `ase` for optional ML pre-relaxation workflows
- `fairchem`, `mace`, and `orb-models` for optional UMA/MACE/ORB backends
- `scikit-learn` for optional random-forest benchmarking in the predictive modeling demo

## Demo

Run the example pipeline on a small set of wastewater-relevant compounds:

```bash
python demo/demo_pipeline.py
```

Run the ML potential benchmark:

```bash
python demo/benchmark_ml_potentials.py
```

Run the predictive modeling demo:

```bash
python demo/predictive_modeling_demo.py
```

This writes model artifacts to `demo/output/predictive_modeling/`, including:

- `bootstrap_ridge_metrics.csv` with RMSE, MAE, R2, 90% interval coverage, and mean uncertainty
- `bootstrap_ridge_predictions.csv` with held-out predictions and uncertainty intervals
- `bootstrap_ridge_feature_importance.csv` with interpretable descriptor coefficients
- `decision_table.csv` ranking compounds where uncertainty or error suggests confirmatory experiments

Run the end-to-end study demo from raw compound inputs through cleaning, modeling,
synthetic benchmarking, and next-measurement ranking:

```bash
python demo/end_to_end_study_demo.py
```

Add `--query-pubchem` to resolve missing SMILES from molecule names when `pubchempy`,
RDKit, and network access are available:

```bash
python demo/end_to_end_study_demo.py --query-pubchem
```

This writes numbered artifacts to `demo/output/end_to_end_study/`:

- `01_clean_compound_inputs.csv` records input cleanup and optional PubChem status.
- `02_clean_descriptor_table.csv` stores the cleaned descriptor/outcome table.
- `03_real_small_data_metrics.csv` through `05_real_small_data_decision_table.csv` analyze the small curated dataset honestly.
- `06_synthetic_known_regime_rates.csv` contains clearly labeled synthetic rates from known equations.
- `07_synthetic_regime_recovery.csv` tests whether the workflow recovers the intended dominant descriptors.
- `08_active_learning_candidates.csv` ranks compounds for the next measurement using uncertainty and descriptor diversity.

Prepare UMA NEB-TS jobs for pharmaceutical H abstraction and OH addition:

```bash
python scripts/prepare_uma_neb_jobs.py demo/sample_molecules.csv \
  --output-dir demo/output/uma_neb_jobs \
  --uma-wrapper /full/path/to/oet_uma \
  --ext-params "--model uma-s-1p1"
```

The script writes one folder per molecule, reaction family, and site. Each job contains
`reactant.xyz`, `product.xyz`, `neb.inp`, and `submit.sh`. The reactant and product
XYZ files preserve atom order and atom count for ORCA NEB. For H abstraction, the
endpoint is `drug-H + OH radical -> drug radical + H2O`. For OH addition, the endpoint
is `drug + OH radical -> drug-OH radical adduct`.

This writes demo outputs to:

```text
demo/output/
```

## What each module does

### `fetch.py`

- standardizes SMILES with RDKit
- optionally resolves molecule names to isomeric SMILES using PubChem
- builds a small molecule table for downstream calculations

### `fragments.py`

- wraps ALFABET prediction if available
- returns a fragment table for BDE-oriented analysis

### `preopt.py`

- builds 3D conformers with ETKDG
- performs quick UFF or MMFF pre-optimization
- optionally pre-relaxes structures with ML interatomic potentials such as UMA, MACE, or ORB
- exports XYZ coordinates for DFT

### `benchmark.py`

- compares optional ML calculators on the same geometry
- records energy, max-force, timing, and status

### `modeling.py`

- validates descriptor/outcome tables for predictive modeling
- trains a dependency-light bootstrap ridge ensemble for uncertainty-aware regression
- optionally benchmarks scikit-learn random forests when `.[predictive]` is installed
- emits interpretable feature rankings and decision tables for model-guided follow-up

### `study.py`

- loads and cleans compound input tables
- optionally resolves missing structures from PubChem when dependencies and network access are available
- generates explicitly labeled synthetic benchmark rates from known reaction-regime equations
- ranks active-learning candidates by uncertainty and descriptor-space diversity

### `neb.py`

- enumerates default pharmaceutical H abstraction and OH addition sites
- builds atom-order-preserving reactant/product XYZ endpoints
- generates ORCA `ExtOpt NEB-TS` inputs using a UMA external-method wrapper

### `dft.py`

- generates ORCA input files from XYZ structures
- writes a portable Slurm submission script
- supports a simple folder-per-molecule job layout for descriptor calculations such as BDE and HOMO

## Example workflow

1. Create or load a list of wastewater-relevant pharmaceuticals.
2. Standardize molecules with RDKit.
3. Predict or enumerate bond-breaking sites relevant to radical degradation.
4. Generate 3D geometries and pre-optimize them.
5. Optionally pre-relax structures with UMA, MACE, or ORB and benchmark ML potential behavior.
6. Write ORCA inputs and submit DFT jobs for descriptors such as BDE and HOMO.
7. Generate UMA NEB-TS jobs for selected H abstraction and OH addition pathways.
8. Train and evaluate predictive models against observed degradation kinetics.
9. Use prediction uncertainty and feature importance to prioritize experiments, DFT refinement, or NEB calculations.

## Interview relevance

This project is structured as a compact analog of an AI-enabled pharmaceutical predictive platform:

- **Data ingestion and feature engineering:** molecule tables, computed chemistry descriptors, radical-site counts, and assay-style outcomes.
- **Model training and benchmarking:** transparent bootstrap ridge regression plus optional random-forest comparison.
- **Uncertainty and robustness:** bootstrap prediction intervals, held-out metrics, and coverage estimates for data-sparse settings.
- **Interpretability:** descriptor-level feature rankings that can be mapped back to mechanistic chemistry.
- **Decision support:** ranked follow-up tables that translate model uncertainty into experimental or simulation next steps.

## Continuing the study without large experimental data

The recommended structure is to keep real and synthetic evidence separate:

1. **Small-data analysis:** use any real or curated degradation rates to ask which descriptors are most consistent with the observed trends, with leave-one-compound-out style validation, bootstrap intervals, sensitivity analysis, and cautious mechanistic interpretation.
2. **Synthetic benchmark:** generate rates from known equations for electron-transfer, hydrogen-abstraction, and hydroxyl-addition regimes. These data test preprocessing, model comparison, feature recovery, missing-data robustness, and uncertainty estimation.
3. **Active learning:** use model uncertainty and descriptor diversity to rank the next compounds or calculations that would most improve the model.

Synthetic rate constants in this repository are generated only to test the workflow.
They are not experimental measurements and should not be interpreted as chemical
predictions.

## Current scope and limitations

<!-- - The fallback RDKit fragmentation code is only a lightweight stand-in for a production BDE workflow. -->
- The demo descriptor table is intentionally small and public-facing; production use would replace it with curated experimental, clinical, or internally generated datasets.
- Synthetic rates are explicitly labeled and are suitable only for pipeline validation, not scientific conclusions about a molecule.
- The repository does not yet include automated ORCA output parsing into the predictive descriptor table.
- The ORCA helper is intentionally simple and meant to be adapted to a local cluster environment.

## Testing

```bash
pytest
```

The tests intentionally cover only lightweight logic so they can run quickly in a clean environment.

## Notes

- The ALFABET, PubChem, UMA, MACE, and ORB integrations are optional by design.
- The fallback fragmentation code is there to keep the repository runnable without a full ML stack.
- The DFT helper targets ORCA because that is the code used in the accompanying workflow.
