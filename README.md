# Radical Degradation Kinetics

This repository collects the pieces of a computational workflow for studying radical-driven degradation of pharmaceuticals in wastewater.

The specific use case behind it is oxidative electrochemical treatment, where reactive radical species can attack drug-like molecules through hydrogen abstraction, addition, or other bond-breaking pathways. The main descriptors of interest here are bond dissociation energies (BDEs) and frontier orbital properties such as HOMO energies, with the longer-term goal of relating those intrinsic properties to observed degradation kinetics.

At the moment the repository is set up to handle:

- molecule retrieval and standardization with RDKit
- homolytic bond fragmentation with ALFABET or an RDKit fallback
- 3D conformer generation, force-field pre-optimization, and optional ML pre-relaxation
- ORCA input generation and Slurm submission helpers for DFT
- ML potential benchmarking across UMA, MACE, and ORB-style backends
- a lightweight demo pipeline
- simple tests for core workflow utilities

## Project focus

- fragment generation for BDE calculations
- structure generation before DFT
- optional ML pre-relaxation before DFT
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
│   ├── preopt.py
│   └── dft.py
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

## Demo

Run the example pipeline on a small set of wastewater-relevant compounds:

```bash
python demo/demo_pipeline.py
```

Run the ML potential benchmark:

```bash
python demo/benchmark_ml_potentials.py
```

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
7. Correlate computed intrinsic properties with observed degradation kinetics in a downstream analysis step.

## Current scope and limitations

<!-- - The fallback RDKit fragmentation code is only a lightweight stand-in for a production BDE workflow. -->
- The repository does not yet include output parsing or the final regression analysis against kinetics.
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
