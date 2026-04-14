# Radical Degradation Kinetics

Compact repository for studying the degradation of prominent pharmaceutical molecules in wastewater by radical species generated under oxidative electrochemical conditions.

The computational goal is to generate intrinsic molecular descriptors, especially bond dissociation energies (BDEs) and frontier orbital properties such as HOMO energies, and relate them to experimentally observed degradation kinetics.

The repository includes:

- molecule retrieval and standardization with RDKit
- homolytic bond fragmentation with ALFABET or an RDKit fallback
- 3D conformer generation, force-field pre-optimization, and optional ML pre-relaxation
- ORCA input generation and Slurm submission helpers for DFT
- ML potential benchmarking across UMA, MACE, and ORB-style backends
- a lightweight demo pipeline
- simple tests for core workflow utilities

## Why this repo exists

This repository is designed as a compact, readable example for a job application. It emphasizes:

- practical cheminformatics with RDKit
- workflow engineering for quantum chemistry
- clean, modular Python code
- simple testing and reproducibility

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

Run the example pipeline:

```bash
python demo/demo_pipeline.py
```

Example ML potential benchmark:

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
- creates a small molecule table ready for downstream processing

### `fragments.py`

- wraps ALFABET prediction if available
- provides an RDKit fallback that enumerates single-bond homolytic cleavage candidates
- returns a fragment table that can be used for BDE-oriented analysis

### `preopt.py`

- builds 3D conformers with ETKDG
- performs quick UFF or MMFF pre-optimization
- optionally pre-relaxes structures with ML interatomic potentials such as UMA, MACE, or ORB
- exports XYZ coordinates for DFT

### `benchmark.py`

- compares optional ML calculators on the same geometry
- records energy, max-force, timing, and status
- helps decide which pre-relaxation backend is most practical before DFT

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

## Testing

```bash
pytest
```

The tests intentionally cover only lightweight logic so they can run quickly in a clean environment.

## Notes

- The ALFABET and PubChem integrations are optional by design.
- The UMA, MACE, and ORB integrations are optional by design.
- The fallback fragment enumeration is not a replacement for a production BDE model; it exists to keep the repository self-contained.
- The DFT helper targets ORCA because it is easy to demonstrate in a small example repository.
