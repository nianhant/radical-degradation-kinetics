from __future__ import annotations

from pathlib import Path

from radical_degradation.benchmark import benchmark_ml_potentials
from radical_degradation.preopt import embed_and_optimize_smiles


def main() -> None:
    smiles = "CC(=O)NC1=CC=C(C=C1)O"
    mol = embed_and_optimize_smiles(smiles)

    benchmark_df = benchmark_ml_potentials(
        mol,
        backends=["uma", "mace", "orb"],
        calculator_kwargs={
            "uma": {"model_name": "uma-s-1p1"},
            "mace": {"model": "medium", "device": "cpu"},
            "orb": {"model": "orb-v2", "device": "cpu"},
        },
    )

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ml_potential_benchmark.csv"
    benchmark_df.to_csv(out_path, index=False)
    print(benchmark_df.to_string(index=False))
    print(f"\nSaved benchmark table to {out_path}")


if __name__ == "__main__":
    main()
