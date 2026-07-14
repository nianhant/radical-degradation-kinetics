from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radical_degradation.modeling import train_bootstrap_ridge
from radical_degradation.study import (
    SYNTHETIC_RATE_NOTICE,
    add_missingness,
    clean_descriptor_table,
    generate_synthetic_regime_data,
    load_compound_inputs,
    optional_pubchem_enrichment,
    rank_active_learning_candidates,
    run_synthetic_recovery_benchmark,
)


REAL_FEATURE_COLUMNS = [
    "min_bde_kcal_mol",
    "homo_ev",
    "oh_addition_sites",
    "h_abstraction_sites",
    "aromatic_atom_fraction",
    "hetero_atom_fraction",
    "pka_nearest_neutral",
    "logp",
]

SYNTHETIC_FEATURE_COLUMNS = [
    "min_bde_kcal_mol",
    "homo_ev",
    "oh_addition_energy_kcal_mol",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a transparent degradation-kinetics study workflow."
    )
    parser.add_argument(
        "--query-pubchem",
        action="store_true",
        help="Resolve missing SMILES through PubChem when optional dependencies/network are available.",
    )
    parser.add_argument(
        "--synthetic-per-regime",
        type=int,
        default=40,
        help="Number of synthetic compounds to generate for each known reaction regime.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    output_dir = here / "output" / "end_to_end_study"
    output_dir.mkdir(parents=True, exist_ok=True)

    compounds = load_compound_inputs(here / "study_compounds.csv")
    compounds = optional_pubchem_enrichment(
        compounds,
        query_missing_smiles=args.query_pubchem,
    )
    compounds.to_csv(output_dir / "01_clean_compound_inputs.csv", index=False)

    descriptors = clean_descriptor_table(
        pd.read_csv(here / "degradation_descriptor_training_data.csv")
    )
    descriptors.to_csv(output_dir / "02_clean_descriptor_table.csv", index=False)

    real_result = train_bootstrap_ridge(
        descriptors,
        feature_columns=REAL_FEATURE_COLUMNS,
        target_column="observed_log_k_deg",
        n_bootstrap=300,
        random_state=13,
    )
    pd.DataFrame([real_result.metrics]).to_csv(
        output_dir / "03_real_small_data_metrics.csv", index=False
    )
    real_result.feature_importance.to_csv(
        output_dir / "04_real_small_data_feature_importance.csv", index=False
    )
    real_result.decision_table.to_csv(
        output_dir / "05_real_small_data_decision_table.csv", index=False
    )

    synthetic = generate_synthetic_regime_data(
        descriptors,
        n_per_regime=args.synthetic_per_regime,
    )
    synthetic = add_missingness(
        synthetic,
        columns=SYNTHETIC_FEATURE_COLUMNS,
        missing_fraction=0.08,
    )
    synthetic.to_csv(output_dir / "06_synthetic_known_regime_rates.csv", index=False)

    synthetic_complete = synthetic.dropna(subset=SYNTHETIC_FEATURE_COLUMNS).reset_index(drop=True)
    recovery = run_synthetic_recovery_benchmark(
        synthetic_complete,
        feature_columns=SYNTHETIC_FEATURE_COLUMNS,
    )
    recovery.to_csv(output_dir / "07_synthetic_regime_recovery.csv", index=False)

    candidate_pool = descriptors.copy()
    candidate_pool["descriptor_uncertainty"] = (
        candidate_pool["min_bde_kcal_mol"].rank(pct=True, ascending=False)
        + candidate_pool["homo_ev"].rank(pct=True)
    ) / 2.0
    active_learning = rank_active_learning_candidates(
        candidate_pool,
        feature_columns=REAL_FEATURE_COLUMNS,
        uncertainty_columns=["descriptor_uncertainty"],
    )
    active_learning.to_csv(output_dir / "08_active_learning_candidates.csv", index=False)

    readme = output_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "End-to-end radical degradation modeling study artifacts",
                "",
                SYNTHETIC_RATE_NOTICE,
                "",
                "Files 01-05 use curated demonstration descriptors and observed_log_k_deg from the demo table.",
                "Files 06-07 use synthetic rates only to test recovery of known relationships.",
                "File 08 ranks candidates for the next measurement using uncertainty and descriptor diversity.",
                "",
            ]
        )
    )

    print(f"Wrote end-to-end study artifacts to {output_dir}")
    print("Synthetic data notice:")
    print(SYNTHETIC_RATE_NOTICE)
    print("\nSynthetic regime recovery:")
    print(recovery.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
