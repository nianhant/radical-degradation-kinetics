from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radical_degradation.modeling import (
    train_bootstrap_ridge,
    train_random_forest_if_available,
)


FEATURE_COLUMNS = [
    "min_bde_kcal_mol",
    "homo_ev",
    "oh_addition_sites",
    "h_abstraction_sites",
    "aromatic_atom_fraction",
    "hetero_atom_fraction",
    "pka_nearest_neutral",
    "logp",
]


def main() -> None:
    here = Path(__file__).resolve().parent
    output_dir = here / "output" / "predictive_modeling"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(here / "degradation_descriptor_training_data.csv")
    result = train_bootstrap_ridge(
        data,
        feature_columns=FEATURE_COLUMNS,
        target_column="observed_log_k_deg",
        n_bootstrap=300,
    )

    pd.DataFrame([result.metrics]).to_csv(output_dir / "bootstrap_ridge_metrics.csv", index=False)
    result.predictions.to_csv(output_dir / "bootstrap_ridge_predictions.csv", index=False)
    result.feature_importance.to_csv(output_dir / "bootstrap_ridge_feature_importance.csv", index=False)
    result.decision_table.to_csv(output_dir / "decision_table.csv", index=False)

    try:
        forest_result = train_random_forest_if_available(
            data,
            feature_columns=FEATURE_COLUMNS,
            target_column="observed_log_k_deg",
        )
    except ImportError:
        forest_result = None

    if forest_result is not None:
        pd.DataFrame([forest_result["metrics"]]).to_csv(
            output_dir / "random_forest_metrics.csv", index=False
        )
        forest_result["feature_importance"].to_csv(
            output_dir / "random_forest_feature_importance.csv", index=False
        )

    print(f"Wrote predictive modeling artifacts to {output_dir}")
    print(pd.DataFrame([result.metrics]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
