from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .modeling import train_bootstrap_ridge


SYNTHETIC_RATE_NOTICE = (
    "Synthetic rate constants were generated to test the workflow. They are not "
    "experimental measurements and should not be interpreted as chemical predictions."
)


def load_compound_inputs(path: str | Path) -> pd.DataFrame:
    """Load a molecule input table with explicit source and structure fields."""

    frame = pd.read_csv(path)
    required_any = {"name", "smiles"}
    if not required_any.intersection(frame.columns):
        raise ValueError("Input table must include at least one of: name, smiles")

    for column in ["name", "smiles", "source"]:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str).str.strip()

    frame["compound_id"] = np.arange(1, len(frame) + 1)
    frame["has_structure"] = frame["smiles"].str.len() > 0
    frame["input_status"] = np.where(frame["has_structure"], "ready", "needs_structure")
    return frame


def optional_pubchem_enrichment(
    compounds: pd.DataFrame,
    *,
    query_missing_smiles: bool = False,
) -> pd.DataFrame:
    """Resolve missing structures from PubChem when pubchempy/RDKit are installed.

    This function is deliberately opt-in because it may require network access. Rows
    that cannot be resolved are retained and labeled rather than dropped silently.
    """

    enriched = compounds.copy()
    if "pubchem_status" not in enriched.columns:
        enriched["pubchem_status"] = "not_queried"
    if "canonical_smiles" not in enriched.columns:
        enriched["canonical_smiles"] = enriched["smiles"]

    if not query_missing_smiles:
        return enriched

    try:
        from .fetch import canonicalize_smiles, resolve_name_to_smiles
    except ImportError:
        enriched.loc[~enriched["has_structure"], "pubchem_status"] = "rdkit_or_pubchempy_missing"
        return enriched

    for idx, row in enriched.iterrows():
        if row["smiles"]:
            try:
                enriched.at[idx, "canonical_smiles"] = canonicalize_smiles(row["smiles"])
                enriched.at[idx, "pubchem_status"] = "canonicalized_input_smiles"
            except ValueError:
                enriched.at[idx, "pubchem_status"] = "invalid_input_smiles"
            continue

        if not row["name"]:
            enriched.at[idx, "pubchem_status"] = "missing_name_and_smiles"
            continue

        try:
            enriched.at[idx, "canonical_smiles"] = resolve_name_to_smiles(row["name"])
            enriched.at[idx, "pubchem_status"] = "resolved_from_pubchem"
            enriched.at[idx, "has_structure"] = True
            enriched.at[idx, "input_status"] = "ready"
        except Exception as exc:  # pragma: no cover - depends on network/API state
            enriched.at[idx, "pubchem_status"] = f"pubchem_failed: {exc}"

    return enriched


def clean_descriptor_table(
    descriptors: pd.DataFrame,
    *,
    id_column: str = "molecule",
) -> pd.DataFrame:
    """Normalize descriptor tables before modeling."""

    clean = descriptors.copy()
    if id_column not in clean.columns:
        raise ValueError(f"Descriptor table is missing id column: {id_column}")

    clean[id_column] = clean[id_column].astype(str).str.strip()
    clean = clean[clean[id_column].str.len() > 0].drop_duplicates(subset=[id_column])

    numeric_candidates = [
        col
        for col in clean.columns
        if col not in {id_column, "canonical_smiles", "source", "regime", "rate_source"}
    ]
    for col in numeric_candidates:
        converted = pd.to_numeric(clean[col], errors="coerce")
        if converted.notna().sum() == clean[col].notna().sum():
            clean[col] = converted

    return clean.reset_index(drop=True)


def generate_synthetic_regime_data(
    base_descriptors: pd.DataFrame,
    *,
    n_per_regime: int = 40,
    random_state: int = 11,
    noise_sd: float = 0.08,
) -> pd.DataFrame:
    """Create clearly labeled synthetic rates from known reaction-regime equations."""

    rng = np.random.default_rng(random_state)
    regimes = {
        "electron_transfer_controlled": {
            "intercept": -1.6,
            "homo_ev": 1.2,
            "min_bde_kcal_mol": -0.002,
            "oh_addition_energy_kcal_mol": -0.005,
        },
        "hydrogen_abstraction_controlled": {
            "intercept": 3.2,
            "homo_ev": 0.04,
            "min_bde_kcal_mol": -0.055,
            "oh_addition_energy_kcal_mol": -0.005,
        },
        "hydroxyl_addition_controlled": {
            "intercept": 1.0,
            "homo_ev": 0.03,
            "min_bde_kcal_mol": -0.01,
            "oh_addition_energy_kcal_mol": -0.075,
        },
    }

    required = ["min_bde_kcal_mol", "homo_ev"]
    missing = [col for col in required if col not in base_descriptors.columns]
    if missing:
        raise ValueError(f"Base descriptors are missing required columns: {missing}")

    rows: list[dict[str, object]] = []
    base = base_descriptors.reset_index(drop=True)
    for regime, coeffs in regimes.items():
        for i in range(n_per_regime):
            parent = base.iloc[int(rng.integers(0, len(base)))]
            min_bde = float(parent["min_bde_kcal_mol"] + rng.normal(0.0, 3.0))
            homo = float(parent["homo_ev"] + rng.normal(0.0, 0.18))
            oh_energy = float(
                parent.get("oh_addition_energy_kcal_mol", rng.normal(-12.0, 5.0))
                + rng.normal(0.0, 2.5)
            )
            logk = (
                coeffs["intercept"]
                + coeffs["homo_ev"] * homo
                + coeffs["min_bde_kcal_mol"] * min_bde
                + coeffs["oh_addition_energy_kcal_mol"] * oh_energy
                + rng.normal(0.0, noise_sd)
            )
            rows.append(
                {
                    "molecule": f"synthetic_{regime}_{i:03d}",
                    "parent_molecule": parent.get("molecule", ""),
                    "regime": regime,
                    "min_bde_kcal_mol": min_bde,
                    "homo_ev": homo,
                    "oh_addition_energy_kcal_mol": oh_energy,
                    "observed_log_k_deg": logk,
                    "rate_source": "synthetic",
                    "synthetic_notice": SYNTHETIC_RATE_NOTICE,
                }
            )

    return pd.DataFrame(rows)


def add_missingness(
    frame: pd.DataFrame,
    *,
    columns: Iterable[str],
    missing_fraction: float = 0.1,
    random_state: int = 19,
) -> pd.DataFrame:
    if not 0.0 <= missing_fraction < 1.0:
        raise ValueError("missing_fraction must be in [0, 1)")

    rng = np.random.default_rng(random_state)
    out = frame.copy()
    for col in columns:
        mask = rng.random(len(out)) < missing_fraction
        out.loc[mask, col] = np.nan
    return out


def rank_active_learning_candidates(
    candidates: pd.DataFrame,
    *,
    feature_columns: list[str],
    uncertainty_columns: list[str] | None = None,
    id_column: str = "molecule",
) -> pd.DataFrame:
    """Rank unmeasured candidates by uncertainty and descriptor-space diversity."""

    uncertainty_columns = uncertainty_columns or []
    required = [id_column, *feature_columns]
    missing = [col for col in required if col not in candidates.columns]
    if missing:
        raise ValueError(f"Candidate table is missing columns: {missing}")

    matrix = candidates.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.median(numeric_only=True))
    scaled = (matrix - matrix.mean()) / matrix.std(ddof=0).replace(0.0, 1.0)
    centroid_distance = np.sqrt((scaled**2).sum(axis=1))

    if uncertainty_columns:
        unc = candidates.loc[:, uncertainty_columns].apply(pd.to_numeric, errors="coerce")
        uncertainty_score = unc.fillna(unc.median(numeric_only=True)).mean(axis=1)
        uncertainty_score = (uncertainty_score - uncertainty_score.min()) / (
            uncertainty_score.max() - uncertainty_score.min() or 1.0
        )
    else:
        uncertainty_score = pd.Series(np.zeros(len(candidates)), index=candidates.index)

    diversity_score = (centroid_distance - centroid_distance.min()) / (
        centroid_distance.max() - centroid_distance.min() or 1.0
    )
    out = candidates.copy()
    out["descriptor_diversity_score"] = diversity_score
    out["measurement_uncertainty_score"] = uncertainty_score
    out["active_learning_score"] = 0.6 * uncertainty_score + 0.4 * diversity_score
    return out.sort_values("active_learning_score", ascending=False).reset_index(drop=True)


def run_synthetic_recovery_benchmark(
    synthetic_data: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str = "observed_log_k_deg",
) -> pd.DataFrame:
    """Fit one transparent model per synthetic regime and report top recovered feature."""

    rows: list[dict[str, object]] = []
    for regime, group in synthetic_data.groupby("regime"):
        result = train_bootstrap_ridge(
            group,
            feature_columns=feature_columns,
            target_column=target_column,
            id_column="molecule",
            test_fraction=0.25,
            n_bootstrap=150,
            random_state=23,
        )
        top = result.feature_importance.iloc[0]
        rows.append(
            {
                "regime": regime,
                "top_recovered_feature": top["feature"],
                "top_feature_importance": top["importance"],
                **result.metrics,
            }
        )
    return pd.DataFrame(rows)
