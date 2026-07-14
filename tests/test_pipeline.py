from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radical_degradation.dft import build_orca_input, build_slurm_script
from radical_degradation.modeling import regression_metrics, train_bootstrap_ridge
from radical_degradation.neb import build_uma_neb_input
from radical_degradation.study import (
    SYNTHETIC_RATE_NOTICE,
    generate_synthetic_regime_data,
    rank_active_learning_candidates,
    run_synthetic_recovery_benchmark,
)


def test_build_orca_input_includes_xyzfile() -> None:
    text = build_orca_input("molecule.xyz", charge=-1, multiplicity=2)
    assert "* xyzfile -1 2 molecule.xyz" in text


def test_build_slurm_script_contains_orca_call() -> None:
    text = build_slurm_script(job_name="demo_job", input_name="demo.inp")
    assert "#SBATCH -J demo_job" in text
    assert "orca demo.inp > job.out" in text


def test_build_uma_neb_input_contains_extopt_and_endpoint() -> None:
    text = build_uma_neb_input(
        "reactant.xyz",
        "product.xyz",
        uma_wrapper="/opt/oet_uma",
        ext_params="--model uma-s-1p1",
        charge=1,
        multiplicity=2,
        nimages=10,
    )
    assert "! ExtOpt NEB-TS PAL16" in text
    assert 'ProgExt "/opt/oet_uma"' in text
    assert 'Ext_Params "--model uma-s-1p1"' in text
    assert 'NEB_END_XYZFILE "product.xyz"' in text
    assert "NImages 10" in text
    assert "* xyzfile 1 2 reactant.xyz" in text


def test_regression_metrics_are_reasonable() -> None:
    metrics = regression_metrics(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.5, 2.5]),
    )
    assert metrics["rmse"] == pytest.approx(0.408248, rel=1.0e-5)
    assert metrics["mae"] == pytest.approx(1.0 / 3.0)
    assert metrics["r2"] == pytest.approx(0.75)


def test_train_bootstrap_ridge_outputs_uncertainty_and_decision_table() -> None:
    frame = pd.DataFrame(
        {
            "molecule": [f"mol_{i}" for i in range(8)],
            "bde": [91.0, 88.0, 86.0, 83.0, 81.0, 89.0, 84.0, 82.0],
            "homo": [-6.1, -5.9, -5.7, -5.4, -5.2, -5.8, -5.5, -5.3],
            "observed_log_k": [0.2, 0.35, 0.55, 0.78, 0.95, 0.38, 0.70, 0.86],
        }
    )
    result = train_bootstrap_ridge(
        frame,
        feature_columns=["bde", "homo"],
        target_column="observed_log_k",
        n_bootstrap=25,
        random_state=7,
    )
    assert result.model_name == "bootstrap_ridge"
    assert set(result.metrics) >= {"rmse", "mae", "r2", "coverage_90"}
    assert {"prediction_std", "lower_90", "upper_90"}.issubset(result.predictions.columns)
    assert result.feature_importance.iloc[0]["importance"] >= 0.0
    assert "recommended_action" in result.decision_table.columns


def test_synthetic_regime_data_is_labeled_and_recoverable() -> None:
    base = pd.DataFrame(
        {
            "molecule": ["a", "b", "c", "d"],
            "min_bde_kcal_mol": [82.0, 86.0, 90.0, 94.0],
            "homo_ev": [-5.2, -5.5, -5.8, -6.1],
            "oh_addition_energy_kcal_mol": [-20.0, -15.0, -10.0, -5.0],
        }
    )
    synthetic = generate_synthetic_regime_data(
        base,
        n_per_regime=18,
        noise_sd=0.01,
        random_state=5,
    )
    assert set(synthetic["rate_source"]) == {"synthetic"}
    assert synthetic["synthetic_notice"].eq(SYNTHETIC_RATE_NOTICE).all()

    recovery = run_synthetic_recovery_benchmark(
        synthetic,
        feature_columns=["min_bde_kcal_mol", "homo_ev", "oh_addition_energy_kcal_mol"],
    )
    recovered = dict(zip(recovery["regime"], recovery["top_recovered_feature"]))
    assert recovered["hydrogen_abstraction_controlled"] == "min_bde_kcal_mol"
    assert recovered["hydroxyl_addition_controlled"] == "oh_addition_energy_kcal_mol"


def test_rank_active_learning_candidates_combines_uncertainty_and_diversity() -> None:
    candidates = pd.DataFrame(
        {
            "molecule": ["central", "uncertain", "diverse"],
            "bde": [86.0, 87.0, 98.0],
            "homo": [-5.6, -5.5, -6.4],
            "model_uncertainty": [0.1, 0.9, 0.2],
        }
    )
    ranked = rank_active_learning_candidates(
        candidates,
        feature_columns=["bde", "homo"],
        uncertainty_columns=["model_uncertainty"],
    )
    assert ranked.iloc[0]["molecule"] in {"uncertain", "diverse"}
    assert ranked["active_learning_score"].is_monotonic_decreasing


class FakeCalculator:
    def __init__(self, energy: float, force_scale: float) -> None:
        self.energy = energy
        self.force_scale = force_scale

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return self.energy

    def get_forces(self, atoms=None):
        natoms = len(atoms)
        return np.full((natoms, 3), self.force_scale)


def test_benchmark_ml_potentials_with_fake_calculators() -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("ase")
    from radical_degradation.benchmark import benchmark_ml_potentials
    from radical_degradation.preopt import embed_and_optimize_smiles

    mol = embed_and_optimize_smiles("CCO")
    benchmark_df = benchmark_ml_potentials(
        mol,
        backends=["uma", "mace", "orb"],
        calculators={
            "uma": FakeCalculator(-1.0, 0.1),
            "mace": FakeCalculator(-1.2, 0.2),
            "orb": FakeCalculator(-0.8, 0.05),
        },
    )
    assert set(benchmark_df["backend"]) == {"uma", "mace", "orb"}
    assert (benchmark_df["status"] == "ok").all()
