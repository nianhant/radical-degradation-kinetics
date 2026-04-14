from __future__ import annotations

import numpy as np
import pytest

from radical_degradation.dft import build_orca_input, build_slurm_script


def test_build_orca_input_includes_xyzfile() -> None:
    text = build_orca_input("molecule.xyz", charge=-1, multiplicity=2)
    assert "* xyzfile -1 2 molecule.xyz" in text


def test_build_slurm_script_contains_orca_call() -> None:
    text = build_slurm_script(job_name="demo_job", input_name="demo.inp")
    assert "#SBATCH -J demo_job" in text
    assert "orca demo.inp > job.out" in text


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
