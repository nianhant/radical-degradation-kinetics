from __future__ import annotations

from time import perf_counter
from typing import Any

import pandas as pd

from .preopt import get_ml_calculator, rdkit_mol_to_ase_atoms


def benchmark_ml_potentials(
    mol: Any,
    *,
    backends: list[str] | tuple[str, ...] = ("uma", "mace", "orb"),
    calculators: dict[str, Any] | None = None,
    calculator_kwargs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    calculators = calculators or {}
    calculator_kwargs = calculator_kwargs or {}

    atoms = rdkit_mol_to_ase_atoms(mol)
    rows: list[dict[str, Any]] = []

    for backend in backends:
        atoms_copy = atoms.copy()
        calc = calculators.get(backend)
        status = "ok"
        energy = None
        max_force = None
        elapsed_s = None
        error = None

        try:
            atoms_copy.calc = calc if calc is not None else get_ml_calculator(
                backend, **calculator_kwargs.get(backend, {})
            )
            start = perf_counter()
            energy = float(atoms_copy.get_potential_energy())
            forces = atoms_copy.get_forces()
            elapsed_s = perf_counter() - start
            max_force = float(abs(forces).max())
        except Exception as exc:  # pragma: no cover
            status = "failed"
            error = str(exc)

        rows.append(
            {
                "backend": backend,
                "status": status,
                "energy_eV": energy,
                "max_force_eV_per_A": max_force,
                "elapsed_s": elapsed_s,
                "error": error,
            }
        )

    return pd.DataFrame(rows)
