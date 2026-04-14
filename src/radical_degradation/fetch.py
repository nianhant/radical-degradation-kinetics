from __future__ import annotations

from typing import Iterable

import pandas as pd

try:
    from rdkit import Chem
except ImportError as exc:  # pragma: no cover
    raise ImportError("RDKit is required for fetch.py") from exc


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def resolve_name_to_smiles(name: str) -> str:
    try:
        import pubchempy as pcp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pubchempy is not installed. Install it to resolve molecule names."
        ) from exc

    compounds = pcp.get_compounds(name, namespace="name")
    if not compounds:
        raise ValueError(f"No PubChem match found for molecule name: {name}")

    smiles = compounds[0].isomeric_smiles
    if not smiles:
        raise ValueError(f"PubChem match for {name} did not include SMILES.")

    return canonicalize_smiles(smiles)


def build_molecule_table(
    molecules: Iterable[dict],
    *,
    name_key: str = "name",
    smiles_key: str = "smiles",
) -> pd.DataFrame:
    rows = []
    for molecule in molecules:
        name = molecule.get(name_key)
        smiles = molecule.get(smiles_key)

        if smiles:
            canonical = canonicalize_smiles(smiles)
        elif name:
            canonical = resolve_name_to_smiles(name)
        else:
            raise ValueError("Each molecule entry must include either a name or a SMILES.")

        rows.append(
            {
                "name": name if name else canonical,
                "input_smiles": smiles,
                "canonical_smiles": canonical,
            }
        )

    return pd.DataFrame(rows)
