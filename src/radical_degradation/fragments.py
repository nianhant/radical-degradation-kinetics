from __future__ import annotations

import pandas as pd

try:
    from rdkit import Chem
except ImportError as exc:  # pragma: no cover
    raise ImportError("RDKit is required for fragments.py") from exc


def _rdkit_homolytic_fragments(smiles: str) -> list[dict]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    rows: list[dict] = []
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue

        bond_idx = bond.GetIdx()
        fragmented = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=False)
        frags = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=True)
        if len(frags) != 2:
            continue

        frag_smiles = sorted(Chem.MolToSmiles(frag, canonical=True) for frag in frags)
        rows.append(
            {
                "bond_index": bond_idx,
                "bond_type": str(bond.GetBondType()),
                "fragment1": frag_smiles[0],
                "fragment2": frag_smiles[1],
                "source": "rdkit_fallback",
            }
        )

    return rows


def enumerate_homolytic_fragments(smiles: str) -> pd.DataFrame:
    try:
        from alfabet import model
    except ImportError:
        return pd.DataFrame(_rdkit_homolytic_fragments(smiles))

    prediction_df = model.predict([smiles]).copy()
    keep_cols = [
        "bond_index",
        "bond_type",
        "fragment1",
        "fragment2",
        "bde_pred",
        "bdfe_pred",
    ]
    available_cols = [col for col in keep_cols if col in prediction_df.columns]
    out_df = prediction_df[available_cols].copy()
    out_df["source"] = "alfabet"
    return out_df
