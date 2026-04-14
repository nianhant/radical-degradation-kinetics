from __future__ import annotations

from pathlib import Path

import pandas as pd

from radical_degradation.dft import write_orca_job
from radical_degradation.fetch import build_molecule_table
from radical_degradation.fragments import enumerate_homolytic_fragments
from radical_degradation.preopt import embed_and_optimize_smiles, mol_to_xyz_block


def main() -> None:
    here = Path(__file__).resolve().parent
    output_dir = here / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_df = pd.read_csv(here / "sample_molecules.csv")
    molecule_table = build_molecule_table(input_df.to_dict(orient="records"))
    molecule_table.to_csv(output_dir / "molecule_table.csv", index=False)

    fragment_tables = []
    for _, row in molecule_table.iterrows():
        fragments = enumerate_homolytic_fragments(row["canonical_smiles"])
        fragments["molecule"] = row["name"]
        fragment_tables.append(fragments)

        mol = embed_and_optimize_smiles(row["canonical_smiles"])
        xyz_text = mol_to_xyz_block(mol, comment=row["name"])
        write_orca_job(
            output_dir / row["name"],
            xyz_text,
            job_name=f"{row['name']}_dft",
        )

    pd.concat(fragment_tables, ignore_index=True).to_csv(
        output_dir / "fragment_manifest.csv", index=False
    )
    print(f"Wrote demo outputs to {output_dir}")


if __name__ == "__main__":
    main()
