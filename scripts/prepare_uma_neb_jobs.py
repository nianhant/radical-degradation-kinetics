from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare ORCA ExtOpt NEB-TS jobs using a UMA external-method wrapper "
            "for pharmaceutical H abstraction and OH addition pathways."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="CSV with at least name and smiles columns; charge is optional.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("neb_uma_jobs"),
        help="Directory where job folders and manifest CSVs are written.",
    )
    parser.add_argument(
        "--reaction",
        choices=["h_abstraction", "oh_addition", "both"],
        default="both",
        help="Which endpoint family to generate.",
    )
    parser.add_argument(
        "--max-sites",
        type=int,
        default=6,
        help="Maximum sites per reaction per molecule before descriptor filtering.",
    )
    parser.add_argument(
        "--uma-wrapper",
        default="/full/path/to/oet_uma",
        help="Full path to the ORCA External Tools UMA wrapper or client script.",
    )
    parser.add_argument(
        "--ext-params",
        default="",
        help="Optional Ext_Params string forwarded to the UMA wrapper.",
    )
    parser.add_argument("--nprocs", type=int, default=16)
    parser.add_argument("--nimages", type=int, default=8)
    parser.add_argument("--walltime", default="24:00:00")
    parser.add_argument(
        "--pre-command",
        default="",
        help="Optional shell command placed before the ORCA call, e.g. start a UMA server.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from radical_degradation.fetch import build_molecule_table
    from radical_degradation.neb import (
        h_abstraction_endpoints,
        oh_addition_endpoints,
        ranked_default_sites,
        site_manifest_rows,
        write_uma_neb_job,
    )
    from radical_degradation.preopt import embed_and_optimize_smiles

    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_df = pd.read_csv(args.input_csv)
    molecule_table = build_molecule_table(input_df.to_dict(orient="records"))
    molecule_table.to_csv(args.output_dir / "molecule_table.csv", index=False)

    reactions = (
        ["h_abstraction", "oh_addition"]
        if args.reaction == "both"
        else [args.reaction]
    )
    manifest_rows: list[dict[str, object]] = []

    for _, row in molecule_table.iterrows():
        name = str(row["name"])
        smiles = str(row["canonical_smiles"])
        charge = int(row["charge"]) if "charge" in row and not pd.isna(row["charge"]) else 0
        mol = embed_and_optimize_smiles(smiles)

        for reaction in reactions:
            sites = ranked_default_sites(mol, reaction=reaction, max_sites=args.max_sites)
            manifest_rows.extend(site_manifest_rows(name, sites))

            for site in sites:
                if reaction == "h_abstraction":
                    reactant_xyz, product_xyz = h_abstraction_endpoints(
                        mol,
                        site.atom_index,
                        comment_prefix=f"{name} {site.label}",
                    )
                else:
                    reactant_xyz, product_xyz = oh_addition_endpoints(
                        mol,
                        site.atom_index,
                        comment_prefix=f"{name} {site.label}",
                    )

                job_dir = args.output_dir / _slug(name) / reaction / _slug(site.label)
                write_uma_neb_job(
                    job_dir,
                    reactant_xyz,
                    product_xyz,
                    job_name=_slug(f"{name}_{reaction}_{site.label}")[:120],
                    uma_wrapper=args.uma_wrapper,
                    ext_params=args.ext_params,
                    charge=charge,
                    multiplicity=2,
                    nprocs=args.nprocs,
                    nimages=args.nimages,
                    walltime=args.walltime,
                    pre_command=args.pre_command,
                )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.output_dir / "neb_site_manifest.csv", index=False)
    print(f"Wrote UMA NEB jobs to {args.output_dir}")
    print(f"Wrote site manifest to {args.output_dir / 'neb_site_manifest.csv'}")


if __name__ == "__main__":
    main()
