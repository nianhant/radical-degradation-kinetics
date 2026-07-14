#!/usr/bin/env python3
"""Set up DFT OH-addition adduct calculations from UMA NEB products.

UMA NEB geometries are used only as starting structures. The generated jobs run
r2SCAN-3c optimization/frequency calculations followed by wB97X-V single points.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path("/global/u2/n/nianhant/data/pharma_degradation")
DEFAULT_ORCA = Path("/global/homes/n/nianhant/m526_software/orca/orca_6_1_0/orca")
OPT_METHOD_DIR = "SMD(water)_r2SCAN-3c_def2-TZVP"
SP_METHOD_DIR = "SMD(water)_wB97X-V_def2-TZVP"


def slugify(text: str, max_len: int = 56) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text[:max_len].strip("_") or "entry"


def stable_id(dataset: str, molecule: str, label: str, product_xyz: Path) -> str:
    digest = hashlib.sha1(
        f"{dataset}|{molecule}|{label}|{product_xyz}".encode("utf-8")
    ).hexdigest()[:10]
    return f"ohadd_{slugify(molecule, 28)}_{slugify(label, 18)}_{digest}"


def read_neb_rows(datasets: Iterable[dict[str, object]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for info in datasets:
        manifest = Path(info["manifest"])
        if not manifest.exists():
            continue
        with manifest.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("reaction") != "oh_addition":
                    continue
                job_dir = Path(row["job_dir"])
                product_xyz = job_dir / "product.xyz"
                if not product_xyz.exists():
                    row["skip_reason"] = "missing_product_xyz"
                    continue
                row["dataset"] = str(info["dataset"])
                row["charge"] = str(info["charge"])
                row["multiplicity"] = "2"
                row["product_xyz"] = str(product_xyz)
                rows.append(row)
    return rows


def dataset_specs(root: Path) -> tuple[dict[str, object], ...]:
    return (
        {
            "dataset": "neutral",
            "charge": 0,
            "manifest": root / "ts_search/uma_neb_ts/neb_site_manifest.csv",
        },
        {
            "dataset": "carboxylate",
            "charge": -1,
            "manifest": root / "ts_search/uma_neb_ts_carboxylate/neb_site_manifest.csv",
        },
    )


def smiles_from_xyz(xyz_path: Path, charge: int) -> tuple[str, str]:
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
    except Exception as exc:  # pragma: no cover - depends on environment
        return "", f"rdkit_unavailable: {exc}"

    mol = None
    try:
        xyz_block = xyz_path.read_text()
        mol = Chem.MolFromXYZBlock(xyz_block)
        if mol is None:
            return "", "rdkit_xyz_parse_failed"
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
        mol_no_h = Chem.RemoveHs(mol, sanitize=False)
        Chem.SanitizeMol(mol_no_h)
        return Chem.MolToSmiles(mol_no_h, isomericSmiles=True), "ok"
    except Exception as exc:
        bond_order_error = exc

    try:
        mol = Chem.MolFromXYZBlock(xyz_path.read_text())
        if mol is None:
            return "", "rdkit_xyz_parse_failed"
        rdDetermineBonds.DetermineConnectivity(mol)
        mol_no_h = Chem.RemoveHs(mol, sanitize=False)
        return (
            Chem.MolToSmiles(mol_no_h, isomericSmiles=True),
            f"connectivity_only: {bond_order_error}",
        )
    except Exception as exc:
        return "", f"rdkit_bond_perception_failed: {bond_order_error}; connectivity_failed: {exc}"


def load_parent_smiles(root: Path) -> dict[tuple[str, str], str]:
    smiles: dict[tuple[str, str], str] = {}
    neutral_path = root / "results/molecule_smiles_functional_groups.csv"
    if neutral_path.exists():
        with neutral_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                smiles[("neutral", row["molecule"])] = row["smiles"]

    carboxylate_path = root / "scripts/utils/uma_deprotonated_molecule.csv"
    if carboxylate_path.exists():
        with carboxylate_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                smiles[("carboxylate", row["molecule"])] = row["fragment_smiles"]
    return smiles


def molecule_base_name(molecule: str) -> str:
    base = molecule
    for suffix in ("_carboxylate", "_deprot"):
        if suffix in base:
            base = base.split(suffix, 1)[0]
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        base = parts[0]
    if base.endswith("_0") or base.endswith("_1"):
        base = base.rsplit("_", 1)[0]
    return base


def parent_smiles_for(row: dict[str, str], smiles_map: dict[tuple[str, str], str]) -> tuple[str, str]:
    dataset = row["dataset"]
    base = molecule_base_name(row["molecule"])
    for key in ((dataset, base), ("neutral", base)):
        if key in smiles_map:
            return smiles_map[key], "ok"
    return "", "parent_smiles_not_found"


def write_text(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755 if path.name.endswith(".sh") else 0o644)


def orca_input(simple_input: str, charge: int, mult: int, xyz_name: str, nprocs: int, maxcore: int) -> str:
    return (
        f"! {simple_input}\n"
        f"%pal nprocs {nprocs} end\n"
        f"%maxcore {maxcore}\n"
        f"* xyzfile {charge} {mult} {xyz_name}\n"
    )


def submit_script(job_name: str, orca_cmd: Path, walltime: str, nprocs: int, mem_gb: int, preflight: str = "") -> str:
    return f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --output=log_files/%j.out
#SBATCH --error=log_files/%j.err
#SBATCH -t {walltime}
#SBATCH --ntasks={nprocs}
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem={mem_gb}G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nianhant@mit.edu

set -euo pipefail

module load openmpi
cd "$(dirname "$0")"
{preflight}
"{orca_cmd}" orca.inp > orca.out
"""


def prepare_jobs(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    root = Path(args.root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "structures").mkdir(exist_ok=True)
    opt_root = output_root / "OPT" / OPT_METHOD_DIR
    sp_root = output_root / "SP" / SP_METHOD_DIR
    rows = read_neb_rows(dataset_specs(root))
    parent_smiles_map = load_parent_smiles(root)

    manifest_rows: list[dict[str, str]] = []
    fragment_rows: list[dict[str, str]] = []

    for row in rows:
        charge = int(row["charge"])
        mult = int(row["multiplicity"])
        product_xyz = Path(row["product_xyz"])
        source_xyz = Path(row.get("source_xyz", ""))
        ohadduct_id = stable_id(row["dataset"], row["molecule"], row["label"], product_xyz)
        opt_dir = opt_root / ohadduct_id
        sp_dir = sp_root / ohadduct_id
        opt_dir.mkdir(parents=True, exist_ok=True)
        sp_dir.mkdir(parents=True, exist_ok=True)
        (opt_dir / "log_files").mkdir(exist_ok=True)
        (sp_dir / "log_files").mkdir(exist_ok=True)

        initial_xyz = opt_dir / "init.xyz"
        shutil.copy2(product_xyz, initial_xyz)
        shutil.copy2(product_xyz, output_root / "structures" / f"{ohadduct_id}.xyz")

        parent_smiles, parent_status = parent_smiles_for(row, parent_smiles_map)
        if not parent_smiles and source_xyz.exists():
            parent_smiles, parent_status = smiles_from_xyz(source_xyz, charge)
        adduct_smiles, adduct_status = smiles_from_xyz(product_xyz, charge)

        write_text(
            opt_dir / "orca.inp",
            orca_input(
                "r2SCAN-3c def2-TZVP OPT FREQ SMD(water) TightSCF",
                charge,
                mult,
                "init.xyz",
                args.nprocs,
                args.maxcore,
            ),
        )
        write_text(
            opt_dir / "submit.sh",
            submit_script(
                f"ohadd_opt_{ohadduct_id[-10:]}",
                Path(args.orca_cmd),
                args.opt_walltime,
                args.nprocs,
                args.mem_gb,
            ),
        )

        opt_xyz = opt_dir / "orca.xyz"
        preflight = (
            f'OPT_XYZ="{opt_xyz}"\n'
            'if [[ ! -f "$OPT_XYZ" ]]; then\n'
            '  echo "Missing optimized geometry: $OPT_XYZ" >&2\n'
            "  exit 1\n"
            "fi\n"
            'cp "$OPT_XYZ" opt.xyz'
        )
        write_text(
            sp_dir / "orca.inp",
            orca_input(
                "wB97X-V def2-TZVP SP SMD(water) TightSCF PrintMOs Printbasis defgrid3",
                charge,
                mult,
                "opt.xyz",
                args.nprocs,
                args.maxcore,
            ),
        )
        write_text(
            sp_dir / "submit.sh",
            submit_script(
                f"ohadd_sp_{ohadduct_id[-10:]}",
                Path(args.orca_cmd),
                args.sp_walltime,
                args.nprocs,
                args.mem_gb,
                preflight=preflight,
            ),
        )

        manifest_rows.append(
            {
                "ohadduct_id": ohadduct_id,
                "parent_molecule": row["molecule"],
                "dataset": row["dataset"],
                "reaction": row["reaction"],
                "label": row["label"],
                "addition_atom_index": row["atom_index"],
                "addition_atom_symbol": row["atom_symbol"],
                "charge": row["charge"],
                "multiplicity": row["multiplicity"],
                "oh_adduct_smiles": adduct_smiles,
                "source_xyz": str(source_xyz),
                "uma_neb_job_dir": row["job_dir"],
                "uma_product_xyz": str(product_xyz),
                "initial_xyz": str(initial_xyz),
                "structure_xyz": str(output_root / "structures" / f"{ohadduct_id}.xyz"),
                "opt_dir": str(opt_dir),
                "opt_input": str(opt_dir / "orca.inp"),
                "opt_submit": str(opt_dir / "submit.sh"),
                "sp_dir": str(sp_dir),
                "sp_input": str(sp_dir / "orca.inp"),
                "sp_submit": str(sp_dir / "submit.sh"),
                "parent_smiles_status": parent_status,
                "adduct_smiles_status": adduct_status,
            }
        )
        fragment_rows.append(
            {
                "ohadduct_id": ohadduct_id,
                "parent_molecule": row["molecule"],
                "dataset": row["dataset"],
                "reaction": row["reaction"],
                "label": row["label"],
                "addition_atom_index": row["atom_index"],
                "addition_atom_symbol": row["atom_symbol"],
                "before_parent_molecule_smiles": parent_smiles,
                "before_radical_smiles": "[OH]",
                "after_oh_adduct_radical_smiles": adduct_smiles,
                "before_parent_xyz": str(source_xyz),
                "before_radical_xyz": "",
                "after_oh_adduct_xyz": str(product_xyz),
                "charge": row["charge"],
                "multiplicity": row["multiplicity"],
                "parent_smiles_status": parent_status,
                "adduct_smiles_status": adduct_status,
            }
        )

    return manifest_rows, fragment_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_submit_all(output_root: Path, manifest_rows: list[dict[str, str]]) -> None:
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'LOG="$ROOT/submitted_jobs.tsv"',
        'echo -e "ohadduct_id\\topt_job_id\\tsp_job_id" > "$LOG"',
    ]
    for row in manifest_rows:
        lines.extend(
            [
                f'opt_job=$(sbatch --parsable "{row["opt_submit"]}")',
                f'sp_job=$(sbatch --parsable --dependency=afterok:${{opt_job}} "{row["sp_submit"]}")',
                f'echo -e "{row["ohadduct_id"]}\\t${{opt_job}}\\t${{sp_job}}" >> "$LOG"',
            ]
        )
    write_text(output_root / "submit_all.sh", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_ROOT / "oh_addition",
        help="Root directory for generated OH-addition DFT jobs and manifests.",
    )
    parser.add_argument("--orca-cmd", type=Path, default=DEFAULT_ORCA)
    parser.add_argument("--nprocs", type=int, default=8)
    parser.add_argument("--maxcore", type=int, default=4000)
    parser.add_argument("--mem-gb", type=int, default=64)
    parser.add_argument("--opt-walltime", default="48:00:00")
    parser.add_argument("--sp-walltime", default="12:00:00")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "structures").mkdir(exist_ok=True)
    manifest_rows, fragment_rows = prepare_jobs(args)
    write_csv(args.output_root / "oh_addition_adduct_manifest.csv", manifest_rows)
    write_csv(args.output_root / "oh_addition_fragment_pairs.csv", fragment_rows)
    write_submit_all(args.output_root, manifest_rows)
    protocol = {
        "geometry_source": "UMA NEB oh_addition product.xyz files only",
        "dft_workflow": [
            {
                "step": "OPT_FREQ",
                "method_dir": OPT_METHOD_DIR,
                "orca_simple_input": "r2SCAN-3c def2-TZVP OPT FREQ SMD(water) TightSCF",
            },
            {
                "step": "SP",
                "method_dir": SP_METHOD_DIR,
                "orca_simple_input": "wB97X-V def2-TZVP SP SMD(water) TightSCF PrintMOs Printbasis defgrid3",
            },
        ],
        "charge_policy": {
            "neutral_uma_neb_ts": 0,
            "uma_neb_ts_carboxylate": -1,
        },
        "multiplicity": 2,
        "n_jobs": len(manifest_rows),
        "manifest": str(args.output_root / "oh_addition_adduct_manifest.csv"),
        "fragment_pairs": str(args.output_root / "oh_addition_fragment_pairs.csv"),
    }
    (args.output_root / "oh_addition_dft_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    print(f"Wrote {len(manifest_rows)} OH-addition DFT job sets under {args.output_root}")
    print(f"Manifest: {args.output_root / 'oh_addition_adduct_manifest.csv'}")
    print(f"Fragment pairs: {args.output_root / 'oh_addition_fragment_pairs.csv'}")
    print(f"Submit helper: {args.output_root / 'submit_all.sh'}")


if __name__ == "__main__":
    main()
