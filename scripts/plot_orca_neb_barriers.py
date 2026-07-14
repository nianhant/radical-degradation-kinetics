from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HARTREE_TO_KCAL_MOL = 627.509474
EV_TO_KCAL_MOL = 23.060548
KJ_TO_KCAL = 0.239005736

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


@dataclass(frozen=True)
class ParsedBarrier:
    barrier_kcal_mol: float
    source: Path
    method: str


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "plot"


def _float_tokens(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?", text)]


def _convert_to_kcal(value: float, unit: str) -> float:
    unit = unit.lower().replace(" ", "")
    if unit in {"kcal/mol", "kcalmol-1", "kcal"}:
        return value
    if unit in {"kj/mol", "kjmol-1", "kj"}:
        return value * KJ_TO_KCAL
    if unit in {"ev"}:
        return value * EV_TO_KCAL_MOL
    if unit in {"eh", "hartree", "hartrees", "au", "a.u."}:
        return value * HARTREE_TO_KCAL_MOL
    return value


def _parse_explicit_barrier(path: Path) -> ParsedBarrier | None:
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return None

    patterns = [
        re.compile(
            r"(?:activation\s+energy|forward\s+barrier|barrier)"
            r"[^-\d\n]{0,80}([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?)"
            r"\s*(kcal/mol|kJ/mol|eV|Eh|Hartree|a\.u\.)",
            re.IGNORECASE,
        ),
        re.compile(
            r"([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?)"
            r"\s*(kcal/mol|kJ/mol|eV|Eh|Hartree|a\.u\.)"
            r"[^\n]{0,80}(?:activation\s+energy|forward\s+barrier|barrier)",
            re.IGNORECASE,
        ),
    ]
    for pattern in patterns:
        matches = pattern.findall(text)
        if matches:
            value, unit = matches[-1]
            return ParsedBarrier(_convert_to_kcal(float(value), unit), path, "explicit_barrier")
    return None


def _read_xyz_comment_energies(path: Path) -> list[float]:
    lines = path.read_text(errors="ignore").splitlines()
    energies: list[float] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped.isdigit():
            i += 1
            continue
        natoms = int(stripped)
        if i + 1 < len(lines):
            comment = lines[i + 1]
            match = re.search(
                r"(?:energy|E)\s*[=:]?\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?)",
                comment,
                re.IGNORECASE,
            )
            if match:
                energies.append(float(match.group(1)))
        i += natoms + 2
    return energies


def _read_numeric_energy_series(path: Path) -> list[float]:
    suffix = path.suffix.lower()
    if suffix == ".xyz":
        return _read_xyz_comment_energies(path)

    energies: list[float] = []
    for line in path.read_text(errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "!", "%", "*")):
            continue
        numbers = _float_tokens(stripped)
        if len(numbers) < 2:
            continue
        lower = stripped.lower()
        if "image" in lower or "energy" in lower or path.suffix.lower() in {".dat", ".csv", ".txt"}:
            energies.append(numbers[-1])
    return energies


def _series_unit_multiplier(energies: list[float]) -> float:
    finite = [x for x in energies if math.isfinite(x)]
    if not finite:
        return 1.0
    median_abs = sorted(abs(x) for x in finite)[len(finite) // 2]
    if median_abs > 10.0:
        return HARTREE_TO_KCAL_MOL
    if median_abs < 20.0:
        return EV_TO_KCAL_MOL
    return 1.0


def _barrier_from_series(path: Path) -> ParsedBarrier | None:
    try:
        energies = _read_numeric_energy_series(path)
    except OSError:
        return None
    if len(energies) < 2:
        return None
    multiplier = _series_unit_multiplier(energies)
    barrier = (max(energies) - energies[0]) * multiplier
    if barrier < -1.0e-8:
        return None
    return ParsedBarrier(barrier, path, "energy_series")


def parse_barrier(job_dir: Path) -> ParsedBarrier | None:
    text_candidates = sorted(
        p
        for p in job_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".out", ".log", ".txt"}
    )
    for path in text_candidates:
        parsed = _parse_explicit_barrier(path)
        if parsed is not None:
            return parsed

    energy_candidates = sorted(
        p
        for p in job_dir.iterdir()
        if p.is_file()
        and (
            p.suffix.lower() in {".dat", ".csv", ".txt", ".xyz"}
            or re.search(r"(energy|energies|mep|path|interp|neb)", p.name, re.IGNORECASE)
        )
    )
    for path in energy_candidates:
        parsed = _barrier_from_series(path)
        if parsed is not None:
            return parsed
    return None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def find_job_dirs(root: Path, exclude: Path | None = None) -> list[Path]:
    markers = {"neb.inp", "reactant.xyz", "product.xyz", "neb.out"}
    dirs: set[Path] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in {
            "orca_neb_barriers.csv",
            "neb_site_manifest.csv",
            "molecule_table.csv",
        } or path.suffix.lower() in {".png", ".pdf", ".svg"}:
            continue
        if exclude is not None and _is_relative_to(path.resolve(), exclude.resolve()):
            continue
        likely_energy_file = re.search(
            r"(neb|mep|path|interp|energ(?:y|ies)|barrier)", path.name, re.IGNORECASE
        )
        if path.name in markers or path.suffix.lower() == ".out" or likely_energy_file:
            if path.name in markers or "neb" in path.name.lower() or likely_energy_file:
                dirs.add(path.parent)
    return sorted(dirs)


def load_manifest(root: Path, explicit: Path | None) -> pd.DataFrame:
    candidates = [explicit] if explicit else []
    candidates.extend(root.rglob("neb_site_manifest.csv"))
    for path in candidates:
        if path and path.exists():
            df = pd.read_csv(path)
            df["manifest_path"] = str(path)
            return df
    return pd.DataFrame()


def load_molecule_table(root: Path, explicit: Path | None) -> pd.DataFrame:
    candidates = [explicit] if explicit else []
    candidates.extend(root.rglob("molecule_table.csv"))
    for path in candidates:
        if path and path.exists():
            df = pd.read_csv(path)
            df["molecule_table_path"] = str(path)
            return df
    return pd.DataFrame()


def infer_metadata(job_dir: Path, root: Path) -> dict[str, object]:
    rel = job_dir.relative_to(root)
    parts = rel.parts
    molecule = parts[-3] if len(parts) >= 3 else (parts[0] if parts else job_dir.name)
    reaction = parts[-2] if len(parts) >= 2 else ""
    label = parts[-1] if parts else job_dir.name
    return {"molecule": molecule, "reaction": reaction, "label": label, "relative_dir": str(rel)}


def gather_results(root: Path, manifest: pd.DataFrame, exclude: Path | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    manifest_keyed = pd.DataFrame()
    if not manifest.empty and {"molecule", "reaction", "label"}.issubset(manifest.columns):
        manifest_keyed = manifest.copy()
        manifest_keyed["_molecule_slug"] = manifest_keyed["molecule"].astype(str).map(_slug)
        manifest_keyed["_label_slug"] = manifest_keyed["label"].astype(str).map(_slug)

    for job_dir in find_job_dirs(root, exclude=exclude):
        row = infer_metadata(job_dir, root)
        parsed = parse_barrier(job_dir)
        row["job_dir"] = str(job_dir)
        row["barrier_kcal_mol"] = parsed.barrier_kcal_mol if parsed else math.nan
        row["parse_source"] = str(parsed.source) if parsed else ""
        row["parse_method"] = parsed.method if parsed else "not_found"

        if not manifest_keyed.empty:
            matches = manifest_keyed[
                (manifest_keyed["_molecule_slug"] == str(row["molecule"]))
                & (manifest_keyed["reaction"].astype(str) == str(row["reaction"]))
                & (manifest_keyed["_label_slug"] == str(row["label"]))
            ]
            if not matches.empty:
                for col, value in matches.iloc[0].items():
                    if not col.startswith("_"):
                        row[col] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _get_rdkit_mol(smiles: str):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    for atom in mol.GetAtoms():
        atom.SetProp("atomLabel", f"{atom.GetSymbol()}{atom.GetIdx()}")
    return mol


def _draw_molecule(ax, smiles: str | None, group: pd.DataFrame) -> None:
    ax.axis("off")
    if not smiles:
        ax.text(0.5, 0.5, "No molecule structure available", ha="center", va="center")
        return
    mol = _get_rdkit_mol(smiles)
    if mol is None:
        ax.text(0.5, 0.5, "RDKit unavailable or invalid SMILES", ha="center", va="center")
        return

    from rdkit.Chem import Draw

    highlight_atoms: set[int] = set()
    highlight_bonds: set[int] = set()
    for _, row in group.iterrows():
        atom_index = row.get("atom_index")
        partner_index = row.get("partner_index")
        if pd.notna(atom_index):
            highlight_atoms.add(int(atom_index))
        if pd.notna(partner_index):
            highlight_atoms.add(int(partner_index))
            bond = mol.GetBondBetweenAtoms(int(atom_index), int(partner_index))
            if bond is not None:
                highlight_bonds.add(bond.GetIdx())

    image = Draw.MolToImage(
        mol,
        size=(900, 520),
        highlightAtoms=sorted(highlight_atoms),
        highlightBonds=sorted(highlight_bonds),
    )
    ax.imshow(image)


def plot_results(results: pd.DataFrame, molecule_table: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    smiles_by_name: dict[str, str] = {}
    if not molecule_table.empty:
        name_col = "name" if "name" in molecule_table.columns else "molecule"
        smiles_col = (
            "canonical_smiles"
            if "canonical_smiles" in molecule_table.columns
            else ("smiles" if "smiles" in molecule_table.columns else "")
        )
        if smiles_col:
            smiles_by_name = dict(
                zip(molecule_table[name_col].astype(str), molecule_table[smiles_col].astype(str))
            )

    plottable = results[pd.notna(results["barrier_kcal_mol"])].copy()
    if plottable.empty:
        return

    for (molecule, reaction), group in plottable.groupby(["molecule", "reaction"], dropna=False):
        group = group.sort_values("label")
        fig = plt.figure(figsize=(max(7.0, 0.55 * len(group) + 3.0), 7.5), constrained_layout=True)
        grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15])
        mol_ax = fig.add_subplot(grid[0, 0])
        bar_ax = fig.add_subplot(grid[1, 0])

        smiles = smiles_by_name.get(str(molecule))
        _draw_molecule(mol_ax, smiles, group)
        mol_ax.set_title(f"{molecule} - {reaction}", fontsize=13)

        labels = group["label"].astype(str).tolist()
        values = group["barrier_kcal_mol"].astype(float).tolist()
        colors = ["#4B8BBE" if str(reaction) == "h_abstraction" else "#D95F02"] * len(values)
        bar_ax.bar(labels, values, color=colors, edgecolor="#222222", linewidth=0.8)
        bar_ax.set_ylabel("Forward barrier (kcal/mol)")
        bar_ax.set_xlabel("Bond/site label from molecule structure")
        bar_ax.tick_params(axis="x", rotation=45)
        bar_ax.grid(axis="y", alpha=0.25)
        for tick in bar_ax.get_xticklabels():
            tick.set_horizontalalignment("right")

        out_path = output_dir / f"{_slug(str(molecule))}_{_slug(str(reaction))}_barriers.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gather ORCA NEB-TS results and plot per-molecule barrier bar charts."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("ts_search"),
        help="Root directory containing molecule/reaction/site ORCA NEB folders.",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Optional neb_site_manifest.csv")
    parser.add_argument("--molecule-table", type=Path, default=None, help="Optional molecule_table.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/orca_neb_barriers"),
        help="Directory for PNG plots and the gathered CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    if not root.exists():
        raise SystemExit(f"Root directory does not exist: {root}")

    manifest = load_manifest(root, args.manifest)
    molecule_table = load_molecule_table(root, args.molecule_table)
    results = gather_results(root, manifest, exclude=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "orca_neb_barriers.csv"
    results.to_csv(csv_path, index=False)
    plot_results(results, molecule_table, output_dir)

    parsed = int(pd.notna(results.get("barrier_kcal_mol", pd.Series(dtype=float))).sum())
    print(f"Wrote gathered results to {csv_path}")
    print(f"Parsed barriers for {parsed} of {len(results)} job directories")
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
