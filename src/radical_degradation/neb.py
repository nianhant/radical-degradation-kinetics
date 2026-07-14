from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class NebSite:
    reaction: str
    atom_index: int
    atom_symbol: str
    label: str
    partner_index: int | None = None
    partner_symbol: str | None = None


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1.0e-8:
        return np.array([1.0, 0.0, 0.0])
    return vector / norm


def _perpendicular_unit(vector: np.ndarray) -> np.ndarray:
    trial = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(_unit_vector(vector), trial))) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    return _unit_vector(np.cross(vector, trial))


def _xyz_text(symbols: list[str], positions: np.ndarray, comment: str) -> str:
    lines = [str(len(symbols)), comment]
    for symbol, pos in zip(symbols, positions):
        lines.append(f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
    return "\n".join(lines) + "\n"


def build_uma_neb_input(
    reactant_xyz_filename: str = "reactant.xyz",
    product_xyz_filename: str = "product.xyz",
    *,
    uma_wrapper: str = "/full/path/to/oet_uma",
    ext_params: str = "",
    charge: int = 0,
    multiplicity: int = 2,
    nprocs: int = 16,
    nimages: int = 8,
    neb_climb: bool = True,
    freq: bool = False,
) -> str:
    keyword_line = f"! ExtOpt NEB-TS PAL{nprocs}"
    if freq:
        keyword_line += " FREQ"

    ext_params_line = f'  Ext_Params "{ext_params}"\n' if ext_params else ""
    climb_line = "  Climb true\n" if neb_climb else ""
    return (
        f"{keyword_line}\n\n"
        f"%pal nprocs {nprocs} end\n\n"
        "%method\n"
        f'  ProgExt "{uma_wrapper}"\n'
        f"{ext_params_line}"
        "end\n\n"
        "%NEB\n"
        f'  NEB_END_XYZFILE "{product_xyz_filename}"\n'
        f"  NImages {nimages}\n"
        f"{climb_line}"
        "end\n\n"
        f"* xyzfile {charge} {multiplicity} {reactant_xyz_filename}\n"
    )


def build_neb_slurm_script(
    *,
    job_name: str,
    input_name: str = "neb.inp",
    walltime: str = "24:00:00",
    nodes: int = 1,
    ntasks: int = 16,
    orca_module: str = "orca",
    pre_command: str = "",
) -> str:
    pre_block = f"{pre_command}\n" if pre_command else ""
    return (
        "#!/bin/bash\n"
        f"#SBATCH -J {job_name}\n"
        f"#SBATCH -N {nodes}\n"
        f"#SBATCH -n {ntasks}\n"
        f"#SBATCH -t {walltime}\n"
        "#SBATCH -C cpu\n"
        "#SBATCH -q shared\n"
        "#SBATCH --output=%x.%j.out\n"
        "#SBATCH --error=%x.%j.err\n\n"
        "set -euo pipefail\n"
        f"module load {orca_module}\n"
        f"{pre_block}"
        f"orca {input_name} > neb.out\n"
    )


def write_uma_neb_job(
    output_dir: str | Path,
    reactant_xyz: str,
    product_xyz: str,
    *,
    job_name: str,
    input_name: str = "neb.inp",
    reactant_name: str = "reactant.xyz",
    product_name: str = "product.xyz",
    uma_wrapper: str = "/full/path/to/oet_uma",
    ext_params: str = "",
    charge: int = 0,
    multiplicity: int = 2,
    nprocs: int = 16,
    nimages: int = 8,
    walltime: str = "24:00:00",
    pre_command: str = "",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reactant_path = output_dir / reactant_name
    product_path = output_dir / product_name
    input_path = output_dir / input_name
    submit_path = output_dir / "submit.sh"

    reactant_path.write_text(reactant_xyz)
    product_path.write_text(product_xyz)
    input_path.write_text(
        build_uma_neb_input(
            reactant_name,
            product_name,
            uma_wrapper=uma_wrapper,
            ext_params=ext_params,
            charge=charge,
            multiplicity=multiplicity,
            nprocs=nprocs,
            nimages=nimages,
        )
    )
    submit_path.write_text(
        build_neb_slurm_script(
            job_name=job_name,
            input_name=input_name,
            walltime=walltime,
            ntasks=nprocs,
            pre_command=pre_command,
        )
    )

    return {
        "reactant": reactant_path,
        "product": product_path,
        "inp": input_path,
        "submit": submit_path,
    }


def enumerate_h_abstraction_sites(mol) -> list[NebSite]:
    sites: list[NebSite] = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H":
            continue
        neighbors = list(atom.GetNeighbors())
        if len(neighbors) != 1:
            continue
        parent = neighbors[0]
        parent_symbol = parent.GetSymbol()
        if parent_symbol not in {"C", "N", "O", "S"}:
            continue
        label = f"H{atom.GetIdx()}_from_{parent_symbol}{parent.GetIdx()}"
        sites.append(
            NebSite(
                reaction="h_abstraction",
                atom_index=atom.GetIdx(),
                atom_symbol="H",
                label=label,
                partner_index=parent.GetIdx(),
                partner_symbol=parent_symbol,
            )
        )
    return sites


def enumerate_oh_addition_sites(mol) -> list[NebSite]:
    sites: list[NebSite] = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol == "C":
            include = atom.GetIsAromatic()
            include = include or any(
                bond.GetBondTypeAsDouble() > 1.1 for bond in atom.GetBonds()
            )
        else:
            include = symbol in {"N", "S", "P"}
        if not include:
            continue
        label = f"OH_to_{symbol}{atom.GetIdx()}"
        sites.append(
            NebSite(
                reaction="oh_addition",
                atom_index=atom.GetIdx(),
                atom_symbol=symbol,
                label=label,
            )
        )
    return sites


def ranked_default_sites(
    mol,
    *,
    reaction: str,
    max_sites: int | None = None,
) -> list[NebSite]:
    if reaction == "h_abstraction":
        sites = enumerate_h_abstraction_sites(mol)
    elif reaction == "oh_addition":
        sites = enumerate_oh_addition_sites(mol)
    else:
        raise ValueError(f"Unsupported reaction: {reaction}")
    return sites if max_sites is None else sites[:max_sites]


def _symbols_and_positions_from_mol(mol) -> tuple[list[str], np.ndarray]:
    conf = mol.GetConformer()
    symbols: list[str] = []
    positions: list[list[float]] = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        symbols.append(atom.GetSymbol())
        positions.append([pos.x, pos.y, pos.z])
    return symbols, np.array(positions, dtype=float)


def h_abstraction_endpoints(
    mol,
    hydrogen_index: int,
    *,
    comment_prefix: str = "H abstraction",
) -> tuple[str, str]:
    symbols, positions = _symbols_and_positions_from_mol(mol)
    h_atom = mol.GetAtomWithIdx(hydrogen_index)
    neighbors = list(h_atom.GetNeighbors())
    if len(neighbors) != 1:
        raise ValueError(f"Hydrogen atom {hydrogen_index} must have one neighbor")

    parent_index = neighbors[0].GetIdx()
    parent_pos = positions[parent_index]
    h_pos = positions[hydrogen_index]
    approach = _unit_vector(h_pos - parent_pos)
    side = _perpendicular_unit(approach)

    oh_o_reactant = h_pos + 1.70 * approach
    oh_h_reactant = oh_o_reactant + 0.97 * side
    oh_o_product = h_pos + 0.98 * approach
    transferred_h_product = oh_o_product - 0.98 * approach
    oh_h_product = oh_o_product + 0.97 * side

    reactant_symbols = symbols + ["O", "H"]
    reactant_positions = np.vstack([positions, oh_o_reactant, oh_h_reactant])
    product_positions = positions.copy()
    product_positions[hydrogen_index] = transferred_h_product
    product_positions = np.vstack([product_positions, oh_o_product, oh_h_product])

    reactant = _xyz_text(
        reactant_symbols,
        reactant_positions,
        f"{comment_prefix} reactant: H{hydrogen_index} + OH radical",
    )
    product = _xyz_text(
        reactant_symbols,
        product_positions,
        f"{comment_prefix} product: radical + H2O",
    )
    return reactant, product


def oh_addition_endpoints(
    mol,
    atom_index: int,
    *,
    comment_prefix: str = "OH addition",
) -> tuple[str, str]:
    symbols, positions = _symbols_and_positions_from_mol(mol)
    target = mol.GetAtomWithIdx(atom_index)
    neighbor_positions = [
        positions[neighbor.GetIdx()] for neighbor in target.GetNeighbors()
    ]
    if neighbor_positions:
        outward = _unit_vector(
            positions[atom_index] - np.mean(np.array(neighbor_positions), axis=0)
        )
    else:
        outward = np.array([1.0, 0.0, 0.0])
    side = _perpendicular_unit(outward)

    bond_distance = 1.42 if target.GetSymbol() == "C" else 1.65
    oh_o_product = positions[atom_index] + bond_distance * outward
    oh_h_product = oh_o_product + 0.97 * side
    oh_o_reactant = positions[atom_index] + 2.80 * outward
    oh_h_reactant = oh_o_reactant + 0.97 * side

    all_symbols = symbols + ["O", "H"]
    reactant_positions = np.vstack([positions, oh_o_reactant, oh_h_reactant])
    product_positions = np.vstack([positions, oh_o_product, oh_h_product])

    reactant = _xyz_text(
        all_symbols,
        reactant_positions,
        f"{comment_prefix} reactant: OH radical near {target.GetSymbol()}{atom_index}",
    )
    product = _xyz_text(
        all_symbols,
        product_positions,
        f"{comment_prefix} product: OH adduct at {target.GetSymbol()}{atom_index}",
    )
    return reactant, product


def site_manifest_rows(molecule_name: str, sites: Iterable[NebSite]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for site in sites:
        rows.append(
            {
                "molecule": molecule_name,
                "reaction": site.reaction,
                "atom_index": site.atom_index,
                "atom_symbol": site.atom_symbol,
                "partner_index": site.partner_index,
                "partner_symbol": site.partner_symbol,
                "label": site.label,
            }
        )
    return rows
