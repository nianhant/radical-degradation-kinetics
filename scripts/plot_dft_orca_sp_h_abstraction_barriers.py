#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_SUMMARY = Path(
    "/global/homes/n/nianhant/data/pharma_degradation/ts_search/"
    "orca_cpu_sp_on_uma_h_abstraction_paths/analysis/orca_sp_on_uma_path_summary.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/global/homes/n/nianhant/data/pharma_degradation/ts_search/"
    "orca_cpu_sp_on_uma_h_abstraction_paths/analysis/dft_bond_barrier_plots"
)

COVALENT_RADII = {
    "H": 0.31,
    "B": 0.85,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}

ATOM_COLORS = {
    "H": "#d6d6d6",
    "C": "#30343b",
    "N": "#2563eb",
    "O": "#dc2626",
    "F": "#16a34a",
    "P": "#d97706",
    "S": "#ca8a04",
    "Cl": "#22c55e",
    "Br": "#92400e",
    "I": "#7c3aed",
}


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "plot"


def finite_float(text: str) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def parse_label(label: str) -> tuple[int, str, int, str]:
    match = re.fullmatch(r"([A-Z][a-z]?)(\d+)_from_([A-Z][a-z]?)(\d+)", label)
    if match is None:
        raise ValueError(f"Unsupported H-abstraction label: {label}")
    atom_symbol, atom_index, partner_symbol, partner_index = match.groups()
    return int(atom_index), atom_symbol, int(partner_index), partner_symbol


def read_first_xyz_frame(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(errors="replace").splitlines()
    natoms = int(lines[0].strip())
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : 2 + natoms]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.array(coords, dtype=float)


def infer_bonds(symbols: list[str], coords: np.ndarray) -> list[tuple[int, int]]:
    bonds: list[tuple[int, int]] = []
    for i, symbol_i in enumerate(symbols):
        for j in range(i + 1, len(symbols)):
            symbol_j = symbols[j]
            cutoff = 1.25 * (
                COVALENT_RADII.get(symbol_i, 0.77)
                + COVALENT_RADII.get(symbol_j, 0.77)
            )
            if np.linalg.norm(coords[i] - coords[j]) <= cutoff:
                bonds.append((i, j))
    return bonds


def project_to_2d(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=0)
    if len(coords) < 3:
        out = np.zeros((len(coords), 2))
        out[:, 0] = centered[:, 0]
        return out
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def load_rows(summary: Path, require_success: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with summary.open() as handle:
        for row in csv.DictReader(handle):
            if require_success and row.get("orca_status") != "success":
                continue
            barrier = finite_float(row.get("dft_barrier_kcal_mol", ""))
            if barrier is None:
                continue
            trajectory = row.get("trajectory_xyz", "")
            if not trajectory or not Path(trajectory).exists():
                continue
            atom_index, atom_symbol, partner_index, partner_symbol = parse_label(row["label"])
            row["atom_index"] = str(atom_index)
            row["atom_symbol"] = atom_symbol
            row["partner_index"] = str(partner_index)
            row["partner_symbol"] = partner_symbol
            rows.append(row)
    return rows


def atom_name(symbol: str, index: str | int) -> str:
    return f"{symbol}{int(index):04d}"


def bond_label(row: dict[str, str]) -> str:
    partner = atom_name(row["partner_symbol"], row["partner_index"])
    atom = atom_name(row["atom_symbol"], row["atom_index"])
    return f"{partner}-{atom}"


def x_tick_label(row: dict[str, str]) -> str:
    return f"{row['label']}\n{bond_label(row)}\n{row['partner_symbol']}-H"


def draw_molecule(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    symbols, coords = read_first_xyz_frame(Path(rows[0]["trajectory_xyz"]))
    # The NEB trajectory includes the incoming OH radical as the final two atoms.
    molecule_atom_limit = max(0, len(symbols) - 2)
    symbols = symbols[:molecule_atom_limit]
    coords = coords[:molecule_atom_limit]
    xy = project_to_2d(coords)
    bonds = infer_bonds(symbols, coords)
    highlight_pairs = {
        tuple(sorted((int(row["atom_index"]), int(row["partner_index"])))) for row in rows
    }
    highlight_atoms = {index for pair in highlight_pairs for index in pair}

    for i, j in bonds:
        pair = tuple(sorted((i, j)))
        highlighted = pair in highlight_pairs
        ax.plot(
            [xy[i, 0], xy[j, 0]],
            [xy[i, 1], xy[j, 1]],
            color="#e11d48" if highlighted else "#a3a3a3",
            linewidth=2.5 if highlighted else 0.9,
            linestyle="--" if highlighted else "-",
            zorder=2 if highlighted else 1,
        )

    for index, (symbol, point) in enumerate(zip(symbols, xy)):
        ax.scatter(
            point[0],
            point[1],
            s=26 if symbol == "H" else 68,
            color=ATOM_COLORS.get(symbol, "#737373"),
            edgecolor="#ffffff",
            linewidth=0.55,
            zorder=3,
        )
        if index in highlight_atoms or symbol != "H":
            ax.text(
                point[0],
                point[1] + 0.07,
                f"{symbol}{index:04d}",
                fontsize=5.6 if index in highlight_atoms else 4.4,
                ha="center",
                va="bottom",
                color="#111827",
                zorder=4,
            )

    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")


def plot_group(rows: list[dict[str, str]], output_dir: Path, pdf: PdfPages) -> Path:
    rows = sorted(rows, key=lambda row: (float(row["dft_barrier_kcal_mol"]), row["label"]))
    title = f"{rows[0]['dataset']} / {rows[0]['molecule']}"
    labels = [x_tick_label(row) for row in rows]
    values = [float(row["dft_barrier_kcal_mol"]) for row in rows]

    width = max(9.5, 0.78 * len(rows) + 3.0)
    fig, (mol_ax, bar_ax) = plt.subplots(
        2,
        1,
        figsize=(width, 8.4),
        gridspec_kw={"height_ratios": [1.25, 2.0]},
        constrained_layout=True,
    )
    draw_molecule(mol_ax, rows)
    mol_ax.set_title(title, fontsize=12)

    x = np.arange(len(rows))
    bars = bar_ax.bar(
        x,
        values,
        color=["#3b82f6" if row.get("orca_status") == "success" else "#93c5fd" for row in rows],
        edgecolor="#111827",
        linewidth=0.45,
    )
    for bar, row in zip(bars, rows):
        if row.get("orca_status") != "success":
            bar.set_hatch("///")
    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    bar_ax.set_ylabel("DFT ORCA barrier (kcal/mol)")
    bar_ax.set_xlabel("Bond label from molecule structure")
    bar_ax.grid(axis="y", color="#d4d4d4", linewidth=0.7, alpha=0.65)
    bar_ax.spines["top"].set_visible(False)
    bar_ax.spines["right"].set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#3b82f6", ec="#111827", lw=0.45),
        plt.Rectangle((0, 0), 1, 1, color="#93c5fd", ec="#111827", lw=0.45, hatch="///"),
    ]
    bar_ax.legend(handles, ["complete ORCA image profile", "partial ORCA image profile"], frameon=False, fontsize=8)

    out_path = output_dir / f"dft_orca_h_abstraction_bond_barriers_{slug(title)}.png"
    fig.savefig(out_path, dpi=220)
    pdf.savefig(fig)
    plt.close(fig)
    return out_path


def write_filtered_csv(rows: list[dict[str, str]], output_dir: Path) -> Path:
    path = output_dir / "dft_orca_h_abstraction_bond_barriers_plotted.csv"
    fields = [
        "dataset",
        "molecule",
        "base_molecule",
        "label",
        "partner_symbol",
        "atom_index",
        "atom_symbol",
        "partner_index",
        "dft_barrier_kcal_mol",
        "dft_barrier_ev",
        "dft_transition_image",
        "dft_reaction_energy_kcal_mol",
        "orca_status",
        "orca_status_reason",
        "reliable_comparison",
        "trajectory_xyz",
        "orca_job_dir",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot DFT ORCA single-point H-abstraction barriers along selected NEB paths "
            "with highlighted molecule bonds."
        )
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Keep only rows with complete ORCA image coverage. Default includes every finite DFT barrier.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.summary, require_success=args.success_only)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["molecule"])].append(row)

    pdf_path = args.output_dir / "dft_orca_h_abstraction_bond_barriers_by_molecule.pdf"
    with PdfPages(pdf_path) as pdf:
        for _, group_rows in sorted(grouped.items()):
            plot_group(group_rows, args.output_dir, pdf)
    csv_path = write_filtered_csv(rows, args.output_dir)

    print(f"Loaded {len(rows)} DFT ORCA rows from {args.summary}")
    print(f"Wrote {len(grouped)} molecule plots to {args.output_dir}")
    print(f"Wrote combined PDF to {pdf_path}")
    print(f"Wrote plotted CSV to {csv_path}")


if __name__ == "__main__":
    main()
