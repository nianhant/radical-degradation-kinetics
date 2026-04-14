from __future__ import annotations

from pathlib import Path


def build_orca_input(
    xyz_filename: str,
    *,
    functional: str = "wB97X-V",
    basis: str = "def2-TZVP",
    solvation: str = "CPCM(water)",
    charge: int = 0,
    multiplicity: int = 1,
) -> str:
    header = f"! {functional} {basis} TightSCF"
    if solvation:
        header = f"{header} {solvation}"

    return (
        f"{header}\n"
        "%pal nprocs 16 end\n"
        "%maxcore 2000\n\n"
        f"* xyzfile {charge} {multiplicity} {xyz_filename}\n"
    )


def build_slurm_script(
    *,
    job_name: str,
    input_name: str = "job.inp",
    walltime: str = "12:00:00",
    nodes: int = 1,
    ntasks: int = 16,
) -> str:
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
        "module load orca\n"
        f"orca {input_name} > job.out\n"
    )


def write_orca_job(
    output_dir: str | Path,
    xyz_text: str,
    *,
    xyz_name: str = "structure.xyz",
    input_name: str = "job.inp",
    job_name: str = "pharma_dft",
    functional: str = "wB97X-V",
    basis: str = "def2-TZVP",
    solvation: str = "CPCM(water)",
    charge: int = 0,
    multiplicity: int = 1,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xyz_path = output_dir / xyz_name
    inp_path = output_dir / input_name
    submit_path = output_dir / "submit.sh"

    xyz_path.write_text(xyz_text)
    inp_path.write_text(
        build_orca_input(
            xyz_name,
            functional=functional,
            basis=basis,
            solvation=solvation,
            charge=charge,
            multiplicity=multiplicity,
        )
    )
    submit_path.write_text(build_slurm_script(job_name=job_name, input_name=input_name))

    return {
        "xyz": xyz_path,
        "inp": inp_path,
        "submit": submit_path,
    }
