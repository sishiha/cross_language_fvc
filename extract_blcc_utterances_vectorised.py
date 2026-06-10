#!/usr/bin/env python3
"""
Extract BLCC matrices from full-band cepstral coefficients stored in .npz files.

Typical use:
  ./extract_blcc_utterances.py \
    --indir  ./australian_english_database/female_int_selected_ccs_16k_n12000/round1 \
    --outdir ./australian_english_database/female_int_selected_blccs_16k_n12000/round1 \
    --omega1 0 --omega2 600 --sampf 16000 --n_ccs 14

Notes:
- Speaker/session subdirectories are expected under --indir.
- For each .npz, the script uses --npz_key if provided; otherwise it uses the first array in the file.
- Output .npz contains: blcc (matrix), omega1, omega2, sampf, source_key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm


def add_blcc_tools_to_path() -> None:
    """Add ./blcc_tools (relative to this script) to sys.path."""
    script_dir = Path(__file__).resolve().parent
    blcc_tools_dir = script_dir / "blcc_tools"
    if not blcc_tools_dir.exists():
        raise FileNotFoundError(f"Cannot find blcc_tools directory: {blcc_tools_dir}")
    sys.path.insert(0, str(blcc_tools_dir))


def list_immediate_subdirectories(directory_path: Path) -> List[Path]:
    """Return immediate subdirectories (not recursive), sorted by name."""
    return sorted([p for p in directory_path.iterdir() if p.is_dir()])


def load_fbcc_matrix(npz_path: Path, npz_key: Optional[str]) -> tuple[np.ndarray, str]:
    """
    Load a 2D matrix of full-band CCs from an .npz file.
    Returns (matrix, used_key).
    """
    with np.load(npz_path) as data:
        keys = list(data.files)
        if not keys:
            raise ValueError(f"No arrays found in: {npz_path}")

        if npz_key:
            if npz_key not in keys:
                raise KeyError(f"Key '{npz_key}' not found in {npz_path}. Available: {keys}")
            key = npz_key
        else:
            key = keys[0]  # keep original behaviour

        mat = data[key]

    if mat.ndim != 2:
        raise ValueError(f"Expected a 2D matrix for key '{key}' in {npz_path}, got shape {mat.shape}")

    return mat, key


def make_blcc_transform_matrix(
    n_fullband_ccs: int,
    omega1_hz: int,
    omega2_hz: int,
    sf_hz: int,
) -> np.ndarray:
    """
    Create the BLCC transformation matrix used by blcc_main_functions.blcc().

    If the full-band CC matrix has shape (n_rows, n_fullband_ccs), the
    returned matrix has shape (n_fullband_ccs + 1, n_fullband_ccs). The BLCC
    matrix can then be computed as:

        blcc_matrix = fbccs @ A.T

    This is equivalent to applying blcc() to each row, but avoids rebuilding the
    same transformation matrix for every row.
    """
    if n_fullband_ccs < 1:
        raise ValueError("n_fullband_ccs must be at least 1.")

    omega1 = omega1_hz * 2.0 * np.pi / sf_hz
    omega2 = omega2_hz * 2.0 * np.pi / sf_hz
    half_sf = np.pi

    if np.isclose(omega2, omega1):
        raise ValueError("omega2 must differ from omega1.")

    # L indexes output BLCC coefficients, including c0.
    # K indexes input full-band cepstral coefficients, starting from 1.
    L = np.arange(0, n_fullband_ccs + 1, dtype=float)[:, None]  # (N, 1)
    K = np.arange(1, n_fullband_ccs + 1, dtype=float)[None, :]  # (1, M)

    W = (omega2 - omega1) / half_sf
    A = np.empty((n_fullband_ccs + 1, n_fullband_ccs), dtype=float)

    # l == 0 row
    k = K.ravel()
    A[0, :] = (
        np.sin(k * omega2) - np.sin(k * omega1)
    ) / (
        k * (omega2 - omega1)
    )

    # l > 0 rows
    L_pos = L[1:, :]
    denom = L_pos**2 - (K * W)**2

    with np.errstate(divide="ignore", invalid="ignore"):
        part1 = (2.0 * K * W) / (half_sf * denom)
        part2 = ((-1.0) ** (L_pos + 1.0)) * np.sin(K * omega2) + np.sin(K * omega1)
        A[1:, :] = part1 * part2

    # Preserve the original scalar implementation's special case: l == k * W.
    # A tolerance is safer than exact float equality after vectorisation.
    singular = np.isclose(
        denom,
        0.0,
        rtol=0.0,
        atol=np.finfo(float).eps * 100,
    )
    if np.any(singular):
        replacement = np.broadcast_to(np.cos(K * omega1), denom.shape)
        A[1:, :][singular] = replacement[singular]

    return A


def extract_blcc_for_file(
    infile: Path,
    outfile: Path,
    omega1: int,
    omega2: int,
    sampf: int,
    npz_key: Optional[str],
) -> None:
    fbccs, used_key = load_fbcc_matrix(infile, npz_key)
    fbccs = np.asarray(fbccs, dtype=float)

    A = make_blcc_transform_matrix(
        n_fullband_ccs=fbccs.shape[1],
        omega1_hz=omega1,
        omega2_hz=omega2,
        sf_hz=sampf,
    )

    # Vectorised equivalent of applying blcc(row, omega1, omega2, sampf) to
    # every row in fbccs and stacking the result.
    blcc_matrix = fbccs @ A.T

    outfile.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        outfile,
        blcc=blcc_matrix,
        omega1=np.array([omega1], dtype=int),
        omega2=np.array([omega2], dtype=int),
        sampf=np.array([sampf], dtype=int),
        source_key=np.array([used_key]),
    )

def main() -> int:
    ap = argparse.ArgumentParser(description="Extract BLCCs from .npz full-band CC matrices.")
    ap.add_argument("--indir", type=Path, required=True, help="Input root directory containing subdirectories of .npz files")
    ap.add_argument("--outdir", type=Path, required=True, help="Output root directory (mirrors input subdirectory structure)")
    ap.add_argument("--omega1", type=int, default=0, help="Lower cutoff frequency (Hz)")
    ap.add_argument("--omega2", type=int, default=600, help="Upper cutoff frequency (Hz)")
    ap.add_argument("--sampf", type=int, default=16000, help="Sampling frequency (Hz)")
    ap.add_argument("--n_ccs", type=int, default=14, help="Number of CCs (used only for mw() check)")
    ap.add_argument("--npz_key", type=str, default=None, help="Key inside .npz to use (e.g., LFCC/MFCC). Default: first array.")
    ap.add_argument("--pattern", type=str, default="*.npz", help="File glob pattern within each subdirectory (default: *.npz)")
    ap.add_argument("--gender", type=str, default="male", choices=["male", "female"], help="Gender (default: male)")

    # Subdirectory selection
    ap.add_argument("--subdir_start", type=int, default=0, help="Start index (inclusive) of sorted subdirectories to process")
    ap.add_argument("--subdir_end", type=int, default=None, help="End index (exclusive) of sorted subdirectories to process")

    # Behaviour
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output files if they already exist")
    args = ap.parse_args()

    if not args.indir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.indir}")

    add_blcc_tools_to_path()
    import blcc_main_functions as blcc_functions  # noqa: E402

    # Optional sanity: compute mw (your original script did this; value was unused)
    _mw_value = blcc_functions.mw(args.omega1, args.omega2, args.sampf, args.n_ccs, rounded=True)

    subdirs = list_immediate_subdirectories(args.indir)
    subdirs = subdirs[args.subdir_start : args.subdir_end]

    if not subdirs:
        raise RuntimeError("No subdirectories found to process (check --indir and --subdir_start/--subdir_end).")

    for subdir in subdirs:
        rel = subdir.name
        in_subdir = subdir
        out_subdir = args.outdir / rel
        out_subdir.mkdir(parents=True, exist_ok=True)

        npz_files = sorted(in_subdir.glob(args.pattern))
        if not npz_files:
            print(f"Warning: no files matched {args.pattern} in {in_subdir}")
            continue

        for infile in tqdm(npz_files, desc=f"Processing {rel}", unit="file"):
            outfile = out_subdir / f"{infile.stem}_omega1_{args.omega1}_omega2_{args.omega2}.npz"

            if outfile.exists() and not args.overwrite:
                continue

            extract_blcc_for_file(
                infile=infile,
                outfile=outfile,
                omega1=args.omega1,
                omega2=args.omega2,
                sampf=args.sampf,
                npz_key=args.npz_key,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
