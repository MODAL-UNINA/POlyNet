#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Synthetic dataset generation for DeepNMR.

This script generates synthetic polyolefin 13C-NMR spectra from fingerprint
envelope libraries. It loads the class-specific envelope libraries, harmonizes
their spectral domains, builds a common ppm domain, samples synthetic mixture
parameters, and generates the final synthetic dataset through ``utils.create_mixture``.

Generated outputs:
  - <output_dir>/<dataset_name>.pkl
  - <output_dir>/ppm_domain_general.pkl
  - <output_dir>/<dataset_name>_metadata.json
"""

from __future__ import annotations

import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import create_mixture, generate_synthetic_params


# %% FIXED CONFIGURATION

LABELS = ["LDPE", "PE", "PP", "EH", "EO", "EB", "RACO", "EPR"]

DEFAULT_OUTPUT_DIR = "DATASET"
DEFAULT_PPM_DOMAIN_FILENAME = "ppm_domain_general.pkl"


# %% ARGUMENTS


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic DeepNMR spectra from envelope libraries.",
        allow_abbrev=False,
    )

    parser.add_argument(
        "--envelopes_folder",
        type=str,
        default="DATASET/LIBS",
        help="Folder containing the class-specific envelope libraries.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the generated dataset and ppm domain will be saved.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="synthetic_dataset",
        help="Name of the generated dataset, with or without '.pkl'.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Number of synthetic spectra to generate.",
    )
    parser.add_argument(
        "--composition_points",
        type=int,
        default=30,
        help="Number of composition grid points for each copolymer class.",
    )
    parser.add_argument(
        "--n_max_components",
        type=int,
        default=3,
        help="Maximum number of components in each mixture.",
    )
    parser.add_argument(
        "--proportion_weights",
        type=str,
        default="0.40,0.55,0.05",
        help=(
            "Sampling proportions for 1-, 2-, ..., n_max_components-component "
            "spectra. Example: '0.25,0.25,0.25,0.25'."
        ),
    )
    parser.add_argument(
        "--weight_step",
        type=float,
        default=0.02,
        help="Weight-grid step used by generate_synthetic_params.",
    )
    parser.add_argument(
        "--min_weight",
        type=float,
        default=0.04,
        help="Minimum component weight used by generate_synthetic_params.",
    )
    parser.add_argument(
        "--max_weight",
        type=float,
        default=0.96,
        help="Maximum component weight used by generate_synthetic_params.",
    )
    parser.add_argument(
        "--chem_data",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to use chemical-data generation mode: 1=True, 0=False.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=96,
        help="Number of parallel jobs used by create_mixture.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed used before dataset generation.",
    )

    args, _ = parser.parse_known_args()
    return args


# %% UTILITIES


def parse_proportions(proportion_string: str, n_max_components: int) -> list[float]:
    """Parse and validate mixture-type sampling proportions."""
    proportions = [float(value.strip()) for value in proportion_string.split(",")]

    if len(proportions) != n_max_components:
        raise ValueError(
            "The number of mixture-type proportions must be equal to "
            f"n_max_components. Got {len(proportions)} proportions for "
            f"n_max_components={n_max_components}."
        )

    if not np.isclose(sum(proportions), 1.0):
        raise ValueError(
            f"Mixture-type proportions must sum to 1. Got {sum(proportions):.6f}."
        )

    return proportions


def normalized_dataset_name(dataset_name: str) -> str:
    """Return dataset name without the '.pkl' extension."""
    return Path(dataset_name).stem


def load_envelope_libraries(envelopes_folder: str | Path, labels: list[str]) -> dict:
    """Load all class-specific envelope libraries."""
    envelopes_folder = Path(envelopes_folder)
    library = {}

    for copolymer in labels:
        path = envelopes_folder / f"lib_{copolymer}.pkl"

        if not path.exists():
            raise FileNotFoundError(f"Missing envelope library: {path}")

        library[copolymer] = pd.read_pickle(path)

    return library


def harmonize_library_domains(library: dict, labels: list[str]) -> dict:
    """Trim all spectra and ppm domains to a shared number of columns.

    The first column of the ppm-domain DataFrames is skipped because it stores
    the composition coordinate rather than a spectral ppm coordinate.
    """
    reference_columns = min(
        library[copolymer]["ppm_domains"].shape[1] for copolymer in labels
    )

    for copolymer in labels:
        library[copolymer]["ppm_domains"] = library[copolymer]["ppm_domains"].iloc[
            :, 1:reference_columns
        ]
        library[copolymer]["spectra"] = library[copolymer]["spectra"].iloc[
            :, : reference_columns - 1
        ]

    return library


def print_library_shapes(library: dict, labels: list[str]) -> None:
    """Print spectra/domain shapes for consistency checks."""
    print("[INFO] Checking spectra consistency.")

    for copolymer in labels:
        ppm_shape = library[copolymer]["ppm_domains"].shape
        spectra_shape = library[copolymer]["spectra"].shape

        print(
            f"[INFO] {copolymer}: "
            f"ppm_domain shape={ppm_shape} | spectra shape={spectra_shape}"
        )


def build_noise_pool(library: dict, labels: list[str]) -> np.ndarray:
    """Collect real noise segments from the envelope spectra."""
    noise_segments = []

    for copolymer in labels:
        noise_df = library[copolymer]["spectra"].iloc[:, 1:2501]
        noise_segments.append(noise_df)

    noise_pool_df = pd.concat(noise_segments, ignore_index=True)
    return noise_pool_df.values.astype(float)


def build_ppm_domain_general(library: dict, labels: list[str]) -> np.ndarray:
    """Build a shared ppm domain by averaging class-specific ppm domains."""
    ppm_domains = [
        library[copolymer]["ppm_domains"].mean(axis=0).values
        for copolymer in labels
    ]

    return np.mean(np.asarray(ppm_domains), axis=0)


def build_interpolating_functions(library: dict, labels: list[str]) -> dict:
    """Collect class-specific interpolating functions."""
    return {
        copolymer: library[copolymer]["interpolating_func"]
        for copolymer in labels
    }


def build_composition_grid(
    library: dict,
    labels: list[str],
    composition_points: int,
) -> dict[str, list[float]]:
    """Build composition grids for all polymer classes."""
    compositions = {}

    for copolymer in labels:
        spectra_index = library[copolymer]["spectra"].index

        if len(library[copolymer]["spectra"]) > 1 and copolymer != "LDPE":
            compositions[copolymer] = np.linspace(
                min(spectra_index),
                max(spectra_index),
                composition_points,
            ).tolist()
        else:
            first_index = spectra_index[0]

            if isinstance(first_index, float):
                composition_value = first_index
            else:
                composition_value = float(str(first_index).split("_")[0])

            compositions[copolymer] = [composition_value]

    return compositions


def save_metadata(
    output_path: Path,
    *,
    args: argparse.Namespace,
    dataset_path: Path,
    ppm_domain_path: Path,
    labels: list[str],
    proportions: list[float],
) -> None:
    """Save generation metadata for reproducibility."""
    metadata = {
        "dataset_path": str(dataset_path),
        "ppm_domain_path": str(ppm_domain_path),
        "labels": labels,
        "envelopes_folder": args.envelopes_folder,
        "n_samples": args.n_samples,
        "composition_points": args.composition_points,
        "n_max_components": args.n_max_components,
        "proportion_weights": proportions,
        "weight_step": args.weight_step,
        "weight_bounds": [args.min_weight, args.max_weight],
        "chem_data": bool(args.chem_data),
        "n_jobs": args.n_jobs,
        "random_seed": args.random_seed,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


# %% MAIN


def main() -> None:
    """Generate and save a synthetic DeepNMR dataset."""
    args = parse_args()

    np.random.seed(args.random_seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = normalized_dataset_name(args.dataset_name)
    dataset_path = output_dir / f"{dataset_name}.pkl"
    ppm_domain_path = output_dir / DEFAULT_PPM_DOMAIN_FILENAME
    metadata_path = output_dir / f"{dataset_name}_metadata.json"

    proportions = parse_proportions(
        args.proportion_weights,
        n_max_components=args.n_max_components,
    )

    print("[INFO] Loading envelope libraries.")
    library = load_envelope_libraries(args.envelopes_folder, LABELS)

    print("[INFO] Harmonizing ppm domains and spectra.")
    library = harmonize_library_domains(library, LABELS)
    print_library_shapes(library, LABELS)

    interp_dict = build_interpolating_functions(library, LABELS)
    noise_pool = build_noise_pool(library, LABELS)

    ppm_domain_general = build_ppm_domain_general(library, LABELS)

    with open(ppm_domain_path, "wb") as f:
        pickle.dump(ppm_domain_general, f)

    print(f"[INFO] Saved shared ppm domain: {ppm_domain_path}")

    print("[INFO] Building composition grid.")
    compositions = build_composition_grid(
        library,
        LABELS,
        composition_points=args.composition_points,
    )

    print("[INFO] Generating synthetic mixture parameters.")
    weights, compositions_matrix = generate_synthetic_params(
        copolymer_list=LABELS,
        compositions=compositions,
        max_components=args.n_max_components,
        weight_step=args.weight_step,
        weight_bounds=(args.min_weight, args.max_weight),
    )

    df_w = pd.DataFrame(np.round(weights.astype(np.float32), 2), columns=LABELS)
    df_c = pd.DataFrame(np.round(compositions_matrix.astype(np.float32), 2), columns=LABELS)

    print("[INFO] Generating synthetic spectra.")
    synthetic_dataset = create_mixture(
        df_w,
        df_c,
        portion=proportions,
        max_components=args.n_max_components,
        column_labels=LABELS,
        interp_dict=interp_dict,
        ppm_domain_general=ppm_domain_general,
        noise_pool=noise_pool,
        n_mixture=args.n_samples,
        n_jobs=args.n_jobs,
        chem_data=bool(args.chem_data),
    )

    with open(dataset_path, "wb") as f:
        pickle.dump(synthetic_dataset, f)

    save_metadata(
        metadata_path,
        args=args,
        dataset_path=dataset_path,
        ppm_domain_path=ppm_domain_path,
        labels=LABELS,
        proportions=proportions,
    )

    print(f"[INFO] Dataset saved: {dataset_path}")
    print(f"[INFO] Metadata saved: {metadata_path}")
    print("[INFO] DONE.")


if __name__ == "__main__":
    main()