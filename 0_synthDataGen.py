#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 06 12:12:38 2024

@author: MODAL
"""

# %% IMPORT SECTION

import pickle
import argparse
import numpy as np
import pandas as pd
from fastcore.all import dict2obj

from utils import create_mixture, generate_synthetic_params

# %% PATHS AND OPTIONS
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NMR training script with overrides")

    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Size of the synthetic dataset to be generated."
    )
    parser.add_argument(
        "--composition_points",
        type=int,
        default=30,
        help="Number of composition points for each copolymer in the dataset."
    )
    parser.add_argument(
        "--n_max_components",
        type=int,
        default=2,
        help="Maximum number of components in each mixture."
    )
    parser.add_argument(
        "--proportion_weights",
        type=str,
        default="0.4, 0.6",
        help="Proportions of weights for the components in the mixtures e.g. '0.4, 0.6'. "
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="synthetic_dataset",
        help="Name of the synthetic dataset to be generated."
    )
    # You can add as many argument definitions as you want

    args, _ = parser.parse_known_args()

    ENVELOPES_FOLDER = "DATASET/LIBS"
    synthetic_dataset_name = args.dataset_name

    opts = dict2obj(
        dict(
            copolymer_list = ["LDPE", "PE", "PP", "EH", "EO", "EB", "RACO", "EPR"],
            composition_points = args.composition_points,
            n_mixture = args.n_samples,
            n_max_components = args.n_max_components,
            weight_type_mix = [float(k) for k in args.proportion_weights.split(",")],
        )
    )

    assert (
        len(opts.weight_type_mix) == opts.n_max_components
    ), "The number of weights must be equal to the number of components"

    # %% LIBRARY DATA IMPORT and MANIPULATION

    library = {}
    for copolymer in opts.copolymer_list:
        library[copolymer] = pd.read_pickle(f"{ENVELOPES_FOLDER}/lib_{copolymer}.pkl")

    # Correction for inhomogeneous domains
    reference_columns = min(
        library[copolymer]["ppm_domains"].shape[1] for copolymer in opts.copolymer_list
    )

    # Correction of accounting the presence of composition column in ppm dataframes
    for copolymer in opts.copolymer_list:
        library[copolymer]["ppm_domains"] = library[copolymer]["ppm_domains"].iloc[
            :, 1:reference_columns
        ]
        library[copolymer]["spectra"] = library[copolymer]["spectra"].iloc[
            :, : reference_columns - 1
        ]

    # Check if the spectra have the same number of columns
    print("Checking spectra consistency...")
    for copolymer in opts.copolymer_list:
        print(
            f"{copolymer} -> domain shape: {library[copolymer]["ppm_domains"].shape}| "
            f"spectra shape: {library[copolymer]["spectra"].shape}"
        )

    # Isolate the interpolating functions
    interp_dict = {
        k: v
        for k, v in zip(
            opts.copolymer_list,
            (library[k]["interpolating_func"] for k in opts.copolymer_list),
        )
    }

    # Isolate ppm domains for the copolymers
    ppm_domains_dict = {
        k: v
        for k, v in zip(
            opts.copolymer_list,
            (library[k]["ppm_domains"] for k in opts.copolymer_list),
        )
    }

    # Generate a general ppm domain
    ppm_domain_general = np.mean(
        np.array([ppm_domains_dict[k].mean(axis=0) for k in ppm_domains_dict.keys()]),
        axis=0,
    )

    # %% GENERATE SYNTHETIC DATA

    # Generate the synthetic composition points
    comps_dict = {
        key: (
            np.linspace(
                min(library[key]["spectra"].index),
                max(library[key]["spectra"].index),
                opts.composition_points,
            ).tolist()
            if len(library[key]["spectra"]) > 1 and (key not in ["LDPE"])
            else [
                (
                    library[key]["spectra"].index[0]
                    if isinstance(library[key]["spectra"].index[0], float)
                    else float(library[key]["spectra"].index[0].split("_")[0])
                )
            ]
        )
        for key in opts.copolymer_list
    }

    # Generate the synthetic parameters
    w, c = generate_synthetic_params(
        copolymer_list=opts.copolymer_list,
        compositions=comps_dict,
        max_components=2,
        weight_step=0.02,
        weight_bounds=(0.1, 0.9),
    )

    # Collect the generated parameters into DataFrames
    df_w = pd.DataFrame(np.round(w.astype(np.float32), 2), columns=opts.copolymer_list)
    df_c = pd.DataFrame(np.round(c.astype(np.float32), 2), columns=opts.copolymer_list)

    # Generate the synthetic dataset
    synthetic_dataset = create_mixture(
        df_w,
        df_c,
        portion=opts.weight_type_mix,
        max_components=opts.n_max_components,
        column_labels=opts.copolymer_list,
        interp_dict=interp_dict,
        ppm_domain_general=ppm_domain_general,
        n_mixture=opts.n_mixture,
        n_jobs=96,
    )

    # %% SAVE THE DATASET (PICKLE)

    with open(f"DATASET/{synthetic_dataset_name}.pkl", "wb") as f:
        pickle.dump(synthetic_dataset, f)
    print(f"Dataset saved as {synthetic_dataset_name}.pkl")
    print("DONE!")
