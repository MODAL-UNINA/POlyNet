#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 06 16:22:08 2024

@author: MODAL
"""

# %% IMPORT SECTION
import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from fastcore.all import dict2obj
from utils.mappings import MappingNames
from mpl_toolkits.mplot3d import Axes3D 

# %% OVERALL PARAMETERS

opts = dict2obj(
    dict(
        copolymer_list=["LDPE", "PE", "PP", "EH", "EO", "EB", "RACO", "EPR"],
        synthetic_dataset_name="syntetic_dataset",
        test_dataset_name="test_data",
    )
)

LIBRARY_PATH = f"DATASET/LIBS"

DATA_PATH = f"DATASET/{opts.synthetic_dataset_name}.pkl"
TEST_PATH = f"DATASET/{opts.test_dataset_name}.pkl"

SAVE_PATH = f"OUTPUT/synth_data_analysis/{opts.synthetic_dataset_name}"
os.makedirs(SAVE_PATH, exist_ok=True)

#%% IMPORT DATASETS

library = {}
for copolymer in opts.copolymer_list:
    library[copolymer] = pd.read_pickle(f"{LIBRARY_PATH}/lib_{copolymer}.pkl")

# Correction for inhomogeneous domains
reference_columns = min(
    library[copolymer]["ppm_domains"].shape[1] for copolymer in opts.copolymer_list
)

for copolymer in opts.copolymer_list:
    library[copolymer]["ppm_domains"] = library[copolymer]["ppm_domains"].iloc[
        :, :reference_columns
    ]
    library[copolymer]["spectra"] = library[copolymer]["spectra"].iloc[
        :, :reference_columns-1 # Recall that spectra have no column "c"
    ]

# %% DATA IMPORT AND MANIPULATION
synth_data = pd.read_pickle(f"{DATA_PATH}")
X, y_w, y_c = synth_data[0], synth_data[3], synth_data[4]
y_p = np.where(y_w != 0, 1, 0)

test_data = pd.read_pickle(f"{TEST_PATH}")

# %% SINGLE COMPONENTS ANALYSIS

single_idxs = np.where(np.sum(y_p, axis=1) == 1)[0]

y_w_single = y_w[single_idxs]
y_c_single = y_c[single_idxs]
y_p_single = y_p[single_idxs]

y_c_single *= y_p_single

# Plot of number of samples per copolymer
plt.figure(figsize=(15, 8))
plt.barh(MappingNames()[opts.copolymer_list], y_p_single.sum(axis=0), color="skyblue")
plt.title("Number of samples per copolymer")
plt.xlabel("# Samples")
plt.ylabel("Copolymer")
plt.tight_layout()
plt.savefig(f"{SAVE_PATH}/samples_per_copolymer.png", dpi=150)
plt.show()


n_materials = y_c_single.shape[1]
# Compute the number of rows and columns for subplots
ncols = int(np.ceil(np.sqrt(n_materials)))
nrows = int(np.ceil(n_materials / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4))
axes = axes.flatten()  # Flatten the array of axes for easy indexing

for i in range(n_materials):
    # Filter out zero entries for the current copolymer
    y_c_single_plot = y_c_single[:, i][y_c_single[:, i] != 0.0]
    ax = axes[i]  # Get the corresponding subplot
    # Plot the histogram
    ax.hist(y_c_single_plot, bins=30, alpha=0.5)
    ax.set_title(f"Composition Frequency for {opts.copolymer_list[i]}")
    # ax.legend()

# Remove any unused subplots if n_materials is not a perfect square
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()  # Adjust subplots to fit in the figure area
plt.savefig(f"{SAVE_PATH}/comps_per_copolymer.png", dpi=150)
plt.show()

# %% MIXTURE ANALYSIS

multi_indexes = np.where(np.sum(y_p, axis=1) == 2)[0]

y_w_multi = y_w[multi_indexes]
y_c_multi = y_c[multi_indexes]
y_p_multi = y_p[multi_indexes]

# assert np.all(np.sum(y_p_multi, axis=1) == 2), "Not all samples are mixtures"

y_c_multi *= y_p_multi

# Plot of number of samples per mixture
mixtures_list = []
for i in range(len(y_p_multi)):
    mixtures_list.append(
        tuple(
            [
                label
                for k, label in enumerate(MappingNames()[opts.copolymer_list])
                if y_p_multi[i][k] == 1
            ]
        )
    )

sorted_mixtures_list = [tuple(sorted(mixture)) for mixture in mixtures_list]
mixture_counts = Counter(sorted_mixtures_list)

# Sort mixtures by frequency
mixtures_counts_sorted = mixture_counts.most_common()

# Separate mixtures and counts
mixtures_sorted = [mixture for mixture, _ in mixtures_counts_sorted]
counts_sorted = [count for _, count in mixtures_counts_sorted]

# Create labels for the x-axis
mixture_labels = [" & ".join(mixture) for mixture in mixtures_sorted]

# Plotting
plt.figure(figsize=(12, 6))
plt.barh(mixture_labels, counts_sorted, color="skyblue")
plt.xticks(rotation=45, ha="right")
plt.xlabel("# Samples")
plt.ylabel("Mixtures")
plt.title("Number of samples per Mixtures")
plt.tight_layout()
plt.savefig(f"{SAVE_PATH}/samples_per_mixture.png", dpi=150)
plt.show()

# Analyze the frequencies of weights in the mixtures
n_materials = y_w_multi.shape[1]

# Compute the number of rows and columns for subplots
ncols = int(np.ceil(np.sqrt(n_materials)))
nrows = int(np.ceil(n_materials / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4))
axes = axes.flatten()  # Flatten the array of axes for easy indexing

for i in range(n_materials):
    # Filter out zero entries for the current copolymer
    y_w_multi_plot = y_w_multi[:, i][y_p_multi[:, i] != 0.0]
    ax = axes[i]  # Get the corresponding subplot
    # Plot the histogram
    ax.hist(y_w_multi_plot, bins=100, alpha=0.5)
    ax.set_title(f"Weights Frequency for {MappingNames()[opts.copolymer_list[i]]}")
    # ax.legend()

# Remove any unused subplots if n_materials is not a perfect square
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()  # Adjust subplots to fit in the figure area
plt.savefig(f"{SAVE_PATH}/weights_per_mixture.png", dpi=150)
plt.show()

#  Analyze the frequencies of compositions in the mixtures
n_materials = y_c_multi.shape[1]

# Compute the number of rows and columns for subplots
ncols = int(np.ceil(np.sqrt(n_materials)))
nrows = int(np.ceil(n_materials / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4))
axes = axes.flatten()  # Flatten the array of axes for easy indexing

for i in range(n_materials):
    # Filter out zero entries for the current copolymer
    y_c_multi_plot = y_c_multi[:, i][y_p_multi[:, i] != 0.0]
    ax = axes[i]  # Get the corresponding subplot
    # Plot the histogram
    ax.hist(y_c_multi_plot, bins=30, alpha=0.5)
    ax.set_title(f"Composition Frequency for {MappingNames()[opts.copolymer_list[i]]}")
    # ax.legend()

# Remove any unused subplots if n_materials is not a perfect square
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()  # Adjust subplots to fit in the figure area
plt.savefig(f"{SAVE_PATH}/comps_per_mixture.png", dpi=150)
plt.show()

# %% COMPARISON WITH TEST DATA

for copo in opts.copolymer_list:
    composition_threshold = 0.2 if copo != "EPR" else 1.0

    ppm_domain = library[copo]["ppm_domains"].iloc[0].values[1:]
    lib_data_copo = library[copo]["spectra"]
    lib_data_interp = library[copo]["interpolates"]

    # Filter rows in synth_data that correspond to this copolymer
    synth_data_copo = synth_data[0][
        synth_data[3][:, opts.copolymer_list.index(copo)] == 1
    ]
    synth_data_comp = synth_data[4][
        synth_data[3][:, opts.copolymer_list.index(copo)] == 1
    ][:, opts.copolymer_list.index(copo)]

    synth_data_copo = pd.DataFrame(synth_data_copo, columns=lib_data_copo.columns)
    synth_data_copo.insert(0, "c", synth_data_comp)

    # Filter test_data for this copolymer
    if copo == "PP_RACO":
        test_data_pp = test_data[test_data["copolymer"] == "PP"].copy()
        test_data_raco = test_data[test_data["copolymer"] == "RACO"].copy()
        test_data_copo = pd.concat([test_data_pp, test_data_raco], ignore_index=True)

        test_data_copo.loc[test_data_copo["copolymer"] == "PP", "c"].apply(lambda x: (0.,))
    else:
        test_data_copo = test_data[test_data["copolymer"] == copo].copy()
    
    # Convert the column c drom tuples to floats
    test_data_copo.loc[:,"c"] = test_data_copo.loc[:,"c"].apply(lambda x: x[0])

    # Number of total plots (one for each entry in lib_data_copo)
    comps = lib_data_copo.index
    n = len(comps)

    # Set up subplots: 2 columns, enough rows
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3 * nrows))
    axes = axes.flatten()  # Flatten so we can index in 1D

    for i, comp in enumerate(comps):
        ax = axes[i]

        # Extract the 4 spectra
        lib_spectrum = lib_data_copo.loc[comp].values
        interp_spectrum = lib_data_interp.loc[comp].values

        # Find the row in synth_data_copo close to comp
        # np.isclose(..., atol=0.2) picks any row whose 'c' is within 0.2
        row_synth = synth_data_copo[
            np.isclose(synth_data_copo["c"], comp if isinstance(comp, float) else float(comp.split("_")[0]), atol=composition_threshold)
        ]
        if not row_synth.empty:
            synth_spectrum = row_synth.loc[[np.random.choice(row_synth.index)]].values[0,1:]  # skip the 'c' column
        else:
            synth_spectrum = np.zeros_like(lib_spectrum)  # or handle differently

        # Find row in test_data_copo
        row_test = test_data_copo[
            np.isclose(test_data_copo["c"].astype(np.float32), comp if isinstance(comp, float) else float(comp.split("_")[0]), atol=composition_threshold)
        ]
        if not row_test.empty:
            test_spectrum = row_test.loc[[np.random.choice(row_test.index)]].values[0,4:]  # skip 'copolymer' and 'c'
        else:
            test_spectrum = np.zeros_like(lib_spectrum)  # or handle differently

        # Plot them
        ax.vlines(
            library[copo]["legit_peaks"],
            0,
            max(test_spectrum),
            colors="k",
            linestyles="dashed",
            linewidth=0.5,
            alpha=0.5,
        )
        ax.plot(ppm_domain, test_spectrum, label="Test Spectrum")
        ax.plot(ppm_domain, synth_spectrum, label="Synthetic Spectrum", alpha=0.5)

        ax.set_title(f"Comparison for {MappingNames()[copo]} - {comp}")
        ax.set_yscale("log")
        ax.set_xlim(ppm_domain[0], ppm_domain[-1])
        ax.set_xlabel(r"$\delta$ (ppm)")
        ax.set_ylabel("Normalized Intensity")
        ax.legend()

    # If there are unused subplots (e.g. odd number of comps), remove them
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    # Optionally save to file, e.g.:
    os.makedirs(f"{SAVE_PATH}/Original_VS_Synthetic", exist_ok=True)
    plt.savefig(f"{SAVE_PATH}/Original_VS_Synthetic/comparison_synth_{copo}.png", dpi=200)
    plt.show()

# %% ANALYSIS ON LLDPE-H/B vs LLDPE-O

for copolymer in ["EH", "EB"]:
    if copolymer == "PP_RACO":
        copolymer = "RACO"
    ppm_domain = library[copolymer]["ppm_domains"].iloc[0].values[1:]
    test_data_c = test_data[test_data["copolymer"] == copolymer]
    for idx in test_data_c.index:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
        axes = axes.flatten()  # Make sure we can index as a 1D array

        # --- First subplot ---
        axes[0].plot(ppm_domain, test_data_c.loc[idx].values[4:], label="Test Spectrum")
        axes[0].vlines(
            library[copolymer]["legit_peaks"],
            0,
            1,
            colors="b",
            linestyles="dashed",
            linewidth=0.5,
            alpha=0.5,
            label=f"{MappingNames()[copolymer]} Peaks' Positions",
        )
        axes[0].set_title(f"Test Spectrum for {MappingNames()[copolymer]} - {test_data_c.loc[idx, 'c'][0]}", fontsize=14)
        axes[0].legend()
        axes[0].set_yscale("log")
        axes[0].set_xlim(ppm_domain[0], ppm_domain[-1])
        axes[0].set_ylabel("Normalized Intensity", fontsize=14)

        # --- Second subplot ---
        axes[1].plot(ppm_domain, test_data_c.loc[idx].values[4:], label="Test Spectrum")
        axes[1].vlines(
            library["EO"]["legit_peaks"],
            0,
            1,
            colors="r",
            linestyles="dashed",
            linewidth=0.5,
            alpha=0.5,
            label="EO Peaks' Positions",
        )
        axes[1].legend(fontsize=12)
        axes[1].set_yscale("log")
        axes[1].set_xlim(ppm_domain[0], ppm_domain[-1])
        axes[1].set_ylabel("Normalized Intensity", fontsize=14)
        axes[1].set_xlabel(r"$\delta$ (ppm)", fontsize=14)
        axes[1].tick_params(axis='both', which='major', labelsize=12)

        for ax in [axes[0]]:
            ax.set_xticks([])      # Removes the small tick lines
            ax.set_xticklabels([])
            ax.set_xlabel("")
        # Make them nicely spaced
        plt.tight_layout()
        os.makedirs(f"{SAVE_PATH}/CheckPeaks/Test/{MappingNames()[copolymer]}", exist_ok=True)
        plt.savefig(f"{SAVE_PATH}/CheckPeaks/Test/{MappingNames()[copolymer]}/{idx}.png", dpi=150)
        plt.show()


#%%
