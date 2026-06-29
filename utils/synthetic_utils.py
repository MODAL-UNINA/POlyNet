#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities for synthetic DeepNMR dataset generation.

This module contains the low-level routines used by ``0_synthDataGen.py`` to
build synthetic NMR spectra from fingerprint-envelope interpolation functions.

The current generation pipeline uses:

  - ``generate_synthetic_params`` to build valid weight/composition grids;
  - ``create_mixture`` to sample those grids and generate spectra in parallel;
  - ``create_single_mixture`` to build one synthetic spectrum, including
    mixture superposition, peak copy/removal, chain-end signals, safe spectral
    augmentation, baseline distortion, noise, masking, shift, and final
    area-normalization.

Several older augmentation helpers are kept in the file for backward
compatibility, but are explicitly marked as legacy. They are not called by the
current dataset-generation path.

Important
---------
This file is scientifically sensitive: numerical constants, probabilities,
normalization steps, random branches, and operation order should not be changed
unless the synthetic dataset generation protocol is intentionally revised.
"""

# %% IMPORT SECTION
import itertools
import logging
import math
import random
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.integrate import simpson as simps
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.special import voigt_profile
from tqdm import trange


# %% LEGACY AUGMENTATION FUNCTIONS - NOT USED BY CURRENT GENERATION PATH


def compress_expand_spectrum(intensities, factor):
    """
    Compresses or expands a spectrum by downsampling or upsampling its intensities.

    Parameters:
        intensities (array-like): The original spectrum intensities.
        factor (float):
            - If `factor >= 1`, compress the spectrum by downsampling.
            - If `factor < 1`, expand the spectrum by upsampling.

    Returns:
        numpy.ndarray: The modified intensities after compression or expansion,
                        rescaled to maintain the total area under the curve.
    """
    # Ensure intensities are in floating-point format
    intensities = intensities.astype(float)

    # Define the original frequency range (indices corresponding to intensities)
    frequencies = np.linspace(0, len(intensities) - 1, len(intensities))

    # Calculate the new length of the spectrum after compression/expansion
    new_length = int(len(intensities) / factor)
    new_frequencies = np.linspace(frequencies[0], frequencies[-1], new_length)

    # Initialize the modified intensities array
    modified_intensities = np.zeros(new_length)

    if factor >= 1:
        # Compression: Downsampling using averaging
        for i in range(new_length):
            # Define the range of indices to average over
            start_index = int(i * factor)
            end_index = min(int((i + 1) * factor), len(intensities))

            # Average the intensities within the range
            modified_intensities[i] = np.mean(intensities[start_index:end_index])
    else:
        # Expansion: Upsampling using linear interpolation
        modified_intensities = np.interp(new_frequencies, frequencies, intensities)

    # Rescale to maintain the total area under the curve
    original_area = simps(intensities)  # Area under the original spectrum
    modified_area = simps(modified_intensities)  # Area under the modified spectrum

    # Scaling factor to preserve the total area
    scaling_factor = original_area / modified_area
    modified_intensities *= scaling_factor

    return modified_intensities


def find_main_peak(intensities):
    """
    Finds the main peak (highest intensity) in the given spectrum.

    Parameters:
        intensities (array-like): The spectrum intensities.

    Returns:
        int: The index of the main peak.
    """
    # Identify peaks with a prominence of at least 30% of the maximum intensity
    peaks, _ = find_peaks(intensities, height=np.max(intensities) * 0.3)

    if len(peaks) == 0:  # Handle case with no detected peaks
        raise ValueError("No peaks found above the threshold.")

    # Find the index of the peak with the maximum intensity
    main_peak = peaks[np.argmax(intensities[peaks])]

    return main_peak


def align_main_peaks(original_intensities, compressed_intensities):
    """
    Aligns the main peak of the compressed spectrum with the main peak of the original spectrum.

    Parameters:
        original_intensities (array-like): The original spectrum intensities.
        compressed_intensities (array-like): The compressed or modified spectrum intensities.

    Returns:
        numpy.ndarray: The compressed intensities shifted to align with the original spectrum.
    """
    # Ensure the input spectra have the same length
    if len(original_intensities) != len(compressed_intensities):
        raise ValueError("Original and compressed spectra must have the same length.")

    # Define the range to search for the main peak (e.g., a specific region of interest)
    search_start = 13000
    search_end = 16000

    # Find the main peak in the defined region for both spectra
    original_main_peak = (
        find_main_peak(original_intensities[search_start:search_end]) + search_start
    )
    compressed_main_peak = (
        find_main_peak(compressed_intensities[search_start:search_end]) + search_start
    )

    # Calculate the shift required to align the compressed spectrum with the original spectrum
    shift = original_main_peak - compressed_main_peak

    # Align the compressed spectrum by applying a circular shift
    aligned_intensities = np.roll(compressed_intensities, shift)

    return aligned_intensities


def modify_intensities_and_positions(spectrum, factor=0.1, shift=10):
    """
    Modifies the intensities and positions of peaks in a spectrum.

    Parameters:
        spectrum (array-like): The original spectrum intensities.
        factor (float): The random variation factor for peak intensities (default is 0.1, ±10%).
        shift (int): The maximum random shift in peak positions (default is 10).

    Returns:
        numpy.ndarray: The modified spectrum with altered peak intensities and positions.
    """
    # Find peaks in the spectrum with a prominence threshold
    pks = find_peaks(spectrum, height=np.max(spectrum) * 0.001)[0]

    # Ensure at least one peak is detected
    if len(pks) == 0:
        raise ValueError("No peaks detected in the spectrum.")

    # Compute the left and right bases of peaks using peak prominences
    _, left_bases, right_bases = peak_prominences(spectrum, pks, wlen=30)

    # Calculate the area of the largest peak (used for normalization)
    max_area = simps(
        spectrum[
            left_bases[np.argmax(spectrum[pks])] : right_bases[np.argmax(spectrum[pks])]
        ]
    )

    # Copy the original spectrum for modifications
    intensities = spectrum.copy()

    # Iterate over all detected peaks to modify their intensities and positions
    for i, _ in enumerate(pks):
        # Compute the area under the current peak
        integral = simps(intensities[left_bases[i] : right_bases[i]])

        if integral < 1e-1 * max_area:

            # Introduce random fluctuation to the peak's intensity
            j = np.random.uniform(1.0 - factor, 1.0 + factor) * max_area / integral

            # Calculate a scaling factor (k) for the peak to achieve the new intensity
            k = max_area / (j * integral)

            # Modify the peak's intensities by scaling
            mod_peak = k * intensities[left_bases[i] : right_bases[i]]

            # Replace the original peak region with the baseline value
            intensities[left_bases[i] : right_bases[i]] = intensities[left_bases[i]]

            # Introduce a random shift to the peak position
            s = np.random.randint(-shift, shift)

            # Ensure the shifted peak fits within the spectrum boundaries
            assert len(mod_peak) == len(
                intensities[left_bases[i] + s : right_bases[i] + s]
            ), f"Shifted indices exceed bounds: {left_bases[i] + s}, {right_bases[i] + s}"

            # Place the modified peak into the shifted position
            intensities[left_bases[i] + s : right_bases[i] + s] = mod_peak

    return intensities


# %% CURRENT PEAK AND SPECTRAL AUGMENTATION HELPERS

def remove_and_copy_peaks(
    intensities,
    max_peaks_to_remove=2,
    max_peaks_to_copy=2,
    inclusion_threshold=1e-3,
    random_scaling=(0.05, 0.25),
    wlen=50,
):
    """
    Modifies a spectrum by randomly removing and copying peaks.

    Steps:
    1. Detect peaks and their prominence.
    2. Randomly remove some peaks by replacing them with the median intensity.
    3. Randomly copy some peaks and insert them into new positions.

    Parameters:
        intensities (array-like): The spectrum intensities.
        max_peaks_to_remove (int): Maximum number of peaks to remove (default: 2).
        max_peaks_to_copy (int): Maximum number of peaks to copy (default: 2).
        wlen (int): Window length for peak prominence calculation (default: 50).

    Returns:
        numpy.ndarray: The modified spectrum with removed and copied peaks.
    """
    # Copy the original intensities to avoid modifying them directly
    modified_intensities = intensities.copy()

    # Step 1: Detect all peaks
    peaks, _ = find_peaks(intensities, height=np.max(intensities) * inclusion_threshold)
    if len(peaks) == 0:
        print("No peaks found, returning original intensities.")
        return intensities

    # Remove only small peaks (ignore highest peak)
    peaks = peaks[intensities[peaks] < 1e-2 * np.max(intensities)]

    # Get prominences and bases for all peaks
    _, left_bases, right_bases = peak_prominences(intensities, peaks, wlen=wlen)

    # Step 2: Randomly remove peaks
    num_peaks_to_remove = min(max_peaks_to_remove, len(peaks))
    if num_peaks_to_remove > 0 and np.random.rand() > 0.7:
        num_peaks_to_remove = np.random.randint(1, num_peaks_to_remove + 1)
        peaks_to_remove = np.random.choice(peaks, num_peaks_to_remove, replace=False)
    else:
        peaks_to_remove = np.array([])

    for peak in peaks_to_remove:
        # Get index of peak in `peaks`
        idx = np.where(peaks == peak)[0][0]
        left_base = int(left_bases[idx])
        right_base = int(right_bases[idx])

        # Replace peak region with median intensity
        modified_intensities[left_base : right_base + 1] = np.median(intensities)

    # Step 3: Update peaks after removal
    remaining_peaks_mask = ~np.isin(peaks, peaks_to_remove)
    remaining_peaks = peaks[remaining_peaks_mask]
    remaining_left_bases = left_bases[remaining_peaks_mask]
    remaining_right_bases = right_bases[remaining_peaks_mask]

    # Step 4: Randomly copy peaks
    num_peaks_to_copy = min(max_peaks_to_copy, len(remaining_peaks))
    if num_peaks_to_copy > 0:
        num_peaks_to_copy = np.random.randint(1, num_peaks_to_copy + 1)
        peaks_to_copy = np.random.choice(
            remaining_peaks, num_peaks_to_copy, replace=False
        )
    else:
        peaks_to_copy = np.array([])

    # Create a peak mask to track occupied regions
    peak_mask = np.zeros(len(intensities), dtype=bool)

    # Mark existing peak regions as occupied
    for lb, rb in zip(remaining_left_bases, remaining_right_bases):
        peak_mask[int(lb) : int(rb) + 1] = True

    # Avoid placing peaks near spectrum boundaries
    peak_mask[:3000] = True
    peak_mask[-3000:] = True

    # For each peak to copy
    for peak in peaks_to_copy:
        # Find peak boundaries
        idx = np.where(remaining_peaks == peak)[0][0]
        left_base = int(remaining_left_bases[idx])
        right_base = int(remaining_right_bases[idx])

        # Extract peak profile
        peak_profile = intensities[left_base : right_base + 1].copy()
        peak_width = right_base - left_base

        # Apply a random intensity scaling (5% to 25% of original)
        scaling_factor = np.random.uniform(
            random_scaling[0], random_scaling[1]
        )  # **Fixed incorrect 0.50**
        peak_profile *= scaling_factor

        # Find regions without peaks to insert a new peak
        non_peak_indices = np.where(~peak_mask)[0]
        if len(non_peak_indices) == 0:
            continue  # No space found for insertion

        # Find continuous empty regions
        diff = np.diff(non_peak_indices)
        region_edges = np.where(diff > 1)[0]
        start_indices = np.insert(
            non_peak_indices[region_edges + 1], 0, non_peak_indices[0]
        )
        end_indices = np.append(non_peak_indices[region_edges], non_peak_indices[-1])

        # Filter for regions wide enough to accommodate the peak
        possible_positions = [
            (start, end)
            for start, end in zip(start_indices, end_indices)
            if end - start >= peak_width
        ]

        if not possible_positions:
            continue  # No suitable space found

        # Select a random valid position
        new_region = possible_positions[np.random.randint(len(possible_positions))]
        max_start = new_region[1] - peak_width
        new_start = np.random.randint(new_region[0], max_start + 1)
        new_end = new_start + peak_width

        # Insert the copied peak
        modified_intensities[new_start : new_end + 1] += peak_profile

        # Update peak mask
        peak_mask[new_start : new_end + 1] = True

    return modified_intensities


def chain_ends_addition(
    spectrum, ppm_domain_general, column_labels, w_mixture, probability_threshold=0.5
):
    """
    Augments the input NMR spectrum by probabilistically adding synthetic chain-end signals
    for specific copolymers (excluding LDPE, PE, PP, RACO, PP_RACO).

    Parameters:
        spectrum (np.ndarray): The 1D NMR spectrum to be modified.
        ppm_domain_general (np.ndarray): Chemical shift domain (in ppm), same length as spectrum.
        column_labels (List[str]): Names of copolymers in the mixture.
        w_mixture (List[float]): Corresponding weight fractions of each copolymer.
        probability_threshold (float): Probability threshold controlling the stochastic addition
                                        of chain-end peaks (default: 0.5).

    Returns:
        np.ndarray: The modified spectrum with possible chain-end peaks added.
    """
    # Positions, in ppm, of the peaks to be added as chain end signals
    peaks = [13.995, 22.69, 29.15, 29.25, 29.35, 29.80, 30.8, 32.01, 33.8]
    # Positions on the domain vector
    pos = [np.argmin(np.abs(ppm_domain_general - k)) for k in peaks]
    domain = np.arange(len(spectrum))

    chainends_spectrum_general = np.zeros_like(spectrum)
    for copo, w in zip(column_labels, w_mixture):
        # Chain ends are added only for LLDPE copolymers
        if w > 0 and copo not in ["LDPE", "PE", "PP", "RACO", "PP_RACO"]:
            chainends_spectrum = np.zeros_like(spectrum)
            # intensity = np.random.uniform(2e-3, 1e-5)
            intensity = np.random.uniform(1e-5, 3e-3)
            # Add chain end peaks only with a certain probability to augment the diversity of the dataset
            # if np.random.rand() > probability_threshold:
            if copo in ["EH", "EO"]:
                for p in pos:
                    if np.random.rand() > probability_threshold:
                        chainends_spectrum += np.random.uniform(0.8, 1.2) * intensity * voigt_profile(
                            domain - p,
                            np.random.uniform(0.9, 1.1),
                            np.random.uniform(0.9, 1.1),
                        )
                chainends_spectrum_general += w * chainends_spectrum
            elif copo in ["EB", "EPR"]:
                for p in pos:
                    chainends_spectrum += np.random.uniform(0.8, 1.2) * intensity * voigt_profile(
                        domain - p,
                        np.random.uniform(0.9, 1.1),
                        np.random.uniform(0.9, 1.1),
                    )
                chainends_spectrum_general += w * chainends_spectrum

    spectrum += chainends_spectrum_general

    return spectrum


# %% LEGACY COMPOSITE AUGMENTATION WRAPPER - NOT USED BY CURRENT GENERATION PATH

def augment(spectrum, compress=True, modify_peaks=True):
    """
    Augments a spectrum by applying compression/expansion, alignment, and intensity/position modifications.

    Steps:
    1. Compress or expand the spectrum by a random factor close to 1.
    2. Ensure the modified spectrum matches the original length.
    3. Align the main peak of the modified spectrum with the original spectrum.
    4. Modify intensities and positions of the peaks to introduce randomness.

    Parameters:
        spectrum (array-like): The original spectrum to be augmented.
        compress (bool): Whether to apply compression/expansion.
        modify_peaks (bool): Whether to modify peak intensities and positions.

    Returns:
        numpy.ndarray: The augmented spectrum.
    """
    # Step 1: Randomly compress or expand the spectrum slightly
    # compression_factor = np.random.uniform(0.98, 1.02)  # Random factor near 1
    if compress:
        epsilon = 1e-1
        compression_factor = np.random.uniform(
            1 - epsilon, 1 + epsilon
        )  # Random factor near 1
        if compression_factor != 1:  # Avoid division by zero or no change
            modified_spectrum = compress_expand_spectrum(
                spectrum.copy(), compression_factor
            )
        else:
            modified_spectrum = spectrum.copy()

        # Step 2: Match the length of the modified spectrum to the original spectrum
        delta = np.abs(len(modified_spectrum) - len(spectrum))  # Length difference

        if len(modified_spectrum) < len(spectrum):
            # Extend the spectrum by repeating the last few elements
            modified_spectrum = np.concatenate(
                [modified_spectrum, modified_spectrum[-delta:]]
            )
        else:
            # Trim the spectrum to match the original length
            modified_spectrum = modified_spectrum[: len(spectrum)]

        # Step 3: Align the main peak of the modified spectrum with the original spectrum
        aligned_spectrum = align_main_peaks(spectrum.copy(), modified_spectrum.copy())
    else:
        aligned_spectrum = spectrum.copy()

    # Step 4: Modify intensities and positions of the peaks
    if modify_peaks:
        augmented_spectrum = modify_intensities_and_positions(
            aligned_spectrum.copy(), factor=0.025, shift=5
        )
    else:
        augmented_spectrum = aligned_spectrum

    return augmented_spectrum


# %% CURRENT SAFE AUGMENTATION PROCEDURE

def augment_spectrum_safe(spectrum, compress=True, modify_peaks=True, seed=None):
    y = np.asarray(spectrum, dtype=float).copy()
    n = len(y)
    x = np.arange(n, dtype=float)

    rng = np.random.default_rng(seed)

    # ----------------------------
    # STEP 1: Warp around anchor
    # ----------------------------
    if compress:
        eps = 0.005  # 0.5%
        factor = rng.uniform(1 - eps, 1 + eps)

        if abs(factor - 1.0) > 1e-9:
            search_start, search_end = 13000, 16000
            region = y[search_start:search_end]
            if region.size > 0 and np.max(region) > 0:
                anchor = search_start + int(np.argmax(region))
            else:
                anchor = n // 2

            xq = anchor + (x - anchor) * factor

            # edge-hold, no zeros
            f = interp1d(
                x, y, kind="linear", bounds_error=False, fill_value=(y[0], y[-1])
            )
            y_warp = f(xq)

            # # preserve L2
            # a = np.sqrt(np.mean(y**2)) + 1e-12
            # b = np.sqrt(np.mean(y_warp**2)) + 1e-12
            # y = y_warp * (a / b)

            # preserve L1
            a = simps(y) + 1e-12
            b = simps(y_warp) + 1e-12
            y = y_warp * (a / b)

    # ----------------------------
    # STEP 2: Smooth peak scaling
    # ----------------------------
    if modify_peaks:
        noise_region = np.concatenate([y[:2500], y[-2500:]])
        noise_mean = np.mean(noise_region)
        noise_std = np.std(noise_region)
        thr = noise_mean + 6 * noise_std
        pks, props = find_peaks(y, height=thr)

        if pks.size > 0:
            max_pk = np.max(y[pks])
            mult = np.ones(n, dtype=float)

            # width estimate (at half prominence)
            widths, _, left_ips, right_ips = peak_widths(y, pks, rel_height=0.5)

            for i, pk in enumerate(pks):
                h = y[pk]
                if h < 0.1 * max_pk:
                    # define window around peak, clamp size
                    w = int(np.clip(widths[i] * 3.0, 5, 200))
                    lb = max(0, pk - w)
                    rb = min(n, pk + w + 1)

                    scale = rng.uniform(0.85, 1.15)

                    win = rb - lb
                    bell = np.hanning(win)
                    modifier = 1.0 + (scale - 1.0) * bell

                    # combine but keep bounded
                    mult[lb:rb] *= modifier

            mult = np.clip(mult, 0.7, 1.3)
            y *= mult

    return y


# %% GENERATION OF PARAMETERS FOR SYNTHETIC SPECTRA


def find_weight_combinations(N, step=0.1, precision=2, weight_bounds=(0.1, 0.9)):
    """
    Iterative version of the function that removes recursion and runs efficiently.
    It generates all valid weight combinations for N components that sum to 1,
    excluding small contributions (e.g., 1%-99% but preserving 0 and 1).

    Parameters:
        N (int): Number of components.
        step (float, optional): Increment for weight values (default: 0.1).
        precision (int, optional): Decimal precision for rounding (default: 2).

    Returns:
        np.ndarray: Valid weight distributions.
    """
    if N < 2:
        raise ValueError("N must be greater than 1")

    # Generate valid weight values, excluding very small fractions
    possible_weights = np.round(np.arange(0.0, 1.0 + step, step), precision)
    possible_weights = possible_weights[
        ~((possible_weights > 0) & (possible_weights < weight_bounds[0]))
    ]
    possible_weights = possible_weights[
        ~((possible_weights > weight_bounds[1]) & (possible_weights < 1.0))
    ]

    # Estimated number of valid combinations
    n_combs = math.comb(int(1 / step) + N - 1, N - 1)
    print(f"Expected weight combinations: {n_combs}")

    # Generate valid combinations using a controlled iteration
    valid_combinations = []

    def generate_valid_combinations(remaining, current_combination):
        """Iterative approach to generate valid combinations."""
        if len(current_combination) == N - 1:
            last_weight = np.round(1 - sum(current_combination), precision)
            if 0 <= last_weight <= 1:
                valid_combinations.append(tuple(current_combination + [last_weight]))
            return

        for w in possible_weights:
            if sum(current_combination) + w > 1:
                break  # Stop early if the sum exceeds 1
            generate_valid_combinations(remaining - 1, current_combination + [w])

    # Start the controlled iteration
    generate_valid_combinations(N, [])

    # Convert to NumPy array
    return np.array(valid_combinations, dtype=np.float16)


def generate_synthetic_params(
    copolymer_list,
    compositions,
    max_components="all",
    weight_step=0.1,
    precision=2,
    weight_bounds=(0.1, 0.9),
):
    """
    Generates synthetic compositions and weight distributions for copolymer mixtures.

    Parameters:
        copolymer_list (list): List of copolymers to include in the dataset.
        compositions (dict): Dictionary of composition ranges for each copolymer.
        max_components (int or "all", optional): Maximum number of components allowed in a mixture.
            If "all", no limit is applied. Defaults to "all".
        weight_step (float, optional): Step size for weight distribution. Defaults to 0.1.
        precision (int, optional): Decimal precision for rounding. Defaults to 2.
        n_jobs (int, optional): Number of parallel jobs for weight computation.
            If -1, all available cores are used. Defaults to -1.

    Returns:
        tuple:
            - weights (np.ndarray): Array of valid weight distributions for mixtures.
            - compositions_mix (np.ndarray): Array of valid composition combinations.
    """
    # Step 1: Generate all possible composition combinations
    compositions_mix = np.round(
        np.array(list(itertools.product(*compositions.values()))), precision
    )

    # Step 2: Generate weight combinations based on parallel processing
    weights = find_weight_combinations(
        len(copolymer_list),
        step=weight_step,
        precision=precision,
        weight_bounds=weight_bounds,
    )

    print(f"Computed weights combinations: {len(weights)}")

    # Step 3: Apply max_components filter if needed
    print(f"Masking weights with max_components={max_components}...", end=" ")
    t0 = time.time()
    if isinstance(max_components, int) and max_components <= len(copolymer_list):
        mask = (
            np.count_nonzero(weights, axis=1) <= max_components
        )  # Efficient count of nonzero elements
        weights = np.round(weights[mask], precision)  # Apply mask and round
    print(f"Done in {time.time() - t0:.2f}s")

    return weights, compositions_mix


def add_curved_baseline(spectrum, max_amplitude=0.005, complexity=3):
    """
    Adds a random curved baseline distortion to a spectrum by summing low-frequency sine waves.

    Parameters:
        spectrum (np.ndarray): Input spectrum.
        max_amplitude (float): Maximum distortion amplitude relative to the maximum spectrum intensity.
        complexity (int): Number of sine waves to sum.

    Returns:
        np.ndarray: Spectrum with added curved baseline.
    """
    max_abs_amplitude = np.max(spectrum) * max_amplitude
    x = np.linspace(0, 1, len(spectrum))
    baseline_distortion = np.zeros_like(spectrum)

    for _ in range(complexity):
        amplitude = np.random.uniform(
            -max_abs_amplitude / complexity, max_abs_amplitude / complexity
        )
        frequency = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi)

        baseline_distortion += amplitude * np.sin(2 * np.pi * frequency * x + phase)

    assert len(baseline_distortion) == len(spectrum), "Baseline length mismatch"
    return baseline_distortion


# %% MIXTURE CREATION PROCEDURES

def safe_area_normalize(spectrum, eps=1e-12, clip_min=1e-16):
    """Clip and area-normalize a spectrum using Simpson integration."""
    spectrum = np.asarray(spectrum, dtype=float).reshape(-1)
    spectrum = np.maximum(spectrum, clip_min)

    area = simps(spectrum)

    if not np.isfinite(area) or np.abs(area) < eps:
        raise ValueError(f"Invalid spectrum area during normalization: {area}")

    return spectrum / area

def create_single_mixture(
    i,
    df_w_values,
    df_c_values,
    column_labels,
    interp_dict,
    ppm_domain_general,
    noise_pool=None,
    chem_data=False,
):
    try:
        # Extract weight and concentration values for mixture
        w_mixture = df_w_values[i]
        c_mixture = df_c_values[i]

        # Identify nonzero components
        list_nonzero_w = np.where(w_mixture != 0)[0]

        spectrum_components = []

        for k in list_nonzero_w:
            label = column_labels[k]
            # If the component is not an homopolymer, we use the fingerprint on
            # randomly generated weight and concentration values
            if column_labels[k] not in ["LDPE", "PE", "PP"]:
                # spectrum_components.append(
                #     np.asarray(
                #         interp_dict[column_labels[k]](
                #             np.arange(len(ppm_domain_general)),
                #             c_mixture[k],
                #         ).T[0]
                #     )
                #     * w_mixture[k]
                # )
                component = np.asarray(
                        interp_dict[label](
                            np.arange(len(ppm_domain_general)),
                            c_mixture[k],
                        ).T[0]
                    ).reshape(-1)
            # If the component is PP or HDPE we use the fingerprint on
            # randomly generated weight values
            elif column_labels[k] in ["PE", "PP"]:
                # spectrum_components.append(
                #     interp_dict[column_labels[k]](np.arange(len(ppm_domain_general))).T
                #     * w_mixture[k]
                # )
                component = np.asarray(
                    interp_dict[label](
                        np.arange(len(ppm_domain_general))
                    ).T
                ).reshape(-1)
            # If the component is LDPE, we use a random choice between the extracted fingerprints
            elif column_labels[k] in ["LDPE"]:
                # spectrum_components.append(
                #     random.choice(interp_dict[column_labels[k]])(
                #         np.arange(len(ppm_domain_general))
                #     ).T
                #     * w_mixture[k]
                # )
                component = np.asarray(
                    random.choice(interp_dict[label])(
                        np.arange(len(ppm_domain_general))
                    ).T
                ).reshape(-1)
            else:
                raise ValueError(f"Unknown component: {column_labels[k]}")

            # component = safe_area_normalize(component, clip_min=1e-16)
            spectrum_components.append(component * w_mixture[k])

        # Ensure all spectral components are nonzero
        for idx, spectrum_component in enumerate(spectrum_components):
            assert np.any(spectrum_component != 0), f"Null spectrum for component {idx}"

        # Generate the synthetic spectrum by summing the components
        spectrum = np.sum(spectrum_components, axis=0)
        assert np.any(spectrum != 0), "Null spectrum generated from components."
        spectrum = safe_area_normalize(spectrum, clip_min=1e-16)
        assert np.isclose(simps(spectrum), 1), f"Generated spectrum is not area-normalized: area={simps(spectrum)}"

        # spectrum = safe_area_normalize(spectrum, clip_min=1e-16)

        # Random copying/removal of peaks
        if np.random.rand() < 0.75:
            spectrum = remove_and_copy_peaks(
                spectrum.copy(),
                max_peaks_to_remove=0,
                max_peaks_to_copy=2,
                inclusion_threshold=1e-3,
                random_scaling=(0.20, 0.50),
                wlen=50,
            )

        # Random addition of chain-end signals
        if np.random.rand() < 0.75:
            spectrum = chain_ends_addition(
                spectrum.copy(),
                ppm_domain_general,
                column_labels,
                w_mixture,
                probability_threshold=0.3,
            )

        #Augmentation Procedures for the spectrum
        sigma = np.random.rand()
        if sigma < 0.75:
            if chem_data:
                spectrum = augment_spectrum_safe(spectrum.copy(), compress=False, modify_peaks=True)
            else:
                if sigma < 0.5:
                    spectrum = augment_spectrum_safe(spectrum.copy(), compress=False, modify_peaks=True)
                else:
                    spectrum = augment_spectrum_safe(spectrum.copy(), compress=True, modify_peaks=True)

        if np.random.rand() < 0.5:
            spectrum += add_curved_baseline(
                spectrum,
                max_amplitude=np.random.uniform(1e-6, 5e-5),
                complexity=np.random.randint(2, 5),
            )

        # Noise Addition
        # --- 1. NOISE DECISION LOGIC ---
        has_noise_pool = (noise_pool is not None) and (len(noise_pool) > 0)

        apply_real_noise = (np.random.rand() < 0.5) if has_noise_pool else False
        apply_gauss_noise = np.random.rand() < 0.5

        # Force at least one noise type if both were skipped
        if not apply_real_noise and not apply_gauss_noise:
            if has_noise_pool:
                if np.random.rand() < 0.5:
                    apply_real_noise = True
                else:
                    apply_gauss_noise = True
            else:
                apply_gauss_noise = True

        # --- 2. REAL NOISE APPLICATION ---
        if apply_real_noise:
            len_crop = 1000
            num_segments_needed = int(np.ceil(len(spectrum) / len_crop))
            noise_segments_list = []

            # Choose source segment outside the loop to simulate a single instrument baseline
            source_noise_segment = random.choice(noise_pool)

            for _ in range(num_segments_needed):
                if len(source_noise_segment) > len_crop:
                    max_start_index = len(source_noise_segment) - len_crop
                    start_index = np.random.randint(0, max_start_index + 1)
                    noise_crop = source_noise_segment[
                        start_index : start_index + len_crop
                    ]
                else:
                    noise_crop = source_noise_segment
                noise_segments_list.append(noise_crop)

            full_length_noise = np.concatenate(noise_segments_list)[: len(spectrum)]
            noise_scaling = np.random.uniform(0.5, 1.5)
            scaled_noise = full_length_noise * noise_scaling

            # Random circular shift to avoid recognizable patterns
            scaled_noise = np.roll(
                scaled_noise, np.random.randint(0, len(scaled_noise))
            )

            spectrum += scaled_noise

        # --- 3. GAUSSIAN NOISE APPLICATION ---
        if apply_gauss_noise:
            mean_noise = 0.0

            if has_noise_pool:
                reference_noise = random.choice(noise_pool)
                base_std = np.std(reference_noise)
                
                # Stochastic variation factor (e.g. +/- 20%)
                std_noise = base_std * np.random.uniform(0.8, 1.2)
            else:
                # Fallback based on historical instrument data
                std_noise = np.random.uniform(0.75e-5, 1.75e-5)

            spectrum += np.random.normal(
                loc=mean_noise, scale=std_noise, size=spectrum.shape
            )

        # Random Masking Procedure
        if np.random.rand() < 0.1 and not chem_data:
            win_to_erase = np.random.uniform(0.25, 0.75)  # in ppm
            # convert in samples
            win_to_erase = int(win_to_erase / np.abs(np.diff(ppm_domain_general))[0])
            # chose a random position
            pos_to_erase = np.random.randint(0, len(spectrum) - win_to_erase)
            # Erase the spectrum in that position
            spectrum[pos_to_erase : pos_to_erase + win_to_erase] = np.random.normal(
                np.mean(spectrum), np.max(spectrum), win_to_erase
            )
            # Ensure the erased part is not below a certain threshold
            spectrum[pos_to_erase : pos_to_erase + win_to_erase][
                spectrum[pos_to_erase : pos_to_erase + win_to_erase]
                < np.median(spectrum)
            ] = 1e-6

        # Random Shifting Procedure
        eta = np.abs(ppm_domain_general[1] - ppm_domain_general[0])
        if np.random.rand() < 0.5:
            if chem_data:
                max_shift_ppm = 0.01
            else:
                max_shift_ppm = 0.05

            max_shift = int(round(max_shift_ppm / eta))
            spectrum = np.roll(spectrum, np.random.randint(-max_shift, max_shift + 1))

        spectrum = safe_area_normalize(spectrum, clip_min=1e-16)
        assert np.isclose(simps(spectrum), 1, atol=1e-6), "Final spectrum not area-normalized"

        return spectrum

    except Exception as e:
        logging.warning(f"Error creating mixture {i}: {e}")
        return None


def create_mixture(
    df_w,
    df_c,
    portion,
    max_components,
    column_labels,
    interp_dict,
    ppm_domain_general,
    noise_pool=None,
    n_mixture=10,
    n_jobs=32,
    chem_data=False,
):
    """
    Creates a mixture dataset using parallel processing.

    Parameters:
        df_w (pd.DataFrame): DataFrame containing weight values for mixtures.
        df_c (pd.DataFrame): DataFrame containing concentration values for mixtures.
        portion (list): Proportion of mixtures with a certain number of components.
        n_max_components (int, optional): Maximum number of components per mixture. Defaults to 5.
        column_labels (list): List of labels for the columns in df_w and df_c.
        interp_dict (dict): Dictionary mapping labels to interpolation functions.
        ppm_domain_general (np.ndarray): General ppm domain for spectra.
        n_mixture (int, optional): Number of mixtures to generate. Defaults to 10.
        n_jobs (int, optional): Number of parallel jobs. Defaults to 32.
        chem_data (bool, optional): Whether the data is chemical data. Defaults to False.
    Returns:
        tuple: (np.ndarray of generated mixtures, selected indices from df_c, selected indices from df_w, df_w_values, df_c_values)
    """

    # Fix: Convert portion fractions into absolute counts
    portionints = [int(np.ceil(i * n_mixture)) for i in portion]

    # Select indices from df_w based on nonzero component count
    indices_w = []
    for i in range(max_components):
        available_indices = df_w.loc[np.sum(df_w != 0, axis=1) == i + 1].index

        # Fix: Handle cases where 'available_indices' is smaller than requested
        if len(available_indices) < portionints[i]:
            selected_indices = np.random.choice(
                available_indices, size=portionints[i], replace=True
            )
        else:
            selected_indices = np.random.choice(
                available_indices, size=portionints[i], replace=False
            )

        indices_w.extend(selected_indices)

    indices_w = np.array(indices_w[:n_mixture])

    assert (
        len(indices_w) == n_mixture
    ), f"Wrong number of mixtures: Expected {n_mixture}, got {len(indices_w)}"

    # Fix: Ensure sufficient unique indices in df_c
    if n_mixture > df_c.shape[0]:
        indices_c = np.random.choice(df_c.index, size=n_mixture, replace=True)
    else:
        indices_c = np.random.choice(df_c.index, size=n_mixture, replace=False)

    df_w_values = df_w.loc[indices_w].values
    df_c_values = df_c.loc[indices_c].values

    # Parallelize the mixture creation
    mixes = Parallel(n_jobs=n_jobs)(
        delayed(create_single_mixture)(
            i,
            df_w_values,
            df_c_values,
            column_labels,
            interp_dict,
            ppm_domain_general,
            noise_pool,
            chem_data,
        )
        for i in trange(n_mixture)
    )

    failed = sum(m is None for m in mixes)
    if failed > 0:
        raise RuntimeError(f"{failed} mixtures failed during generation.")

    return np.array(mixes), indices_c, indices_w, df_w_values, df_c_values