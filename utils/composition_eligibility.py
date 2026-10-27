"""Evaluation-only composition selection; independent of models and training losses.

MinMax coordinates are not presence flags. In particular, zero and values
outside [0, 1] can represent genuine measured compositions. Missing physical
labels are tracked separately from the normalized absent-component sentinel.
"""
import ast
import numpy as np


def label_availability(metadata, labels):
    """Return explicit tuple-label availability in the original metadata order."""
    available = np.zeros((len(metadata), len(labels)), dtype=bool)
    index = {name: j for j, name in enumerate(labels)}
    for i, row in enumerate(metadata.itertuples(index=False)):
        names = ast.literal_eval(row.copo_tuple) if isinstance(row.copo_tuple, str) else row.copo_tuple
        values = ast.literal_eval(row.c) if isinstance(row.c, str) else row.c
        if len(names) != len(values):
            raise ValueError(f'Misaligned composition labels at row {i}')
        for name, value in zip(names, values):
            if name not in index:
                raise ValueError(f'Unknown component {name!r} at row {i}')
            # None/empty/TEST are missing metadata, not normalized coordinates.
            available[i, index[name]] = value is not None and not (
                isinstance(value, str) and value.strip().upper() in {'', 'TEST', 'NA', 'N/A', 'NAN'}
            )
    return available


def composition_mask(weight, physical, normalized, predicted, available, *, copolymer=True):
    """Select present, explicitly labelled, finite targets; reject failed predictions.

    `available` must describe original physical labels (tuple membership for
    experimental metadata, or a documented availability array for dense saved
    targets). Never derive it from the sign of a normalized coordinate.
    Nonfinite predictions on eligible targets raise instead of shrinking n.
    """
    weight, physical, normalized, predicted = map(np.asarray, (weight, physical, normalized, predicted))
    if any(a.shape != weight.shape for a in (physical, normalized, predicted)):
        raise ValueError('Composition target/prediction shapes disagree')
    available = np.asarray(available, dtype=bool)
    if available.shape != weight.shape:
        raise ValueError('Physical-label availability shape disagrees')
    eligible = (np.asarray(copolymer, dtype=bool) & (weight > 0) & np.isfinite(weight)
                & available & np.isfinite(physical) & np.isfinite(normalized))
    bad = eligible & ~np.isfinite(predicted)
    if bad.any():
        raise ValueError(f'Nonfinite predictions for eligible compositions: {np.argwhere(bad).tolist()}')
    return eligible


def relative_bounds(axis, percent):
    """Historical relative bands in normalized units, with ordered boundaries.

    For negative coordinates the two historical products exchange order. Their
    geometric region is unchanged: axis ± percent/100 * abs(axis). Positive
    products retain their original arithmetic and precision; zero collapses.
    """
    axis = np.asarray(axis)
    a, b = axis * (1 - percent / 100), axis * (1 + percent / 100)
    return np.minimum(a, b), np.maximum(a, b)


def parity_limits(true, predicted, default=(-.1, 1.1)):
    """Common publication parity axes, as explicitly requested by the author.

    Axis clipping is a display convention only. All eligible observations,
    including out-of-view targets, remain in Source Data and metric populations.
    """
    values = np.r_[np.asarray(true).ravel(), np.asarray(predicted).ravel()]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite parity coordinates')
    return tuple(map(float, default))
