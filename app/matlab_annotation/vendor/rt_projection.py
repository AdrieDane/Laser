"""Dense-code to retention-time design projection used by MATLAB annotation."""

from __future__ import annotations

import numpy as np


def design_row_from_dense_code(
    code: int,
    composition_bases: tuple[int, ...] | np.ndarray,
) -> int:
    """Return MATLAB's one-based row in ``Xp`` for a dense annotation code.

    ``update_MS_annotation_variables`` constructs ``Xp`` over composition
    dimensions 1, 2, 4 and 5; dimension 3 is the isotope coordinate.  The
    mapper therefore projects dense codes by deleting that digit without
    decoding and re-encoding the other coordinates.
    """
    bases = np.asarray(composition_bases, dtype=np.int64).reshape(-1)
    if bases.shape != (5,) or np.any(bases < 1):
        raise ValueError("the MATLAB annotation model requires five positive bases")
    if int(code) == 0:
        raise ValueError("zero is not a dense annotation code")
    value = abs(int(code)) - 1
    full_size = int(np.prod(bases, dtype=np.int64))
    if value >= full_size:
        raise ValueError("dense code exceeds the composition radix")
    suffix_size = int(np.prod(bases[3:5], dtype=np.int64))
    isotope_block = int(np.prod(bases[2:5], dtype=np.int64))
    return value % suffix_size + (value // isotope_block) * suffix_size + 1


def project_dense_codes_to_rt(
    design_prediction: np.ndarray,
    codes: np.ndarray,
    composition_bases: tuple[int, ...] | np.ndarray,
    sibling_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Project fitted design values onto dense codes and sibling positions.

    ``design_prediction`` is the evaluated ``Xp * beta + preprocessing`` matrix
    with one column per sibling. Negative internal-standard codes intentionally
    share the same composition row as their corresponding positive radix code.
    """
    prediction = np.asarray(design_prediction, dtype=float)
    if prediction.ndim == 1:
        prediction = prediction.reshape(-1, 1)
    if prediction.ndim != 2:
        raise ValueError("design prediction must be a vector or matrix")
    code_array = np.asarray(codes, dtype=np.int64).reshape(-1)
    if sibling_positions is None:
        siblings = np.ones(code_array.shape, dtype=np.int64)
    else:
        siblings = np.asarray(sibling_positions, dtype=np.int64).reshape(-1)
    if siblings.shape != code_array.shape:
        raise ValueError("one sibling position is required per dense code")
    if np.any(siblings < 1) or np.any(siblings > prediction.shape[1]):
        raise ValueError("sibling position is outside the prediction matrix")
    bases = np.asarray(composition_bases, dtype=np.int64).reshape(-1)
    if bases.shape != (5,) or np.any(bases < 1):
        raise ValueError("the MATLAB annotation model requires five positive bases")
    if np.any(code_array == 0):
        raise ValueError("zero is not a dense annotation code")
    values = np.abs(code_array) - 1
    full_size = int(np.prod(bases, dtype=np.int64))
    if np.any(values >= full_size):
        raise ValueError("dense code exceeds the composition radix")
    suffix_size = int(np.prod(bases[3:5], dtype=np.int64))
    isotope_block = int(np.prod(bases[2:5], dtype=np.int64))
    rows = values % suffix_size + (values // isotope_block) * suffix_size + 1
    if np.any(rows > prediction.shape[0]):
        raise ValueError("design prediction does not contain the projected row")
    return prediction[rows - 1, siblings - 1]


def design_row_from_composition(
    composition_one_based: np.ndarray,
    composition_bases: tuple[int, ...] | np.ndarray,
) -> np.ndarray:
    """Reference projection from explicit one-based five-digit compositions."""
    values = np.asarray(composition_one_based, dtype=np.int64)
    bases = np.asarray(composition_bases, dtype=np.int64).reshape(-1)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 5 or bases.shape != (5,):
        raise ValueError("composition rows and bases must contain five dimensions")
    if np.any(values < 1) or np.any(values > bases):
        raise ValueError("composition digit is outside its radix")
    retained = values[:, [0, 1, 3, 4]] - 1
    retained_bases = bases[[0, 1, 3, 4]]
    strides = np.asarray(
        [np.prod(retained_bases[index + 1 :], dtype=np.int64)
         for index in range(retained_bases.size)],
        dtype=np.int64,
    )
    return retained @ strides + 1
