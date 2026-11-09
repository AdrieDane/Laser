"""A phase without RT alignment anchors fails with a readable reason.

LipidLive annotates single injections, and a blank or weak run has no
13C-supported peak near an RT-database lipid.
"""

import numpy as np
import pytest

from matlab_annotation.vendor.rtdb_model import fit_project_rt_correction


def test_no_alignment_pairs_names_the_reason():
    peaks = np.array([[700.5, 600.0], [-800.5, 650.0]])
    targets = np.array([[700.5, 610.0, 1.0]])
    pairs = np.empty((0, 2), dtype=np.int64)
    with pytest.raises(ValueError, match="no RT alignment anchors"):
        fit_project_rt_correction(peaks, targets, pairs, study_count=1)
