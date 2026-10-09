"""Vendored subset of AI_Lipidomics_Pipeline's ``headless_annotation`` package.

**Do not edit these modules.** They are a verbatim copy of the upstream
reverse-engineering of the Amsterdam UMC MATLAB annotation application, and
their value is that they reproduce MATLAB's arithmetic exactly. Any local fix
silently breaks that guarantee. Report problems upstream and re-vendor.

Provenance and the exact copied module list are recorded in ``UPSTREAM.md``
next to this package. LASER's own adapter code lives one directory up, in
``app/matlab_annotation``.

Only the modules the annotation path actually reaches are vendored — the GUI,
viewer, review, persistence and native-store layers are not, which is also why
this subset needs no ``h5py``.
"""
