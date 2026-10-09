# Vendored upstream: AI_Lipidomics_Pipeline

`vendor/` is a verbatim copy of part of the `headless_annotation` package from
the reverse-engineering repository of the Amsterdam UMC MATLAB/C/MEX lipidomics
annotation application.

| | |
|---|---|
| Repository | `Core-Facility-Metabolomics-AUMC/AI_Lipidomics_Pipeline` (private) |
| Commit | `e6fe52d1b0987cbe7a081c2b2222740008098da3` |
| Commit subject | `test: align GUI checks with annotation workflow` |
| Commit date | 2026-07-16 |
| Vendored on | 2026-07-31 |

## Why vendored rather than depended on

LASER ships as a folder a lab user launches by double-clicking
`Launch LASER.command`. A submodule or private package dependency would add a
setup step and a credential requirement to that path. The annotation subset is
self-contained and changes rarely, so it is copied in with its provenance
recorded here.

## What was copied

Only the modules the annotation path actually reaches. The GUI, viewer, review,
persistence, untargeted and native-store layers were **not** copied — which is
also why this subset needs no `h5py`.

```
accepted_mapper          engine                   project
annotation_engine        export                   reference
annotation_runner        internal_standard_model  reference_loader
candidate_engine         legacy_loader            rt_model
composition_initializer  mapper                   rt_projection
compound_model           membership_model         rtdb_model
                         model_dispatch           spline
                         naming                   stage_loop
                                                  target_model
```

`vendor/__init__.py` is LASER's own, and is the only file in `vendor/` that is
not upstream's.

## Reference data

`reference_data/matlab_pipeline/` holds the MATLAB inputs this engine reads,
copied from the same commit's `Current_version/`:

- `Code/param.json` — annotation parameters
- `DataBase/*.tsv` — lipid, metabolite, internal-standard and isotope tables
- `RTDB/lipidomics_*.tsv` — the retention-time database

The project's own `internal_standards.tsv` is read from the project folder, not
from here.

## Do not edit `vendor/`

These modules exist to reproduce MATLAB's arithmetic exactly, and are validated
upstream against captured MATLAB oracles. A local fix silently breaks that
guarantee. Report the problem upstream and re-vendor.

## Re-vendoring

1. Copy the module list above from upstream `headless_annotation/` into
   `vendor/`, keeping `vendor/__init__.py`.
2. Copy `Current_version/{Code/param.json,DataBase,RTDB}` into
   `reference_data/matlab_pipeline/`.
3. Update the commit row in this file.
4. Run `PYTHONPATH=app python3 -m unittest discover -s app/tests`, then
   re-run the coverage comparison and confirm recall has not regressed:

```bash
python3 app/tools/compare_matlab_annotation.py --all-projects /path/to/example_data_CFM
```
