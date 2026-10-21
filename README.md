# LASER Standard Lipidomics

Standalone LASER tool for conventional XCMS lipidomics. This repository was
extracted from the standard-lipidomics module in
[`YRJJaspers/LASER-4D-lipidomics`](https://github.com/YRJJaspers/LASER-4D-lipidomics)
at commit `e447bac` on branch `codex/future-ready-review-fixes`.

This tool contains only the standard workflow and its shared scientific
dependencies. It does not expose the LASER 4D, MZmine-processing, Bruker,
targeted-extraction, or dataset-specific workflows.

## Included workflow

- Independent XCMS precursor-mass calibration for RP/NP and POS/NEG streams.
- Automatic MATLAB-equivalent XCMS annotation (no MATLAB installation needed).
- LASER reference-matcher annotation.
- Composition-range and grid-coherence policies.
- Interactive annotation review, reports, exports, saved projects, and
  post-annotation untargeted comparison.
- A small headless CLI for either annotation engine.

## Run the app

Python 3.10–3.13 is supported.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
streamlit run app/app.py
```

On macOS, you can instead double-click `Launch LASER Standard.command`. On
Windows, run `Setup (Windows).bat` once and then `Launch LASER Standard.bat`.

Set `LASER_STANDARD_PROJECT` to prefill the XCMS project folder.

## Headless use

```bash
laser-standard /path/to/Processing/Annotation \
  --engine matlab \
  --phase both \
  --output results/standard_lipidomics
```

For the input contract and scientific behavior, see
[`docs/STANDARD_LIPIDOMICS_MODE.md`](docs/STANDARD_LIPIDOMICS_MODE.md) and
[`docs/STANDARD_LIPIDOMICS_ANNOTATION_REFERENCE.md`](docs/STANDARD_LIPIDOMICS_ANNOTATION_REFERENCE.md).

## Development

The agreed next steps and validation gates are in the
[annotation improvement plan](docs/ANNOTATION_IMPROVEMENT_PLAN.md).

```bash
python -m pip install -e '.[dev]'
pytest
ruff check app
```

The source repository is not modified by this extraction. Future changes can
be developed and released independently in this repository.
