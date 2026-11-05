"""Project paths shared by input discovery and saved workflows."""

from pathlib import Path


def xcms_project_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if root.is_file():
        root = root.parent
    # Current and historical archive layouts.
    if root.parent.name.casefold() in {"calibration", "laser", "untargeted"} and root.parent.parent.name.casefold() == "archive":
        return xcms_project_root(root.parent.parent.parent)
    if root.name.casefold() == "annotation" and root.parent.name.casefold() in {"processing", "laser"}:
        return root.parent.parent
    if root.name.casefold() == "untargeted" and root.parent.name.casefold() == "laser":
        return root.parent.parent
    if root.name.casefold() in {"xcms", "metadata", "calibration", "laser", "untargeted", "laser_calibrated_xcms"}:
        return root.parent
    return root


def workflow_folder(project: str | Path, workflow: str) -> Path:
    names = {"Annotation": "LASER", "Untargeted": "Untargeted", "Calibration": "Calibration"}
    if workflow not in names:
        raise ValueError(f"Unknown LASER workflow: {workflow}")
    return xcms_project_root(project) / names[workflow]


def available_calibrations(project: str | Path) -> dict[str, Path]:
    """Current first, followed by archived calibration exports, newest first."""
    root = xcms_project_root(project)
    candidates = [("Current calibration", root / "Calibration")]
    for archive in (root / "Archive" / "Calibration", root / "LASER" / "Archive" / "Calibration"):
        if archive.is_dir():
            candidates.extend((f"Archived calibration · {path.name}", path)
                              for path in sorted(archive.iterdir(), reverse=True) if path.is_dir())
    candidates.append(("Legacy calibration", root / "LASER_calibrated_XCMS"))
    return {label: path for label, path in candidates
            if (path / "LASER_standard_mass_calibration.json").is_file()}
