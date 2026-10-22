"""Two-stage collection. Annotation algorithms and source data are never edited."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import uuid

from .readers import Workbook, load_rt_predictions, number, table_records

STREAMS = ("RP_Pos", "RP_Neg", "NP_Pos", "NP_Neg")
YEARS = ("2024", "2025", "2026")
DEFAULT_SOURCE = r"G:\divg\GMZ Tandem ms biochemie\Lipidomics"
SKIP = {
    ".git",
    "$recycle.bin",
    "system volume information",
    "raw",
    "raw_data",
    "rawdata",
    "eics",
    "annotated_peakdiagrams",
    "peakdiagrams",
    "_statistics_archive",
    "statistics",
    "node_modules",
    ".venv",
}
SMALL = {
    "internal_standards.tsv",
    "group_definition.tsv",
    "analyse_ms_data.m",
    "lipids.tsv",
    "ms_metadata.mat",
    "ms.mat",
}
SCHEMA = "laser-dataset-scan-v1"


@dataclass(frozen=True)
class Limits:
    min_annotations: int = 50
    min_classes: int = 3
    min_coverage: float = 0.80
    max_ppm: float = 3.0
    max_rt_seconds: float = 10.0
    min_within_limits: float = 0.90
    max_median_ppm: float = 1.5
    max_median_rt_seconds: float = 5.0
    min_profile_correlation: float = 0.995
    min_profile_samples: int = 4

    def validate(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid screening limit: {key}")
        for key in ("min_annotations", "min_classes", "min_profile_samples"):
            if getattr(self, key) != int(getattr(self, key)):
                raise ValueError(f"{key} must be a whole number")
        if self.min_profile_samples < 3:
            raise ValueError(
                "At least 3 samples are required for a profile correlation"
            )
        for key in ("min_coverage", "min_within_limits", "min_profile_correlation"):
            if getattr(self, key) > 1:
                raise ValueError(f"{key} must be a fraction no greater than 1")


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise InterruptedError("Cancelled by user")


def sha256(path, cancel=None):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(4 * 1024**2):
            check_cancel(cancel)
            value.update(chunk)
    return value.hexdigest()


def human_size(size):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:,.1f} {unit}"
        size /= 1024


def canonical(value):
    text = str(value or "").strip().removesuffix(" (IS)").removeprefix("IS: ")
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip()


def mass_name(value):
    text = canonical(value).replace(" ", "")
    text = re.sub(r"^(PC|PE)\[O\]\(", r"\1(O-", text)
    text = re.sub(r"^(PC|PE)\[P\]\(", r"\1(P-", text)
    return text


def canonical_adduct(value):
    return re.sub(r"\]\((\d+)([+-])\)$", r"]\1\2", str(value).strip())


ATOMS = {
    "C": 12.0,
    "H": 1.00782503223,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "Cl": 34.968852682,
    "Na": 22.989769282,
    "K": 38.9637064864,
    "F": 18.998403163,
    "Br": 78.9183376,
}
PROTON = 1.007276466621
ADDUCTS = {
    "[M+H]+": (PROTON, 1),
    "[M-H]-": (-PROTON, 1),
    "[M+NH4]+": (18.033825553, 1),
    "[M+Na]+": (22.989220708, 1),
    "[M+K]+": (38.963157906, 1),
    "[M+HCOO]-": (44.998201, 1),
    "[M+CH3COO]-": (59.013851, 1),
    "[M+Cl]-": (34.969401, 1),
    "[M+2H]2+": (2 * PROTON, 2),
    "[M-2H]2-": (-2 * PROTON, 2),
    "[M+H-H2O]+": (PROTON - 18.010564684, 1),
}


def formula_mass(formula):
    text = str(formula).strip()
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", text)
    if not parts or "".join(a + n for a, n in parts) != text:
        return None
    if any(a not in ATOMS for a, _ in parts):
        return None
    return sum(ATOMS[a] * int(n or "1") for a, n in parts)


def reference_masses(path):
    result = defaultdict(set)
    if path and Path(path).is_file():
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                mz = number(row.get("mz"))
                if mz and mz > 0:
                    result[
                        (mass_name(row["name"]), canonical_adduct(row["adduct"]))
                    ].add(mz)
    return {key: next(iter(v)) for key, v in result.items() if len(v) == 1}


def curated_rows(path):
    result, formulas = [], defaultdict(set)
    with Workbook(path) as book:
        if "Primary list (normalized)" in book.sheets:
            for row in book.records("Primary list (normalized)"):
                mass = formula_mass(row.get("chem formula", ""))
                if row.get("name") and mass:
                    formulas[mass_name(row["name"])].add(mass)
        for sheet in ("Primary list (raw)", "Equivalents (raw)"):
            if sheet not in book.sheets:
                continue
            for excel_row, row in enumerate(book.records(sheet), 2):
                if not row.get("name"):
                    continue  # The second header line is not an annotation.
                phase = str(row.get("column", "")).strip().upper()
                adduct = canonical_adduct(row.get("adduct", ""))
                if phase not in ("RP", "NP") or adduct[-1:] not in ("+", "-"):
                    raise ValueError(
                        f"Unknown phase/polarity in {sheet}, row {excel_row}"
                    )
                mz, rt = number(row.get("m/z")), number(row.get("RT"))
                if not mz or rt is None or rt <= 0:
                    raise ValueError(
                        f"Invalid measured m/z or RT in {sheet}, row {excel_row}"
                    )
                result.append(
                    dict(
                        stream=phase + ("_Pos" if adduct.endswith("+") else "_Neg"),
                        name=row["name"],
                        lipid_class=row.get("class", ""),
                        adduct=adduct,
                        mz=abs(mz),
                        rt_seconds=rt,
                        profile={
                            canonical(k): number(v)
                            for k, v in row.items()
                            if k
                            not in {
                                "name",
                                "adduct",
                                "m/z",
                                "RT",
                                "class",
                                "column",
                                "IS",
                                "",
                            }
                        },
                        sheet=sheet,
                        excel_row=excel_row,
                    )
                )
    if not result:
        raise ValueError("No curated rows in Primary list (raw)/Equivalents (raw)")
    return result, {k: next(iter(v)) for k, v in formulas.items() if len(v) == 1}


def relevant(path):
    name = path.name.casefold()
    return (
        name in SMALL | {"ms_data.xlsx", "ms_data.mat"}
        or bool(re.fullmatch(r"peakgrouplist_(rp|np)_(pos|neg)\.txt", name))
        or bool(re.search(r"_(rp|np)\.(xlsx|csv)$", name))
    )


def discovery(source, cancel=None, progress=lambda message: None):
    source = Path(source).resolve(strict=True)
    if not source.is_dir():
        raise ValueError("Source must be a folder")
    if source.name.isdigit() and len(source.name) == 4:
        if source.name not in YEARS:
            raise ValueError("Only year folders 2024, 2025 and 2026 are allowed")
        years = [source]
    else:
        years = [
            source / year
            for year in YEARS
            if (source / year).is_dir()
            and not (source / year).is_symlink()
            and not getattr((source / year).lstat(), "st_file_attributes", 0) & 0x400
        ]
    if not years:
        raise ValueError(
            "Choose Lipidomics with 2024/2025/2026 subfolders, or one of those year folders"
        )
    groups, issues = defaultdict(list), []
    for year in years:
        progress(f"Discovering projects in {year.name}")

        def error(exc):
            issues.append(f"Cannot read {exc.filename}: {exc.strerror}")

        for folder, dirs, files in os.walk(year, followlinks=False, onerror=error):
            check_cancel(cancel)
            base = Path(folder)
            dirs[:] = sorted(
                d
                for d in dirs
                if d.casefold() not in SKIP
                and not d.startswith("LASER_")
                and not (base / d).is_symlink()
                and not getattr((base / d).lstat(), "st_file_attributes", 0) & 0x400
            )
            ancestors = [
                p
                for p in (base, *base.parents)
                if p != year
                and p.is_relative_to(year)
                and re.match(r"\d{2}P\d{4}(?:_|$)", p.name, re.I)
            ]
            project = ancestors[0] if ancestors else None
            if project:
                groups.setdefault(project, [])
            for name in files:
                path = base / name
                if path.is_symlink() or not relevant(path):
                    continue
                if project is None and name.casefold() == "ms_data.xlsx":
                    project = (
                        base.parent.parent
                        if base.name.casefold() == "annotation"
                        and base.parent.name.casefold() == "processing"
                        else base
                    )
                if project:
                    groups[project].append(path)
    return source, groups, issues


def choose(files, anchor, predicate, role):
    matches = [p for p in files if predicate(p)]
    direct = [p for p in matches if p.parent == anchor]
    if len(direct) == 1:
        return direct[0]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Missing {role}")
    raise ValueError(f"Multiple versions of {role}; cannot safely mix processing runs")


def metadata_maps(paths):
    result = {s: {} for s in STREAMS}
    for phase, path in paths.items():
        for row in table_records(path):
            stream = str(row.get("Mode", "")).strip()
            if stream not in result or not stream.startswith(phase):
                continue
            if str(row.get("Sample Type", "study")).casefold().strip() not in (
                "study",
                "sample",
                "",
            ):
                continue
            filename = str(row.get("File Name", "")).strip()
            name = canonical(row.get("Sample Name", ""))
            protein = number(row.get("Protein"))
            if filename and name and protein and protein > 0:
                if filename in result[stream]:
                    raise ValueError(
                        f"Duplicate metadata filename in {stream}: {filename}"
                    )
                result[stream][filename] = (name, protein)
    return result


def correlation(left, right):
    if len(left) < 3:
        return None
    x = [math.log1p(max(0, v)) for v in left]
    y = [math.log1p(max(0, v)) for v in right]
    mx, my = statistics.fmean(x), statistics.fmean(y)
    xx, yy = sum((v - mx) ** 2 for v in x), sum((v - my) ** 2 for v in y)
    if xx <= 1e-20 or yy <= 1e-20:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(xx * yy)


def read_features(path, sample_map, cancel=None, minimum_samples=4):
    result, ids = [], set()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"name", "mzmed", "rtmed"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Peak list missing columns {sorted(required)}")
        columns = [name for name in sample_map if name in reader.fieldnames]
        if len(columns) < minimum_samples:
            raise ValueError(
                f"Fewer than {minimum_samples} metadata-matched sample columns in peak list"
            )
        for line, row in enumerate(reader, 2):
            if line % 4096 == 0:
                check_cancel(cancel)
            fid, mz, rt = (
                str(row.get("name", "")).strip(),
                number(row.get("mzmed")),
                number(row.get("rtmed")),
            )
            if not fid or fid in ids or mz is None or mz <= 0 or rt is None or rt <= 0:
                raise ValueError(
                    f"Invalid/duplicate original feature or coordinates at peak-list row {line}"
                )
            ids.add(fid)
            result.append((mz, rt * 60, fid, [number(row.get(c)) for c in columns]))
    result.sort(key=lambda r: (r[0], r[1], r[2]))
    return result, columns


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = fraction * (len(values) - 1)
    lower = int(index)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (
        index - lower
    )


def screen_stream(
    stream,
    rows,
    formulas,
    masses,
    prediction_rows,
    peak_path,
    samples,
    limits,
    cancel=None,
):
    features, columns = read_features(
        peak_path, samples, cancel, limits.min_profile_samples
    )
    mzs = [r[0] for r in features]
    predicted = defaultdict(list)
    for mz, rt, pred in prediction_rows:
        if (
            (mz > 0) != stream.endswith("Pos")
            or not all(math.isfinite(v) for v in (mz, rt, pred))
            or pred <= 0
        ):
            continue
        predicted[(round(abs(mz), 5), round(rt, 3))].append(pred)
    errors, mass_errors, rt_errors, proofs = [], [], [], []
    used = set()
    for count, row in enumerate(rows):
        if count % 100 == 0:
            check_cancel(cancel)
        mz, rt = row["mz"], row["rt_seconds"]
        delta = mz * 3e-6
        candidates = []
        for feature in features[
            bisect_left(mzs, mz - delta) : bisect_right(mzs, mz + delta)
        ]:
            if abs(feature[1] - rt) > 10:
                continue
            left, right = [], []
            for col, observed in zip(columns, feature[3]):
                name, factor = samples[col]
                expected = row["profile"].get(name)
                if observed is not None and expected is not None:
                    left.append(observed / factor)
                    right.append(expected)
            corr = correlation(left, right)
            if (
                corr is not None
                and corr >= limits.min_profile_correlation
                and len(left) >= limits.min_profile_samples
            ):
                candidates.append((corr, feature))
        candidates.sort(key=lambda item: (-item[0], abs(item[1][1] - rt), item[1][2]))
        chosen = None
        if candidates and (
            len(candidates) == 1 or candidates[0][0] - candidates[1][0] > 1e-5
        ):
            chosen = candidates[0][1]
        if chosen and chosen[2] in used:
            chosen = None
        if chosen:
            used.add(chosen[2])
        theoretical = masses.get((mass_name(row["name"]), row["adduct"]))
        neutral = formulas.get(mass_name(row["name"]))
        if theoretical is None and neutral and row["adduct"] in ADDUCTS:
            shift, charge = ADDUCTS[row["adduct"]]
            theoretical = (neutral + shift) / charge
        ppm = (
            (chosen[0] - theoretical) / theoretical * 1e6
            if chosen and theoretical
            else None
        )
        predictions = predicted.get((round(mz, 5), round(rt, 3)), [])
        residual = None
        if chosen and predictions and max(predictions) - min(predictions) <= 0.01:
            residual = chosen[1] - statistics.median(predictions)
        if ppm is not None:
            mass_errors.append(abs(ppm))
        if residual is not None:
            rt_errors.append(abs(residual))
        proofs.append(
            dict(
                stream=stream,
                name=row["name"],
                adduct=row["adduct"],
                source_sheet=row["sheet"],
                source_excel_row=row["excel_row"],
                original_feature_id=chosen[2] if chosen else "",
                profile_correlation=candidates[0][0] if chosen else None,
                raw_mass_error_ppm=ppm,
                stored_fit_RT_error_seconds=residual,
                matched_original_feature=chosen is not None,
            )
        )
    n = len(rows)
    classes = len({r["lipid_class"] for r in rows if r["lipid_class"]})
    mapping = len(used) / n if n else 0
    mass_coverage, rt_coverage = (
        len(mass_errors) / n if n else 0,
        len(rt_errors) / n if n else 0,
    )
    within_ppm = (
        sum(v < limits.max_ppm for v in mass_errors) / len(mass_errors)
        if mass_errors
        else 0
    )
    within_rt = (
        sum(v <= limits.max_rt_seconds for v in rt_errors) / len(rt_errors)
        if rt_errors
        else 0
    )
    median_ppm, median_rt = quantile(mass_errors, 0.5), quantile(rt_errors, 0.5)
    if n < limits.min_annotations:
        errors.append(f"Only {n} curated entries; require {limits.min_annotations}")
    if classes < limits.min_classes:
        errors.append(f"Only {classes} curated classes; require {limits.min_classes}")
    for label, coverage in (
        ("Original-feature matching", mapping),
        ("Mass assessment", mass_coverage),
        ("RT assessment", rt_coverage),
    ):
        if coverage < limits.min_coverage:
            errors.append(
                f"{label} available for only {coverage:.1%}; require {limits.min_coverage:.0%}"
            )
    if (
        within_ppm < limits.min_within_limits
        or median_ppm is None
        or median_ppm > limits.max_median_ppm
    ):
        errors.append(
            f"Mass accuracy: {within_ppm:.1%} below {limits.max_ppm:g} ppm; median {median_ppm}"
        )
    if (
        within_rt < limits.min_within_limits
        or median_rt is None
        or median_rt > limits.max_median_rt_seconds
    ):
        errors.append(
            f"RT accuracy: {within_rt:.1%} within {limits.max_rt_seconds:g} s; median {median_rt}"
        )
    return dict(
        stream=stream,
        status="PASS" if not errors else "REVIEW",
        reasons=errors,
        curated_entries=n,
        curated_classes=classes,
        original_features=len(features),
        matched_manual_features=len(used),
        feature_match_percent=100 * mapping,
        mass_assessed=len(mass_errors),
        mass_coverage_percent=100 * mass_coverage,
        mass_within_limit_percent=100 * within_ppm,
        median_absolute_ppm=median_ppm,
        p90_absolute_ppm=quantile(mass_errors, 0.9),
        rt_assessed=len(rt_errors),
        rt_coverage_percent=100 * rt_coverage,
        rt_within_limit_percent=100 * within_rt,
        median_absolute_rt_seconds=median_rt,
        p90_absolute_rt_seconds=quantile(rt_errors, 0.9),
    ), proofs


def file_record(path, source_root, role, cancel=None):
    path = Path(path)
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_relative_to(source_root):
        raise ValueError(f"Linked or out-of-source file is not allowed: {path}")
    before = path.stat()
    digest = sha256(path, cancel)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed while scanning: {path}")
    return dict(
        source_relative=path.relative_to(source_root).as_posix(),
        name=path.name,
        role=role,
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        sha256=digest,
    )


def inspect_dataset(
    project,
    files,
    anchor,
    source_root,
    masses,
    limits=Limits(),
    cancel=None,
    progress=lambda message: None,
):
    limits.validate()
    relative = anchor.relative_to(source_root).as_posix()
    identity = hashlib.sha256(relative.encode()).hexdigest()[:12]
    result = dict(
        id=identity,
        project=project.name,
        source_relative=relative,
        status="INCOMPLETE",
        reasons=[],
        modes=[],
        files=[],
        total_bytes=0,
        proofs=[],
    )
    try:
        manual = choose(
            files,
            anchor,
            lambda p: p.name.casefold() == "ms_data.xlsx",
            "curated MS_data.xlsx",
        )
        mat = choose(
            files,
            anchor,
            lambda p: p.name.casefold() == "ms_data.mat",
            "saved RT predictions MS_data.mat",
        )
        peaks = {
            s: choose(
                files,
                anchor,
                lambda p, s=s: p.name.casefold() == f"peakgrouplist_{s}.txt".casefold(),
                s + " original peak list",
            )
            for s in STREAMS
        }
        meta = {
            phase: choose(
                files,
                anchor,
                lambda p, phase=phase: bool(
                    re.search(f"_{phase}\\.(xlsx|csv)$", p.name, re.I)
                ),
                phase + " sample metadata",
            )
            for phase in ("RP", "NP")
        }
        selected = [
            (manual, "curated annotations and profiles"),
            (mat, "saved RT predictions"),
            *[(p, s + " original features") for s, p in peaks.items()],
            *[
                (p, phase + " sample-to-file mapping and Protein factors")
                for phase, p in meta.items()
            ],
        ]
        # Keep small files that affect annotation settings/standards or allow
        # future MATLAB decoding. Do not collect scripts other than this known
        # configuration file; it is copied as data, never executed.
        for name in sorted(SMALL):
            candidates = [p for p in files if p.name.casefold() == name]
            if candidates:
                selected.append(
                    (
                        choose(
                            files,
                            anchor,
                            lambda p, name=name: p.name.casefold() == name,
                            name,
                        ),
                        "standards/settings or saved annotation definitions",
                    )
                )
        rows, formulas = curated_rows(manual)
        samples = metadata_maps(meta)
        predictions = load_rt_predictions(mat)
        for stream in STREAMS:
            progress(f"Checking {project.name}: {stream}")
            try:
                mode, proofs = screen_stream(
                    stream,
                    [r for r in rows if r["stream"] == stream],
                    formulas,
                    masses,
                    predictions[stream[:2]],
                    peaks[stream],
                    samples[stream],
                    limits,
                    cancel,
                )
                result["modes"].append(mode)
                result["proofs"].extend(proofs)
            except InterruptedError:
                raise
            except Exception as exc:
                result["modes"].append(
                    dict(stream=stream, status="REVIEW", reasons=[str(exc)])
                )
        progress(f"Measuring exact copy size and checking file hashes: {project.name}")
        for path, role in selected:
            result["files"].append(file_record(path, source_root, role, cancel))
        names = [f["name"].casefold() for f in result["files"]]
        if len(names) != len(set(names)):
            raise ValueError("Selected files have conflicting output filenames")
        result["total_bytes"] = sum(f["size"] for f in result["files"])
        result["status"] = (
            "PASS" if all(m["status"] == "PASS" for m in result["modes"]) else "REVIEW"
        )
        result["reasons"] = [
            f"{m['stream']}: {r}" for m in result["modes"] for r in m["reasons"]
        ]
    except InterruptedError:
        raise
    except Exception as exc:
        result["reasons"].append(str(exc))
    present = {m["stream"] for m in result["modes"]}
    result["modes"].extend(
        dict(stream=s, status="NOT_CHECKED", reasons=list(result["reasons"]))
        for s in STREAMS
        if s not in present
    )
    return result


def scan(
    source,
    reference_path=None,
    limits=Limits(),
    cancel=None,
    progress=lambda message: None,
):
    limits.validate()
    source_root, groups, issues = discovery(source, cancel, progress)
    masses = reference_masses(reference_path)
    datasets = []
    for project, files in sorted(groups.items(), key=lambda item: str(item[0])):
        anchors = sorted(
            {p.parent for p in files if p.name.casefold() == "ms_data.xlsx"}
        ) or [project]
        for anchor in anchors:
            check_cancel(cancel)
            # Different annotation snapshots in a project remain separate;
            # direct files win, missing/ambiguous roles never choose newest.
            record = inspect_dataset(
                project, files, anchor, source_root, masses, limits, cancel, progress
            )
            datasets.append(record)
    # Flag identical input collections, even if their folder names differ.
    signatures = {}
    for dataset in datasets:
        if not dataset["files"]:
            continue
        signature = hashlib.sha256(
            json.dumps(
                sorted((f["role"], f["sha256"]) for f in dataset["files"])
            ).encode()
        ).hexdigest()
        if signature in signatures:
            dataset["duplicate_of"] = signatures[signature]
            dataset["status"] = "DUPLICATE"
            dataset["reasons"].append(
                "Identical selected source files already found in another folder"
            )
        else:
            signatures[signature] = dataset["id"]
    return dict(
        schema=SCHEMA,
        scan_id=uuid.uuid4().hex,
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_root=str(source_root),
        allowed_years=list(YEARS),
        limits=asdict(limits),
        reference_sha256=sha256(reference_path)
        if reference_path and Path(reference_path).is_file()
        else None,
        discovery_issues=issues,
        datasets=datasets,
        note="PASS is configurable screening suitability, not proof that curation is correct. RT uses saved fitted predictions, not held-out validation.",
    )


def csv_safe(row):
    return {
        k: (
            "'" + v
            if isinstance(v, str) and v.startswith(("=", "+", "-", "@", "\t"))
            else v
        )
        for k, v in row.items()
    }


def write_reports(plan, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    summary = []
    for d in plan["datasets"]:
        proof = d.pop("proofs", [])
        if proof:
            with (directory / f"{d['id']}_feature_checks.csv").open(
                "w", newline="", encoding="utf-8-sig"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(proof[0]))
                writer.writeheader()
                writer.writerows(csv_safe(row) for row in proof)
        for stream in STREAMS:
            mode = next(
                (m for m in d["modes"] if m["stream"] == stream),
                dict(stream=stream, status="NOT_CHECKED", reasons=d["reasons"]),
            )
            summary.append(
                {
                    "dataset": d["project"],
                    "dataset_id": d["id"],
                    "source_relative": d["source_relative"],
                    "dataset_status": d["status"],
                    "selected_files_bytes": d["total_bytes"],
                    **{
                        k: ("; ".join(v) if isinstance(v, list) else v)
                        for k, v in mode.items()
                    },
                }
            )
    (directory / "scan_plan.json").write_text(
        json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8"
    )
    fields = list(dict.fromkeys(k for row in summary for k in row))
    with (directory / "scan_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_safe(row) for row in summary)
    return directory / "scan_plan.json"


def safe_relative(text):
    # Plans may be edited or supplied from elsewhere; Windows and POSIX path
    # escapes must both be rejected regardless of the testing platform.
    if (
        not text
        or "\\" in text
        or ":" in text
        or text.startswith("/")
        or any(v in ("", ".", "..") for v in text.split("/"))
    ):
        raise ValueError("Unsafe relative source path in scan plan")
    return Path(text)


def copy_approved(
    plan,
    destination,
    selected_ids,
    *,
    confirm=False,
    cancel=None,
    progress=lambda message: None,
):
    if not confirm:
        raise ValueError("Copy requires explicit approval after scanning")
    if plan.get("schema") != SCHEMA or plan.get("allowed_years") != list(YEARS):
        raise ValueError("Unsupported scan plan or year scope")
    Limits(**plan["limits"]).validate()
    if isinstance(destination, str) and not destination.strip():
        raise ValueError("Choose a destination folder before copying")
    source = Path(plan["source_root"]).resolve(strict=True)
    destination = Path(destination).expanduser().resolve()
    if (
        destination == source
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError("Destination must be separate from the source tree")
    selected_ids = set(selected_ids)
    datasets = [d for d in plan["datasets"] if d["id"] in selected_ids]
    if (
        not datasets
        or len(datasets) != len(selected_ids)
        or any(
            d["status"] != "PASS"
            or not d["files"]
            or {m["stream"] for m in d["modes"]} != set(STREAMS)
            or any(m["status"] != "PASS" for m in d["modes"])
            for d in datasets
        )
    ):
        raise ValueError("Select only datasets that passed all four modes")
    required = sum(f["size"] for d in datasets for f in d["files"])
    existing_parent = destination
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if shutil.disk_usage(existing_parent).free < required + 16 * 1024**2:
        raise ValueError("Not enough free destination space for the approved selection")
    work = []
    for d in datasets:
        if not re.fullmatch(r"[0-9a-f]{12}", d["id"]):
            raise ValueError("Unsafe dataset ID in scan plan")
        target_name = (
            re.sub(r"[^A-Za-z0-9_.-]", "_", d["project"])[:100] + "__" + d["id"]
        )
        target = destination / target_name
        if target.exists():
            raise FileExistsError(f"Will not overwrite existing dataset: {target}")
        for f in d["files"]:
            relative = safe_relative(f["source_relative"])
            if source.name not in YEARS and relative.parts[0] not in YEARS:
                raise ValueError(
                    "Source file is outside the approved 2024–2026 year folders"
                )
            candidate = source / relative
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(source) or candidate.is_symlink():
                raise ValueError("Source file escaped the scanned source tree")
            if (
                Path(f["name"]).name != f["name"]
                or "\\" in f["name"]
                or ":" in f["name"]
            ):
                raise ValueError("Unsafe destination filename")
            if f["name"] != candidate.name or not relevant(candidate):
                raise ValueError(
                    "Only recognised necessary dataset files may be copied"
                )
            stat = candidate.stat()
            if (
                stat.st_size != f["size"]
                or stat.st_mtime_ns != f["mtime_ns"]
                or sha256(candidate, cancel) != f["sha256"]
            ):
                raise ValueError(
                    f"Source changed since scan; rescan before copying: {relative}"
                )
            work.append((d, f, candidate, target))
    # Nothing in the destination is created until every selected source passes
    # freshness/integrity checks and the user has explicitly approved copying.
    destination.mkdir(parents=True, exist_ok=True)
    completed = []
    for d in datasets:
        check_cancel(cancel)
        jobs = [w for w in work if w[0]["id"] == d["id"]]
        target = jobs[0][3]
        staging = destination / (".partial_" + target.name + "_" + uuid.uuid4().hex[:8])
        staging.mkdir(exist_ok=False)
        try:
            for _, f, candidate, _ in jobs:
                check_cancel(cancel)
                progress(f"Copying {d['project']}: {f['name']}")
                digest = hashlib.sha256()
                with (
                    candidate.open("rb") as src,
                    (staging / f["name"]).open("xb") as dst,
                ):
                    while chunk := src.read(4 * 1024**2):
                        check_cancel(cancel)
                        dst.write(chunk)
                        digest.update(chunk)
                if (
                    digest.hexdigest() != f["sha256"]
                    or sha256(staging / f["name"], cancel) != f["sha256"]
                ):
                    raise ValueError(f"Copy verification failed: {f['name']}")
            (staging / "collection_manifest.json").write_text(
                json.dumps(
                    dict(
                        scan_id=plan["scan_id"],
                        source=d["source_relative"],
                        limits=plan["limits"],
                        mode_checks=d["modes"],
                        files=d["files"],
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )
            if target.exists():
                raise FileExistsError(
                    "Destination appeared during copying; will not overwrite"
                )
            staging.rename(target)
            completed.append(str(target))
        except BaseException:
            # Partial files are deliberately retained, clearly labelled, and
            # never appear as a successfully collected dataset.
            progress(f"Copy interrupted; incomplete files retained in {staging}")
            raise
    return completed
