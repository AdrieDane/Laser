"""MZmine MS/MS import and editable class-specific rule evidence.

The rule catalogue lives in ``app/rules/lipid_msms_rules.json``. Candidates
with an applicable rule and a spectrum must satisfy the configured product-ion
or neutral-loss target. Candidates without a linked spectrum remain
provisional, because absence of a spectrum is not evidence of absence.
"""

from __future__ import annotations

import io
import json
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, BinaryIO, Iterable, cast

import numpy as np
import pandas as pd

from .fatty_acyl import detect_fatty_acyl_evidence
from persistent_input_cache import atomic_write_text
from resource_paths import RULES_DIRECTORY


RULES_PATH = RULES_DIRECTORY / "lipid_msms_rules.json"
with RULES_PATH.open("r", encoding="utf-8") as _rules_handle:
    RULE_CONFIG = json.load(_rules_handle)
MSMS_RULES = tuple(RULE_CONFIG.get("rules", ()))
FATTY_ACYL_RULES = tuple(RULE_CONFIG.get("fatty_acyl_rules", ()))
DEFAULT_REQUIRE_CLASS_FRAGMENT = bool(RULE_CONFIG.get("strict_filter_default", True))
DEFAULT_FRAGMENT_TOLERANCE_DA = float(RULE_CONFIG.get("default_tolerance_da", 0.1))
DEFAULT_FA_MIN_RELATIVE_INTENSITY_PERCENT = float(
    RULE_CONFIG.get("default_fatty_acyl_min_relative_intensity_percent", 1.0)
)

# Compatibility constants used by existing integrations and tests.
PC_HEADGROUP_MZ = 184.0733


def _iter_text_lines(source: str | Path | bytes | BinaryIO) -> Iterable[str]:
    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8-sig", errors="replace") as handle:
            yield from handle
        return
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    for line in source:
        yield line.decode("utf-8-sig", errors="replace") if isinstance(line, bytes) else line


def _normalise_spectrum(spectrum: dict) -> dict:
    """Return the stable in-memory representation used by every importer."""
    peaks = []
    for peak in spectrum.get("peaks") or []:
        if not isinstance(peak, (list, tuple)) or len(peak) < 2:
            continue
        try:
            mz, intensity = float(peak[0]), float(peak[1])
        except (TypeError, ValueError):
            continue
        if np.isfinite(mz) and np.isfinite(intensity) and intensity >= 0:
            peaks.append([mz, intensity])
    result = dict(spectrum)
    result["feature_id"] = str(spectrum["feature_id"])
    result["peaks"] = peaks
    return result


def load_mzmine_jsonl(source: str | Path | bytes | BinaryIO) -> tuple[dict[str, dict], int]:
    """Load MZmine's line-delimited JSON scan export by exact feature ID."""
    spectra: dict[str, dict] = {}
    malformed = 0
    for line in _iter_text_lines(source):
        if not line.strip():
            continue
        try:
            spectrum = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            malformed += 1
            continue
        if int(spectrum.get("ms_level", 0) or 0) != 2:
            continue
        feature_id = spectrum.get("feature_id")
        if feature_id is None:
            malformed += 1
            continue
        spectrum = _normalise_spectrum(spectrum)
        peaks = spectrum["peaks"]
        key = str(feature_id)
        previous = spectra.get(key)
        if previous is None or len(peaks) > len(previous.get("peaks") or []):
            spectra[key] = spectrum
    return spectra, malformed


@dataclass(frozen=True)
class _SpectrumIndexEntry:
    offset: int
    length: int
    peak_count: int
    metadata: dict


class IndexedMzmineSpectra(Mapping[str, dict]):
    """Disk-backed MZmine JSONL mapping with a bounded decoded-spectrum cache.

    Indexing reads the source once, but peak arrays are decoded only when the
    annotation engine or reviewer asks for a particular feature. This avoids
    retaining tens of thousands of spectra merely to review one lipid species.
    """

    def __init__(
        self,
        path: str | Path,
        entries: Mapping[str, _SpectrumIndexEntry],
        *,
        malformed: int = 0,
        cache_size: int = 256,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self._entries = dict(entries)
        self.malformed = int(malformed)
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._lock = RLock()

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __getitem__(self, feature_id: str) -> dict:
        key = str(feature_id)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
        entry = self._entries[key]
        with self.path.open("rb") as handle:
            handle.seek(entry.offset)
            raw = handle.read(entry.length)
        spectrum = _normalise_spectrum(json.loads(raw.decode("utf-8-sig")))
        with self._lock:
            self._cache[key] = spectrum
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return spectrum

    def audit_items(self) -> Iterator[tuple[str, dict]]:
        """Yield peak-free metadata without decoding spectrum arrays."""
        for feature_id, entry in self._entries.items():
            yield feature_id, entry.metadata

    @property
    def cached_spectrum_count(self) -> int:
        return len(self._cache)


class FilteredSpectra(Mapping[str, dict]):
    """Zero-copy ID-filtered view over an existing spectrum mapping."""

    def __init__(self, source: Mapping[str, dict], feature_ids: Iterable[str]) -> None:
        self.source = source
        allowed = {str(feature_id) for feature_id in feature_ids}
        self._keys = tuple(key for key in source if str(key) in allowed)
        self._key_set = set(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __getitem__(self, feature_id: str) -> dict:
        key = str(feature_id)
        if key not in self._key_set:
            raise KeyError(key)
        return self.source[key]


def _load_index_cache(
    source: Path,
    cache_path: Path,
    *,
    cache_size: int,
) -> tuple[IndexedMzmineSpectra, int]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported spectrum-index cache format.")
    source_size = source.stat().st_size
    if int(payload.get("source_size", -1)) != source_size:
        raise ValueError("Spectrum-index cache does not match its source.")
    entries: dict[str, _SpectrumIndexEntry] = {}
    for raw in payload.get("entries", ()):
        feature_id, offset, length, peak_count, precursor_mz, rt, polarity = raw
        key = str(feature_id)
        offset = int(offset)
        length = int(length)
        peak_count = int(peak_count)
        if offset < 0 or length <= 0 or offset + length > source_size:
            raise ValueError("Spectrum-index cache contains an invalid byte range.")
        if peak_count < 0:
            raise ValueError("Spectrum-index cache contains an invalid peak count.")
        entries[key] = _SpectrumIndexEntry(
            offset=offset,
            length=length,
            peak_count=peak_count,
            metadata={
                "feature_id": key,
                "precursor_mz": precursor_mz,
                "rt": rt,
                "polarity": polarity,
            },
        )
    spectra = IndexedMzmineSpectra(
        source,
        entries,
        malformed=int(payload.get("malformed", 0)),
        cache_size=cache_size,
    )
    return spectra, spectra.malformed


def _write_index_cache(
    source: Path,
    cache_path: Path,
    entries: Mapping[str, _SpectrumIndexEntry],
    malformed: int,
) -> None:
    rows = []
    for feature_id, entry in entries.items():
        metadata = entry.metadata
        rows.append(
            [
                feature_id,
                entry.offset,
                entry.length,
                entry.peak_count,
                metadata.get("precursor_mz"),
                metadata.get("rt"),
                metadata.get("polarity"),
            ]
        )
    payload = {
        "format_version": 1,
        "source_size": source.stat().st_size,
        "malformed": int(malformed),
        "entries": rows,
    }
    atomic_write_text(
        cache_path,
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
    )


def index_mzmine_jsonl(
    path: str | Path,
    *,
    cache_size: int = 256,
    index_cache_path: str | Path | None = None,
) -> tuple[IndexedMzmineSpectra, int]:
    """Index a local scan export and lazily decode its peak arrays."""
    source = Path(path).expanduser().resolve()
    cache_path = (
        Path(index_cache_path).expanduser().resolve()
        if index_cache_path is not None
        else None
    )
    if cache_path is not None and cache_path.is_file():
        try:
            return _load_index_cache(source, cache_path, cache_size=cache_size)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            try:
                cache_path.unlink(missing_ok=True)
            except OSError:
                pass
    entries: dict[str, _SpectrumIndexEntry] = {}
    malformed = 0
    with source.open("rb") as handle:
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            if not raw.strip():
                continue
            try:
                parsed = json.loads(raw.decode("utf-8-sig", errors="replace"))
            except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                malformed += 1
                continue
            if int(parsed.get("ms_level", 0) or 0) != 2:
                continue
            feature_id = parsed.get("feature_id")
            if feature_id is None:
                malformed += 1
                continue
            normalized = _normalise_spectrum(parsed)
            key = str(feature_id)
            metadata = {name: value for name, value in normalized.items() if name != "peaks"}
            entry = _SpectrumIndexEntry(
                offset=offset,
                length=len(raw),
                peak_count=len(normalized["peaks"]),
                metadata=metadata,
            )
            previous = entries.get(key)
            if previous is None or entry.peak_count > previous.peak_count:
                entries[key] = entry
    spectra = IndexedMzmineSpectra(
        source, entries, malformed=malformed, cache_size=cache_size,
    )
    if cache_path is not None:
        try:
            _write_index_cache(source, cache_path, entries, malformed)
        except (OSError, TypeError, ValueError):
            pass
    return spectra, malformed


@dataclass(frozen=True)
class _SpectrumLinkComparison:
    matched: pd.DataFrame
    ppm_error: pd.Series
    rt_error: pd.Series
    bad_mz: pd.Series
    bad_rt: pd.Series
    bad_polarity: pd.Series

    @property
    def incompatible(self) -> pd.Series:
        return self.bad_mz | self.bad_rt | self.bad_polarity


def _linked_spectrum_metadata(
    indexed: pd.DataFrame,
    spectra: Mapping[str, dict],
) -> pd.DataFrame:
    audit_items = getattr(spectra, "audit_items", spectra.items)
    records: list[tuple[str, Any, Any, str]] = []
    for feature_id, spectrum in audit_items():
        key = str(feature_id)
        if key in indexed.index:
            records.append(
                (
                    key,
                    spectrum.get("precursor_mz"),
                    spectrum.get("rt"),
                    str(spectrum.get("polarity", "") or "").strip(),
                )
            )
    return pd.DataFrame.from_records(
        records,
        columns=["feature_id", "precursor_mz", "spectrum_rt", "spectrum_polarity"],
    ).set_index("feature_id")


def _compare_spectrum_links(
    observed: pd.DataFrame,
    indexed: pd.DataFrame,
    *,
    expected_sign: str,
    precursor_ppm_tolerance: float,
    rt_tolerance_min: float,
) -> _SpectrumLinkComparison:
    matched = observed.join(indexed, how="left")
    feature_mz = pd.to_numeric(matched["mz"], errors="coerce")
    precursor_mz = pd.to_numeric(matched["precursor_mz"], errors="coerce")
    feature_rt = pd.to_numeric(matched["rt"], errors="coerce")
    spectrum_rt = pd.to_numeric(matched["spectrum_rt"], errors="coerce")
    ppm_error = (precursor_mz - feature_mz) / feature_mz * 1e6
    rt_error = spectrum_rt - feature_rt
    bad_mz = (
        feature_mz.notna()
        & feature_mz.gt(0)
        & precursor_mz.notna()
        & ppm_error.abs().gt(float(precursor_ppm_tolerance))
    )
    bad_rt = (
        feature_rt.notna()
        & spectrum_rt.notna()
        & rt_error.abs().gt(float(rt_tolerance_min))
    )
    bad_polarity = (
        matched["spectrum_polarity"].ne("")
        & matched["spectrum_polarity"].ne(expected_sign)
        if expected_sign
        else pd.Series(False, index=matched.index)
    )
    return _SpectrumLinkComparison(
        matched, ppm_error, rt_error, bad_mz, bad_rt, bad_polarity,
    )


def _spectrum_link_examples(comparison: _SpectrumLinkComparison) -> list[dict]:
    examples = []
    for position in np.flatnonzero(comparison.incompatible.to_numpy())[:5]:
        reasons = []
        if bool(comparison.bad_mz.iloc[position]):
            reasons.append("precursor_mz")
        if bool(comparison.bad_rt.iloc[position]):
            reasons.append("retention_time")
        if bool(comparison.bad_polarity.iloc[position]):
            reasons.append("polarity")
        ppm_value = comparison.ppm_error.iloc[position]
        rt_value = comparison.rt_error.iloc[position]
        examples.append(
            {
                "feature_id": str(comparison.matched.index[position]),
                "reasons": reasons,
                "ppm_error": None if not np.isfinite(ppm_value) else float(ppm_value),
                "rt_error_min": None
                if not np.isfinite(rt_value)
                else float(rt_value),
                "spectrum_polarity": str(
                    comparison.matched["spectrum_polarity"].iloc[position]
                ),
            }
        )
    return examples


def audit_mzmine_spectrum_links(
    features: pd.DataFrame,
    spectra: Mapping[str, dict],
    *,
    polarity: str | None = None,
    precursor_ppm_tolerance: float = 5.0,
    rt_tolerance_min: float = 0.01,
) -> dict:
    """Verify exact-ID spectra using precursor m/z, RT, and polarity.

    MZmine row IDs restart in every project. Exact ID overlap alone therefore
    cannot distinguish the correct companion scan export from a stale or
    opposite-polarity file.
    """
    required = {"feature_id", "mz", "rt"}
    if not required.issubset(features.columns):
        missing = ", ".join(sorted(required - set(features.columns)))
        raise ValueError(f"Spectrum-link audit requires feature columns: {missing}")
    expected_polarity = str(polarity or "").strip().casefold()
    if expected_polarity not in {"", "pos", "neg", "both"}:
        raise ValueError("Spectrum-link audit polarity must be pos, neg, both, or empty.")
    expected_sign = "+" if expected_polarity == "pos" else "-" if expected_polarity == "neg" else ""
    indexed = features.loc[:, ["feature_id", "mz", "rt"]].copy()
    indexed["feature_id"] = indexed["feature_id"].astype(str)
    indexed = indexed.drop_duplicates("feature_id", keep="first").set_index("feature_id")
    observed = _linked_spectrum_metadata(indexed, spectra)
    linked = len(observed)
    if not observed.empty:
        comparison = _compare_spectrum_links(
            observed,
            indexed,
            expected_sign=expected_sign,
            precursor_ppm_tolerance=precursor_ppm_tolerance,
            rt_tolerance_min=rt_tolerance_min,
        )
        compatible_ids = comparison.matched.index[
            ~comparison.incompatible
        ].astype(str).tolist()
        mismatches = {
            "precursor_mz": int(comparison.bad_mz.sum()),
            "retention_time": int(comparison.bad_rt.sum()),
            "polarity": int(comparison.bad_polarity.sum()),
        }
        examples = _spectrum_link_examples(comparison)
    else:
        compatible_ids = []
        mismatches = {"precursor_mz": 0, "retention_time": 0, "polarity": 0}
        examples = []
    compatible = len(compatible_ids)
    return {
        "spectra_total": int(len(spectra)),
        "exact_id_overlap": int(linked),
        "compatible_links": int(compatible),
        "incompatible_links": int(linked - compatible),
        "compatible_fraction": float(compatible / linked) if linked else 0.0,
        "precursor_ppm_tolerance": float(precursor_ppm_tolerance),
        "rt_tolerance_min": float(rt_tolerance_min),
        "expected_polarity": expected_polarity or "not_checked",
        "mismatch_counts": mismatches,
        "mismatch_examples": examples,
        "compatible_feature_ids": compatible_ids,
    }


def validated_mzmine_spectra(
    features: pd.DataFrame,
    spectra: Mapping[str, dict],
    *,
    polarity: str | None = None,
    minimum_compatible_fraction: float = 0.90,
) -> tuple[Mapping[str, dict], dict]:
    """Return independently verified links or reject a wrong companion file."""
    audit = audit_mzmine_spectrum_links(features, spectra, polarity=polarity)
    if spectra and (
        audit["exact_id_overlap"] == 0
        or audit["compatible_fraction"] < float(minimum_compatible_fraction)
    ):
        raise ValueError(
            "MS/MS scan export does not belong to this feature table or polarity: "
            f"{audit['compatible_links']:,}/{audit['exact_id_overlap']:,} exact-ID "
            "links passed precursor m/z, RT and polarity checks. Select the scan "
            "export produced by the same MZmine run."
        )
    compatible = set(audit.pop("compatible_feature_ids"))
    if isinstance(spectra, IndexedMzmineSpectra):
        return FilteredSpectra(spectra, compatible), audit
    return {
        str(feature_id): spectrum
        for feature_id, spectrum in spectra.items()
        if str(feature_id) in compatible
    }, audit


def find_associated_scan_export(
    feature_name: str, available_names: Iterable[str]
) -> str | None:
    """Find the MZmine JSON scan export associated with a feature-list CSV."""
    available = [str(name) for name in available_names]
    feature_path = Path(feature_name)
    if feature_path.suffix.casefold() == ".gz":
        feature_path = feature_path.with_suffix("")
    stem = feature_path.stem
    prefix = stem[:-len("_EXPORT")] if stem.upper().endswith("_EXPORT") else stem
    expected_name = f"{prefix}_scan_export.json"
    same_directory = str(feature_path.with_name(expected_name))
    if same_directory in available:
        return same_directory
    exact = [name for name in available if Path(name).name.lower() == expected_name.lower()]
    if exact:
        return sorted(exact, key=lambda name: (len(Path(name).parts), name))[0]
    prefix_lower = prefix.lower()
    fallback = [
        name for name in available
        if Path(name).stem.lower().startswith(prefix_lower)
        and "scan_export" in Path(name).stem.lower()
    ]
    return sorted(fallback)[0] if fallback else None


def _candidate_classes(row: pd.Series) -> set[str]:
    lipid_class = str(row.get("class", ""))
    base = row.get("base", "")
    classes = {lipid_class}
    if isinstance(base, str) and base:
        classes.add(f"{lipid_class}-{base}")
    return classes


def _row_polarity(row: pd.Series, polarity: str) -> str:
    if polarity != "both":
        return polarity
    adduct = str(row.get("adduct", ""))
    return "neg" if adduct.endswith("-") else "pos"


def applicable_msms_rule(row: pd.Series, polarity: str) -> dict | None:
    """Return the first configured rule matching class, polarity, and adduct."""
    candidate_classes = _candidate_classes(row)
    candidate_polarity = _row_polarity(row, polarity)
    for rule in MSMS_RULES:
        if candidate_polarity != rule.get("polarity"):
            continue
        if not candidate_classes.intersection(map(str, rule.get("lipid_classes", ()))):
            continue
        adducts = rule.get("adducts")
        if adducts and str(row.get("adduct", "")) not in set(map(str, adducts)):
            continue
        return rule
    return None


def evaluate_msms_rule(
    rule: dict,
    spectrum: dict | None,
    precursor_mz: float | None = None,
    tolerance_da: float = DEFAULT_FRAGMENT_TOLERANCE_DA,
) -> dict:
    """Evaluate one product-ion or neutral-loss rule against one spectrum."""
    evidence = rule["evidence"]
    evidence_type = str(evidence["type"])
    target = float(evidence["target_mz"])
    base = {
        "msms_rule_id": str(rule["id"]),
        "msms_rule_description": str(rule.get("description", "")),
        "msms_rule_evidence_type": evidence_type,
        "msms_rule_evidence_name": str(evidence.get("name", "")),
        "msms_rule_target_mz": target,
        "msms_rule_observed_mz": np.nan,
        "msms_rule_observed_value": np.nan,
        "msms_rule_error_da": np.nan,
        "msms_rule_error_ppm": np.nan,
        "msms_rule_relative_intensity": np.nan,
        "msms_rule_pass": pd.NA,
    }
    if spectrum is None:
        return {**base, "msms_status": "no_msms", "msms_has_spectrum": False}

    peaks = spectrum.get("peaks") or []
    precursor = spectrum.get("precursor_mz") or precursor_mz
    try:
        precursor = float(cast(Any, precursor))
    except (TypeError, ValueError):
        precursor = np.nan

    matches = []
    for fragment_mz, intensity in peaks:
        observed = float(fragment_mz)
        if evidence_type == "neutral_loss":
            if not np.isfinite(precursor):
                continue
            observed = precursor - float(fragment_mz)
        error = observed - target
        if abs(error) <= tolerance_da:
            matches.append((float(fragment_mz), float(intensity), observed, error))

    if not matches:
        return {
            **base, "msms_status": "not_detected", "msms_has_spectrum": True,
            "msms_rule_pass": False,
        }

    fragment_mz, intensity, observed, error = max(matches, key=lambda item: item[1])
    base_intensity = max((float(peak[1]) for peak in peaks), default=0.0)
    relative = 100.0 * intensity / base_intensity if base_intensity > 0 else np.nan
    return {
        **base,
        "msms_status": "supported",
        "msms_has_spectrum": True,
        "msms_rule_observed_mz": fragment_mz,
        "msms_rule_observed_value": observed,
        "msms_rule_error_da": error,
        "msms_rule_error_ppm": error / target * 1e6,
        "msms_rule_relative_intensity": relative,
        "msms_rule_pass": True,
    }


def pc_positive_headgroup_evidence(
    spectrum: dict | None,
    tolerance_da: float = DEFAULT_FRAGMENT_TOLERANCE_DA,
) -> dict:
    """Backward-compatible wrapper for phosphocholine evidence."""
    rule = next(rule for rule in MSMS_RULES if "PC" in rule["lipid_classes"])
    result = evaluate_msms_rule(rule, spectrum, tolerance_da=tolerance_da)
    result.update({
        "msms_headgroup_expected_mz": result["msms_rule_target_mz"],
        "msms_headgroup_mz": result["msms_rule_observed_mz"],
        "msms_headgroup_error_da": result["msms_rule_error_da"],
        "msms_headgroup_error_ppm": result["msms_rule_error_ppm"],
        "msms_headgroup_relative_intensity": result["msms_rule_relative_intensity"],
    })
    return result


def apply_class_specific_rules(
    annotations: pd.DataFrame,
    spectra: Mapping[str, dict],
    polarity: str,
    tolerance_da: float = DEFAULT_FRAGMENT_TOLERANCE_DA,
    require_class_fragment: bool = True,
    fatty_acyl_min_relative_intensity_percent: float = (
        DEFAULT_FA_MIN_RELATIVE_INTENSITY_PERCENT
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach configured MS/MS evidence and optionally remove rule failures."""
    out = annotations.copy()
    defaults = {
        "msms_rule_id": "",
        "msms_rule_description": "",
        "msms_rule_evidence_type": "",
        "msms_rule_evidence_name": "",
        "msms_rule_target_mz": np.nan,
        "msms_rule_observed_mz": np.nan,
        "msms_rule_observed_value": np.nan,
        "msms_rule_error_da": np.nan,
        "msms_rule_error_ppm": np.nan,
        "msms_rule_relative_intensity": np.nan,
        "msms_rule_pass": pd.NA,
        "msms_status": "not_evaluated",
        "msms_has_spectrum": False,
        # Legacy phosphocholine columns remain available to old exports.
        "msms_headgroup_expected_mz": np.nan,
        "msms_headgroup_mz": np.nan,
        "msms_headgroup_error_da": np.nan,
        "msms_headgroup_error_ppm": np.nan,
        "msms_headgroup_relative_intensity": np.nan,
        "fatty_acyl_rule_id": "",
        "fatty_acyl_status": "not_applicable",
        "fatty_acyl_min_relative_intensity_percent": (
            fatty_acyl_min_relative_intensity_percent
        ),
        "fatty_acyl_detected": "",
        "fatty_acyl_detected_count": 0,
        "fatty_acyl_evidence_json": "[]",
        "fatty_acyl_species_mapping_json": "{}",
        "fatty_acyl_fragment_intensities": "",
        "molecular_species_candidates": "",
        "molecular_species_candidate_count": 0,
        "molecular_species_ambiguous": False,
        "molecular_species_ranking_json": "[]",
        "most_likely_molecular_species": "",
        "most_likely_molecular_species_score": np.nan,
        "most_likely_molecular_species_ambiguous": False,
    }
    for column, value in defaults.items():
        out[column] = value
    if out.empty:
        return out, out.copy()

    feature_ids = out["feat_feature_id"].astype(str)
    out["msms_has_spectrum"] = feature_ids.isin(spectra)
    column_values = {column: out[column].tolist() for column in defaults}
    applies_values = np.zeros(len(out), dtype=bool)
    for position, record in enumerate(out.to_dict("records")):
        row = cast(Any, record)
        spectrum = spectra.get(str(row["feat_feature_id"]))
        candidate_polarity = _row_polarity(row, polarity)
        fatty_acyl = detect_fatty_acyl_evidence(
            row, spectrum, candidate_polarity, FATTY_ACYL_RULES, tolerance_da,
            min_relative_intensity_percent=(
                fatty_acyl_min_relative_intensity_percent
            ),
        )
        for column, value in fatty_acyl.items():
            column_values[column][position] = value
        rule = applicable_msms_rule(row, polarity)
        if rule is None:
            continue
        applies_values[position] = True
        if rule["evidence"]["type"] == "fatty_acyl":
            fatty_status = str(fatty_acyl["fatty_acyl_status"])
            status = (
                fatty_status if fatty_status in {"supported", "no_msms"}
                else "not_detected"
            )
            evidence = {
                "msms_rule_id": str(rule["id"]),
                "msms_rule_description": str(rule.get("description", "")),
                "msms_rule_evidence_type": "fatty_acyl",
                "msms_rule_evidence_name": str(rule["evidence"].get("name", "")),
                "msms_rule_target_mz": np.nan,
                "msms_rule_observed_mz": np.nan,
                "msms_rule_observed_value": np.nan,
                "msms_rule_error_da": np.nan,
                "msms_rule_error_ppm": np.nan,
                "msms_rule_relative_intensity": np.nan,
                "msms_rule_pass": (
                    True if status == "supported" else
                    False if status == "not_detected" else pd.NA
                ),
                "msms_status": status,
                "msms_has_spectrum": spectrum is not None,
            }
        else:
            evidence = evaluate_msms_rule(
                rule, spectrum, precursor_mz=row.get("feat_mz"),
                tolerance_da=tolerance_da,
            )
        for column, value in evidence.items():
            column_values[column][position] = value
        if rule["id"] == "POS_PHOSPHOCHOLINE_HEADGROUP_184":
            legacy_values = {
                "msms_headgroup_expected_mz": evidence["msms_rule_target_mz"],
                "msms_headgroup_mz": evidence["msms_rule_observed_mz"],
                "msms_headgroup_error_da": evidence["msms_rule_error_da"],
                "msms_headgroup_error_ppm": evidence["msms_rule_error_ppm"],
                "msms_headgroup_relative_intensity": evidence[
                    "msms_rule_relative_intensity"
                ],
            }
            for column, value in legacy_values.items():
                column_values[column][position] = value

    for column, values in column_values.items():
        out[column] = values
    applies = pd.Series(applies_values, index=out.index)
    failure_mask = applies & out["msms_status"].eq("not_detected")
    failures = out[failure_mask].copy()
    excluded_mask = failure_mask if require_class_fragment else pd.Series(False, index=out.index)
    return out[~excluded_mask].copy(), failures


# Keep the original public name so older scripts remain functional.
apply_pc_positive_rule = apply_class_specific_rules


def msms_preference(annotations: pd.DataFrame) -> np.ndarray:
    """Rank candidates by MS/MS evidence: confirmed, unknown, contradicted.

    Used to choose which feature represents a lipid species when several
    candidates all cleared the acceptance windows. Retention-time proximity
    alone is the wrong tie-break there: a confirming class fragment is far
    stronger evidence than being a tenth of a second nearer the grid, and
    without this the viewer showed an unconfirmed feature for 77 species on the
    positive example while a confirmed one for the same species sat unused.

    The effect is largest for an engine whose grid is fitted to the data,
    because its candidates cluster tightly enough that the tie-break decides
    far more often.
    """
    status = annotations.get(
        "msms_status", pd.Series("", index=annotations.index)
    ).astype(str)
    return np.select(
        [status.eq("supported"), status.eq("no_msms")], [0, 1], default=2
    )


def msms_rule_result(row: pd.Series) -> str:
    """Return one stable, machine-readable configured-rule outcome."""
    rule_id = str(row.get("msms_rule_id", "") or "").strip()
    if not rule_id:
        return "NO_RULE"
    status = str(
        row.get("msms_status", "not_evaluated") or "not_evaluated"
    ).strip().casefold()
    if status == "supported":
        return "PASS"
    if status == "not_detected":
        return "FAIL"
    if status == "no_msms":
        return "NO_SPECTRUM"
    return "NOT_EVALUATED"
