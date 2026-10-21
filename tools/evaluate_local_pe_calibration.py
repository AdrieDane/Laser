"""Replay the rejected local-calibration pilot, never the application workflow.

Saved pickle inputs must come from a trusted local annotation run. Manual
crosswalks are opened only after annotation decisions, solely for evaluation.
Use a fresh output directory; no input, earlier output or TXT is overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "app"))

from matlab_annotation.experimental_pe import (  # noqa: E402 -- local CLI path setup
    ExperimentalPESelectionSettings,
    apply_experimental_pe_selector,
)
from matlab_annotation.pe_standards import complete_pe_internal_standards  # noqa: E402
from standard_lipidomics.validation import (  # noqa: E402
    canonical_feature_id,
    compare_manual_annotations,
    summarize_comparison,
)
from tools.local_pe_calibration import refine_local_pe  # noqa: E402

SOURCES = (
    "tools/local_pe_calibration.py",
    "tools/evaluate_local_pe_calibration.py",
    "app/matlab_annotation/pe_rebuild.py",
    "app/matlab_annotation/experimental_pe.py",
    "app/annotator/peak_evidence.py",
    "reference_data/standard_pe_rebuild.json",
)


def source_hashes():
    return {s: hashlib.sha256((REPO / s).read_bytes()).hexdigest() for s in SOURCES}


def selected_keys(frame):
    return set(
        zip(
            frame.stream,
            frame["name"],
            frame.adduct,
            frame.feat_feature_id.map(canonical_feature_id),
        )
    )


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-runs-dir", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--manual-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-output-dir", type=Path)
    p.add_argument("--expected-datasets", type=int, default=15)
    p.add_argument("--trust-local-pickles", action="store_true")
    args = p.parse_args()
    if not args.trust_local_pickles:
        p.error("Saved pickles can execute code; explicitly trust only your local runs")
    return args


def main():
    args = arguments()
    paths = sorted(args.full_runs_dir.glob("*/full_annotation.pkl"))
    if len(paths) != args.expected_datasets:
        raise ValueError(
            f"Expected {args.expected_datasets} datasets; found {len(paths)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frozen = source_hashes()
    protocol = dict(
        source_hashes=frozen,
        scope="RP-positive endogenous PE only; all other choices fixed",
        baseline="3323003 restored method; earlier optional PE selector enabled",
        calibration="robust mass surface, class-held-out check; target-composition-excluded local RT with held-out check",
        admission="unchanged PE scorer, threshold0.10/margin0.03; uncertainty INSIDE3ppm/10seconds; supported material calibration change required",
        manual_labels_used_to_fit=False,
        acceptance="no correct-peak losses or added different-peak calls; at least4 correct gains per extra; seek substantial gain, not another3-call rule",
        validation="saved full-core replay, development examples, not external validation",
    )
    (args.output_dir / "frozen_method.json").write_text(json.dumps(protocol, indent=2))
    records, audits, timings = [], [], []
    for path in paths:
        name = path.parent.name
        dest = args.output_dir / name
        dest.mkdir()
        with path.open("rb") as handle:
            core = complete_pe_internal_standards(pickle.load(handle))
        core = apply_experimental_pe_selector(
            core, settings=ExperimentalPESelectionSettings(enabled=True)
        )
        before = pd.read_csv(
            args.baseline_dir / name / "final_automatic_annotations.csv",
            dtype={"feat_feature_id": str},
            low_memory=False,
        )
        if selected_keys(core["automatic_annotations"]) != selected_keys(before):
            raise ValueError(f"Baseline selected features differ: {name}")
        p, n = core["streams"]["RP_Pos"], core["streams"]["RP_Neg"]
        start = perf_counter()
        annotations, _, decisions, audit = refine_local_pe(
            p["annotations"],
            p["all_pairs"],
            p["reference"],
            p["features"],
            n["annotations"],
            n["features"],
        )
        # The stream-level table need not have root-level metadata. Never infer
        # the root stream from standard_stream: some old rows lack that alias.
        selected = annotations.copy()
        selected["stream"], selected["standard_stream"] = "RP_Pos", "RP_Pos"
        after = pd.concat(
            [
                core["automatic_annotations"][
                    core["automatic_annotations"].stream.ne("RP_Pos")
                ],
                selected,
            ],
            ignore_index=True,
            sort=False,
        )
        oldkeys, newkeys = selected_keys(before), selected_keys(after)
        if not oldkeys <= newkeys:
            raise ValueError(f"An existing selected feature was removed: {name}")
        new = newkeys - oldkeys
        added = after[
            [
                k in new
                for k in zip(
                    after.stream,
                    after["name"],
                    after.adduct,
                    after.feat_feature_id.map(canonical_feature_id),
                )
            ]
        ].copy()
        if not (added["class"].eq("PE") & added.stream.eq("RP_Pos")).all():
            raise ValueError(f"Unexpected class/stream changed: {name}")
        if selected.feat_feature_id.map(canonical_feature_id).duplicated().any():
            raise ValueError(f"Selected feature collision: {name}")
        if args.expected_output_dir:
            expected = pd.read_csv(
                args.expected_output_dir / name / "final_automatic_annotations.csv",
                dtype={"feat_feature_id": str},
                low_memory=False,
            )
            if newkeys != selected_keys(expected):
                raise ValueError(
                    f"Research-tool choices differ from frozen pilot: {name}"
                )
        after.to_csv(dest / "final_automatic_annotations.csv", index=False)
        added.to_csv(dest / "additions.csv", index=False)
        decisions.to_csv(dest / "candidate_decisions.csv", index=False)
        audit.to_csv(dest / "calibration_audit.csv", index=False)
        audits.append(audit.assign(dataset=name))
        timings.append(dict(dataset=name, seconds=perf_counter() - start))

        # Evaluation-only truth: selection has finished and no fitted model or
        # threshold is updated after this file is opened.
        crosswalk = pd.read_csv(
            args.manual_dir / name / "curated_feature_crosswalk.csv"
        )
        a = compare_manual_annotations(crosswalk, after, dataset=name)
        b = compare_manual_annotations(crosswalk, before, dataset=name)
        a.manual.to_csv(dest / "manual_comparison.csv", index=False)
        a.laser.to_csv(dest / "laser_comparison.csv", index=False)
        grouping = ("dataset", "stream", "lipid_class")
        new_summary = summarize_comparison(a, group_by=grouping)
        old_summary = summarize_comparison(b, group_by=grouping)
        merged = new_summary.merge(
            old_summary,
            on=list(grouping),
            suffixes=("", "_before"),
            validate="one_to_one",
        ).assign(duplicate=name.endswith("_duplicate"))
        records.append(merged)
        pe = merged[merged.stream.eq("RP_Pos") & merged.lipid_class.eq("PE")].iloc[0]
        print(
            name,
            "added",
            len(added),
            "correct gain",
            int(pe.manual_same_peak - pe.manual_same_peak_before),
            "extra gain",
            int(pe.laser_not_in_manual_list - pe.laser_not_in_manual_list_before),
            "different-peak gain",
            int(
                pe.laser_same_manual_lipid_different_peak
                - pe.laser_same_manual_lipid_different_peak_before
            ),
            flush=True,
        )
    if frozen != source_hashes():
        raise ValueError("Experiment sources changed during evaluation")
    summary = pd.concat(records, ignore_index=True)
    summary.to_csv(args.output_dir / "by_dataset_stream_class.csv", index=False)
    pd.concat(audits, ignore_index=True).to_csv(
        args.output_dir / "calibration_audit.csv", index=False
    )
    pd.DataFrame(timings).to_csv(args.output_dir / "timings.csv", index=False)
    (args.output_dir / "verification.json").write_text(
        json.dumps(
            dict(
                datasets_tested=len(paths),
                unique_datasets=sum(
                    not p.parent.name.endswith("_duplicate") for p in paths
                ),
                every_baseline_choice_preserved=True,
                other_streams_and_classes_unchanged=True,
                all_expected_pilot_choices_match=bool(args.expected_output_dir),
                source_hashes=frozen,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
