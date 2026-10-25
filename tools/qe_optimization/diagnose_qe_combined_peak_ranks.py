"""Read-only132 matching INNER cases; no new fits/annotations/cutoff selection.

Freeze ALL potential identity-local moves BEFORE training-truth comparison.
This is NOT a global assignment: multiple moves may compete for one free peak.
Each dataset repeats four outer-training contexts; counts are NOT benchmark
gains or independent new observations. Never use matching outer-case truth.
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from frozen_resources import configure_frozen_resources
import run_qe_raw_mass_profile_experiment as runner

OUT = runner.QE / "combined_raw_model_peak_rank_audit_v1"
GAPS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
SCREENS = ("identity_only", "clear_physical_free_target", "clear_physical_free_profile")


def physical_masks(pool, stream, protection):
    from matlab_annotation.automatic import matlab_annotation_adducts
    from matlab_annotation.coherent_retention import _protected_mass
    from matlab_annotation.standard_rt_library import ion_key

    protected = pool.feat_feature_id.astype(str).isin(
        set(map(str, protection["protected_features"]))
    )
    masses = np.asarray(protection["declared_masses"], float)
    protected |= _protected_mass(pool.feat_mz.to_numpy(float), masses)
    protected |= _protected_mass(pool.ref_mz.to_numpy(float), masses)
    if not pool.is_is.isin([False, True]).all():
        raise ValueError("Explicit physical standard flags required")
    allowed = {ion_key(a) for a in matlab_annotation_adducts(stream[:2])}
    keep = ~protected & ~pool.is_is.astype(bool)
    keep &= ~pool.name.astype(str).str.startswith("IS: ") & ~pool.name.astype(
        str
    ).str.endswith(" (IS)")
    keep &= pool.adduct.map(ion_key).isin(allowed)
    keep &= pool.adduct.map(ion_key).str.endswith(
        "+" if stream.endswith("Pos") else "-"
    )
    keep &= (
        pool.get("isotopologue", pd.Series("", index=pool.index))
        .fillna("")
        .astype(str)
        .eq("")
    )
    values = pool[["ref_mz", "feat_mz", "feat_rt", "feat_intensity"]].to_numpy(float)
    keep &= np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    keep &= ((pool.feat_mz / pool.ref_mz - 1) * 1e6).abs().lt(3)
    deviations = [
        (pool.feat_rt * 60 - pd.to_numeric(pool[c], errors="coerce") * scale).abs()
        for c, scale in (("ref_rt", 60), ("rt_expected", 60), ("atlas_rt_seconds", 1))
        if c in pool
    ]
    if not deviations:
        raise ValueError("Matching authorized RT references required")
    keep &= pd.concat(deviations, axis=1).min(axis=1).le(10 + 1e-9)
    return keep, protected


def blind_moves(
    pool, scores, evidence, current, occupied, physical, protected, clear_indices
):
    """Identity winner fixed FIRST; flags report eligibility, never fallback.

    No curation argument/columns allowed. Gold labels are attached only after
    every one of132 complete blind files has been frozen by main().
    """
    from matlab_annotation.raw_counterpart_rt import SUPERVISION
    from standard_lipidomics.validation import _normalise, _lipid_class

    if SUPERVISION & set(pool) or SUPERVISION & set(current):
        raise ValueError("Blind peak ranking refuses supervised flags")
    if not pool.index.equals(scores.index) or not pool.index.equals(evidence.index):
        raise ValueError("Aligned original candidate/score/evidence indices required")
    if not np.isfinite(scores).all() or not scores.between(0, 1).all():
        raise ValueError("Finite combined-mode agreement ranks required")
    p = pool.assign(
        _species=pool.name.map(runner.inherited.common.species),
        _ion=pool.adduct.map(runner.inherited.common.adduct),
        _rank=scores,
    )
    ordered = p.sort_values(["_rank", "feat_intensity"], ascending=False, kind="stable")
    groups = {
        k: g for k, g in ordered.groupby(["stream", "_species", "_ion"], sort=False)
    }
    normalized = _normalise(current, manual=False)
    normalized.index = current.index
    normalized["adduct"] = normalized.adduct.map(runner.inherited.common.adduct)
    duplicate = normalized.duplicated(["stream", "species", "adduct"], keep=False)
    occupied_keys = set(zip(occupied.stream, occupied.feat_feature_id.astype(str)))
    rows = []
    skipped = Counter()
    for idx, r in normalized.iterrows():
        if r.internal_standard or duplicate.loc[idx]:
            skipped["standard_or_duplicate_old_identity"] += 1
            continue
        key = (r.stream, r.species, r.adduct)
        if key not in groups:
            skipped["identity_absent_from_core"] += 1
            continue
        g = groups[key]
        oldfid = str(current.loc[idx, "feat_feature_id"])
        old = g.loc[g.feat_feature_id.astype(str).eq(oldfid)]
        if old.empty:
            skipped["old_feature_absent_from_core"] += 1
            continue
        previous, best = old.iloc[0], g.iloc[0]
        oldindex, newindex = old.index[0], g.index[0]
        gap = float(best["_rank"]) - float(previous["_rank"])
        if str(best.feat_feature_id) == oldfid or gap <= 0:
            skipped["no_stronger_different_feature"] += 1
            continue
        free = (r.stream, str(best.feat_feature_id)) not in occupied_keys
        clear = newindex in clear_indices
        acceptable = bool(physical.loc[newindex]) and not bool(protected.loc[oldindex])
        profile = bool(
            evidence.loc[newindex, "q_raw_counterpart_profile_measurement_available"]
            == 1
        )
        row = dict(
            stream=r.stream,
            species=r.species,
            adduct=r.adduct,
            lipid_class=_lipid_class(r.lipid_class),
            old_feature_id=oldfid,
            new_feature_id=str(best.feat_feature_id),
            old_combined_rank=float(previous["_rank"]),
            new_combined_rank=float(best["_rank"]),
            rank_gap=gap,
            identity_only=True,
            clear_physical_free_target=bool(free and clear and acceptable),
            clear_physical_free_profile=bool(free and clear and acceptable and profile),
            target_is_free=free,
            target_is_clear=clear,
            target_is_physical=acceptable,
            new_profile_available=profile,
            old_profile_available=bool(
                evidence.loc[
                    oldindex, "q_raw_counterpart_profile_measurement_available"
                ]
                == 1
            ),
        )
        rows.append(row)
    columns = [
        "stream",
        "species",
        "adduct",
        "lipid_class",
        "old_feature_id",
        "new_feature_id",
        "old_combined_rank",
        "new_combined_rank",
        "rank_gap",
        *SCREENS,
        "target_is_free",
        "target_is_clear",
        "target_is_physical",
        "new_profile_available",
        "old_profile_available",
    ]
    return pd.DataFrame(rows, columns=columns), dict(skipped)


def add_truth(moves, manual):
    """Gold SETS, not one overwritten FID when a manual identity has isomers."""
    gold = {}
    identities, unresolved = set(), set()
    for r in manual.loc[~manual.internal_standard].itertuples():
        key = (
            r.stream,
            runner.inherited.common.species(r.species),
            runner.inherited.common.adduct(r.adduct),
        )
        identities.add(key)
        if r.mapped:
            gold.setdefault(key, set()).add(str(r.feature_id))
        else:
            unresolved.add(key)
    rows = moves.copy()
    changes = []
    for r in rows.itertuples():
        key = (r.stream, r.species, r.adduct)
        before = str(r.old_feature_id) in gold.get(key, set())
        after = str(r.new_feature_id) in gold.get(key, set())
        if before and after:
            kind = "correct_to_other_manual_peak"
        elif before and not after:
            kind = "correct_to_no_manual_match"
        elif not before and after:
            kind = "no_manual_match_to_correct"
        elif key not in identities:
            kind = "unconfirmed_identity_move"
        elif key in unresolved:
            kind = "unresolved_identity_move"
        else:
            kind = "wrong_to_other_wrong"
        changes.append(kind)
    rows["diagnostic_change"] = changes
    return rows


def freeze_outer(outer):
    configure_frozen_resources(runner.OUT / "source_snapshot")
    from matlab_annotation.mode_evidence_experts import predict_mode_evidence_experts
    from matlab_annotation.guarded_class_operating_point import clear_identity_winners

    datasets, folds = runner.inventory()
    artifacts = {}
    skipped = []
    for d in datasets:
        inner = folds[d]
        if inner == outer:
            continue
        raw = runner.load_predictors(d, {outer, inner})
        modelpath = runner.parent_path(outer, inner)
        model = pd.read_pickle(modelpath)
        assert (
            runner.sha(modelpath)
            == json.loads(modelpath.with_suffix(".complete.json").read_text())[
                "model_sha256"
            ]
        )
        ranks = predict_mode_evidence_experts(raw, model)
        pd.testing.assert_series_equal(
            ranks,
            pd.read_pickle(
                runner.OUT / "calibration" / f"fold_{outer}" / (d + "_blind_scores.pkl")
            ),
            check_exact=True,
        )
        oldmodel = pd.read_pickle(
            runner.CORE / "calibration" / f"fold_{outer}/inner_block_{inner}_model.pkl"
        )
        _, oldscores = runner.inherited.reproduce(
            d,
            {outer, inner},
            oldmodel,
            runner.CORE / "calibration" / f"fold_{outer}" / (d + "_blind_scores.pkl"),
        )
        # Pure041occupancy reconstruction needs no training flags or curation.
        occupied, current = runner.inner_previous(d, outer, inner, raw, oldscores)
        allowed = pd.Series(False, index=raw["pool"].index)
        protected = allowed.copy()
        clear = set()
        for stream in runner.MODES:
            part = raw["pool"].loc[raw["pool"].stream.eq(stream)]
            protection = runner.inherited.common.physical_protection(
                d, {outer, inner}, stream, raw
            )
            allowed.loc[part.index], protected.loc[part.index] = physical_masks(
                part, stream, protection
            )
            clear.update(
                clear_identity_winners(
                    part, ranks.loc[part.index], ambiguity_margin=0.05
                ).index
            )
        moves, counts = blind_moves(
            raw["pool"],
            ranks,
            raw["evidence"],
            current,
            occupied,
            allowed,
            protected,
            clear,
        )
        path = OUT / "blind" / f"outer_{outer}" / (d + ".pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        moves.to_pickle(path)
        artifacts[str(path)] = runner.sha(path)
        skipped.append(dict(outer=outer, dataset=d, **counts))
        print(
            "BLIND COMBINED-MODE PEAK RANK OPTIONS FROZEN",
            outer,
            d,
            len(moves),
            flush=True,
        )
    return artifacts, skipped


def main():
    configure_frozen_resources(runner.OUT / "source_snapshot")
    protocol, datasets = runner.initialize()
    receipt = json.loads((runner.OUT / "integrity_receipt.json").read_text())
    assert receipt["all_passed"] and receipt["fresh_classifiers"] == 100
    assert receipt["planned_verifier_sha256"] == protocol["planned_verifier_sha256"]
    _, folds = runner.inventory()
    files = {
        Path(__file__),
        runner.OUT / "protocol.json",
        runner.OUT / "integrity_receipt.json",
    }
    files.update(map(Path, protocol["source_input_sha256"]))
    files.update(p for p in (runner.OUT / "calibration").rglob("*") if p.is_file())
    hashes = {str(p): runner.sha(p) for p in sorted(files)}
    OUT.mkdir(exist_ok=False)
    with ProcessPoolExecutor(max_workers=3) as executor:
        outputs = list(executor.map(freeze_outer, range(5)))
    artifacts = {p: h for paths, _ in outputs for p, h in paths.items()}
    skipped = [r for _, rows in outputs for r in rows]
    assert len(artifacts) == len(skipped) == 132
    runner.dump(
        OUT / "blind_receipt.json",
        dict(
            all132_physical_options_frozen=True,
            training_truth_tables_parsed_by_blind_stage=0,
            classifier_fits=0,
            annotation_changes=False,
            source_input_sha256=hashes,
            physical_sha256=artifacts,
        ),
    )
    counts = Counter()
    for outer in range(5):
        allowed = {d[:7] for d in datasets if folds[d] != outer}
        excluded = {d[:7] for d in datasets if folds[d] == outer}
        for d in datasets:
            inner = folds[d]
            if inner == outer:
                continue
            raw = runner.load_predictors(d, {outer, inner})
            _, manual, _ = runner.authorize_labels(raw, allowed, excluded)
            moves = pd.read_pickle(OUT / "blind" / f"outer_{outer}" / (d + ".pkl"))
            labeled = add_truth(moves, manual)
            for screen in SCREENS:
                for gap in GAPS:
                    selected = labeled.loc[labeled[screen] & labeled.rank_gap.gt(gap)]
                    for (s, c, k), n in (
                        selected.groupby(["stream", "lipid_class", "diagnostic_change"])
                        .size()
                        .items()
                    ):
                        counts[(outer, d, s, c, screen, gap, k)] += int(n)
    table = pd.DataFrame(
        [
            dict(
                outer=o,
                dataset=d,
                stream=s,
                lipid_class=c,
                screen=screen,
                minimum_rank_gain=g,
                change=k,
                count=n,
            )
            for (o, d, s, c, screen, g, k), n in sorted(counts.items())
        ]
    )
    table.to_csv(OUT / "by_outer_dataset_mode_class.csv", index=False)
    table.groupby(
        ["stream", "lipid_class", "screen", "minimum_rank_gain", "change"],
        as_index=False,
    )["count"].sum().to_csv(OUT / "by_mode_class.csv", index=False)
    pd.DataFrame(skipped).to_csv(OUT / "skipped_cases.csv", index=False)
    assert all(runner.sha(Path(p)) == h for p, h in hashes.items())
    assert all(runner.sha(Path(p)) == h for p, h in artifacts.items())
    runner.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            training_query_cases=132,
            classifier_fits=0,
            annotation_changes=False,
            matching_outer_case_truth_used=False,
            source_input_sha256=hashes,
            physical_sha256=artifacts,
            note="DiagnosticONLY, no thresholdselection/newfits/annotations. ALL132blindpotentialmoves frozen BEFORE permittedtrainingtruth. Same dataset repeatsfouroutertrainingcontexts; NOTbenchmarkgains/independentobservations. GoldsetsallowmultiplecuratedFIDs. Fixedidentitywinnerbeforefilters/no fallback. Screens report originalphysicalguards/globalclearwinner/free-target/newprofileavailability but competing moves may shareonetarget;no globallycompatibleassignment or calibrationvalidated. Unknown/unresolvedmovesnotcountedfalse. 041reference/defaultsunchanged.",
        ),
    )
    print(
        table.groupby(["screen", "minimum_rank_gain", "change"])["count"]
        .sum()
        .to_string(),
        flush=True,
    )


if __name__ == "__main__":
    main()
