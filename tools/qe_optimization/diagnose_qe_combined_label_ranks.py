"""Read-only wrong-label/right-original-feature audit of 132 INNER cases.

All proposals freeze before permitted training truth is decoded. No fits,
threshold selection, annotation changes or deployment performance claims.
The same dataset repeats four outer-training contexts, not independent gains.
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from frozen_resources import configure_frozen_resources
import diagnose_qe_combined_peak_ranks as peak

runner = peak.runner
OUT = runner.QE / "combined_raw_model_label_rank_audit_v1"
GAPS = peak.GAPS
SCREENS = ("feature_only", "clear_physical_unused_identity", "clear_physical_profile")


def blind_moves(pool, scores, evidence, current, occupied, physical, protected, clear):
    from matlab_annotation.raw_counterpart_rt import SUPERVISION
    from standard_lipidomics.validation import _normalise, _lipid_class

    if SUPERVISION & set(pool) or SUPERVISION & set(current):
        raise ValueError("Blind label ranking refuses supervised flags")
    if (
        not pool.index.is_unique
        or not pool.index.equals(scores.index)
        or not pool.index.equals(evidence.index)
    ):
        raise ValueError("Aligned unique candidate/score/evidence rows required")
    if not np.isfinite(scores).all() or not scores.between(0, 1).all():
        raise ValueError("Finite agreement ranks required")
    if not current.index.is_unique:
        raise ValueError("Unique current annotation row indices required")
    ranked = pool.assign(
        _fid=pool.feat_feature_id.astype(str),
        _species=pool.name.map(runner.inherited.common.species),
        _ion=pool.adduct.map(runner.inherited.common.adduct),
        _rank=scores,
    ).sort_values(["_rank", "feat_intensity"], ascending=False, kind="stable")
    groups = {k: g for k, g in ranked.groupby(["stream", "_fid"], sort=False)}
    # Identity's absolute best feature is fixed BEFORE any physical/occupancy
    # screen. A blocked best must not promote a weaker alternative.
    identity_best = {
        k: g.index[0]
        for k, g in ranked.groupby(["stream", "_species", "_ion"], sort=False)
    }
    normalized = _normalise(current, manual=False)
    normalized.index = current.index
    normalized["adduct"] = normalized.adduct.map(runner.inherited.common.adduct)
    duplicate = normalized.duplicated(["stream", "species", "adduct"], keep=False)
    duplicate |= current.assign(_fid=current.feat_feature_id.astype(str)).duplicated(
        ["stream", "_fid"], keep=False
    )
    other = _normalise(occupied, manual=False)
    other["adduct"] = other.adduct.map(runner.inherited.common.adduct)
    used = set(zip(other.stream, other.species, other.adduct))
    rows, skipped = [], Counter()
    for idx, r in normalized.iterrows():
        if r.internal_standard or duplicate.loc[idx]:
            skipped["standard_or_ambiguous_old_call"] += 1
            continue
        fid = str(current.loc[idx, "feat_feature_id"])
        # Explicit strings prevent numeric FID dtype from dropping a match.
        g = groups.get((r.stream, fid))
        if g is None:
            skipped["old_feature_absent_from_core"] += 1
            continue
        old = g.loc[g._species.eq(r.species) & g._ion.eq(r.adduct)]
        if old.empty:
            skipped["old_identity_score_unknown"] += 1
            continue
        previous, best = old.iloc[0], g.iloc[0]
        oi, ni = old.index[0], g.index[0]
        gap = float(best["_rank"]) - float(previous["_rank"])
        newkey = (r.stream, best["_species"], best["_ion"])
        if newkey == (r.stream, r.species, r.adduct) or gap <= 0:
            skipped["no_stronger_different_label"] += 1
            continue
        unused = newkey not in used
        absolute_best = identity_best[newkey] == ni
        allowed = bool(physical.loc[ni]) and not bool(protected.loc[oi])
        eligible = unused and absolute_best and ni in clear and allowed
        profile = (
            evidence.loc[ni, "q_raw_counterpart_profile_measurement_available"] == 1
        )
        rows.append(
            dict(
                stream=r.stream,
                feature_id=fid,
                old_species=r.species,
                old_adduct=r.adduct,
                new_species=best["_species"],
                new_adduct=best["_ion"],
                old_class=_lipid_class(r.lipid_class),
                new_class=_lipid_class(best["class"]),
                old_rank=float(previous["_rank"]),
                new_rank=float(best["_rank"]),
                rank_gap=gap,
                feature_only=True,
                clear_physical_unused_identity=bool(eligible),
                clear_physical_profile=bool(eligible and profile),
                new_identity_unused=unused,
                new_identity_best_here=absolute_best,
                target_clear=ni in clear,
                target_physical=allowed,
            )
        )
    columns = [
        "stream",
        "feature_id",
        "old_species",
        "old_adduct",
        "new_species",
        "new_adduct",
        "old_class",
        "new_class",
        "old_rank",
        "new_rank",
        "rank_gap",
        *SCREENS,
        "new_identity_unused",
        "new_identity_best_here",
        "target_clear",
        "target_physical",
    ]
    return pd.DataFrame(rows, columns=columns), dict(skipped)


def add_truth(moves, manual):
    gold, identities, unresolved = {}, set(), set()
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
    changes, extras, wrongs = [], [], []
    for r in rows.itertuples():
        old = (r.stream, r.old_species, r.old_adduct)
        new = (r.stream, r.new_species, r.new_adduct)
        before, after = (
            r.feature_id in gold.get(old, set()),
            r.feature_id in gold.get(new, set()),
        )
        if before and after:
            kind = "manual_to_manual_label"
        elif before:
            kind = "correct_to_no_manual_match"
        elif after:
            kind = "no_manual_match_to_correct"
        elif old in unresolved or new in unresolved:
            kind = "unresolved_comparison"
        else:
            kind = "neither_label_matches_manual_peak"
        changes.append(kind)
        extras.append(int(new not in identities) - int(old not in identities))
        wrongs.append(int(new in gold and not after) - int(old in gold and not before))
    rows["diagnostic_change"] = changes
    rows["unconfirmed_extra_delta"] = extras
    rows["different_peak_delta"] = wrongs
    return rows


def freeze_outer(outer):
    configure_frozen_resources(runner.OUT / "source_snapshot")
    from matlab_annotation.mode_evidence_experts import predict_mode_evidence_experts
    from matlab_annotation.guarded_class_operating_point import clear_identity_winners

    datasets, folds = runner.inventory()
    artifacts, skipped = {}, []
    for d in datasets:
        inner = folds[d]
        if inner == outer:
            continue
        raw = runner.load_predictors(d, {outer, inner})
        modelpath = runner.parent_path(outer, inner)
        assert (
            runner.sha(modelpath)
            == json.loads(modelpath.with_suffix(".complete.json").read_text())[
                "model_sha256"
            ]
        )
        ranks = predict_mode_evidence_experts(raw, pd.read_pickle(modelpath))
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
        occupied, current = runner.inner_previous(d, outer, inner, raw, oldscores)
        allowed = pd.Series(False, index=raw["pool"].index)
        protected, clear = allowed.copy(), set()
        for stream in runner.MODES:
            part = raw["pool"].loc[raw["pool"].stream.eq(stream)]
            protection = runner.inherited.common.physical_protection(
                d, {outer, inner}, stream, raw
            )
            allowed.loc[part.index], protected.loc[part.index] = peak.physical_masks(
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
            "BLIND SAME-FEATURE LABEL OPTIONS FROZEN", outer, d, len(moves), flush=True
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
        Path(peak.__file__),
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
            all132_proposals_frozen=True,
            training_truth_tables_parsed_by_blind_stage=0,
            classifier_fits=0,
            annotation_changes=False,
            source_input_sha256=hashes,
            proposal_sha256=artifacts,
        ),
    )
    counts, extras, wrongs = Counter(), Counter(), Counter()
    for outer in range(5):
        allowed = {d[:7] for d in datasets if folds[d] != outer}
        excluded = {d[:7] for d in datasets if folds[d] == outer}
        for d in datasets:
            inner = folds[d]
            if inner == outer:
                continue
            raw = runner.load_predictors(d, {outer, inner})
            _, manual, _ = runner.authorize_labels(raw, allowed, excluded)
            moves = add_truth(
                pd.read_pickle(OUT / "blind" / f"outer_{outer}" / (d + ".pkl")), manual
            )
            for screen in SCREENS:
                for gap in GAPS:
                    chosen = moves.loc[moves[screen] & moves.rank_gap.gt(gap)]
                    for key, part in chosen.groupby(
                        ["stream", "old_class", "new_class", "diagnostic_change"]
                    ):
                        k = (outer, d, screen, gap, *key)
                        counts[k] += len(part)
                        extras[k] += int(part.unconfirmed_extra_delta.sum())
                        wrongs[k] += int(part.different_peak_delta.sum())
    table = pd.DataFrame(
        [
            dict(
                outer=k[0],
                dataset=k[1],
                screen=k[2],
                minimum_rank_gain=k[3],
                stream=k[4],
                old_class=k[5],
                new_class=k[6],
                change=k[7],
                count=n,
                unconfirmed_extra_delta=extras[k],
                different_peak_delta=wrongs[k],
            )
            for k, n in sorted(counts.items())
        ]
    )
    table.to_csv(OUT / "by_outer_dataset_mode_class.csv", index=False)
    sums = ["count", "unconfirmed_extra_delta", "different_peak_delta"]
    table.groupby(
        ["screen", "minimum_rank_gain", "stream", "old_class", "new_class", "change"],
        as_index=False,
    )[sums].sum().to_csv(OUT / "by_mode_class.csv", index=False)
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
            proposal_sha256=artifacts,
            note="Diagnostic only; same-feature label winner fixed BEFORE screens, no weaker fallback, original identity best feature required for guarded screen. All132blind proposals frozen before authorized training truth. Gold sets preserve multiple curated FIDs. Repeats four outer-training contexts per dataset, NOT benchmark gains. Unknown absence is not false. Simultaneous moves/global compatibility and calibration NOT validated. Whole041/defaults unchanged.",
        ),
    )
    print(
        table.groupby(["screen", "minimum_rank_gain", "change"])[sums]
        .sum()
        .to_string(),
        flush=True,
    )


if __name__ == "__main__":
    main()
