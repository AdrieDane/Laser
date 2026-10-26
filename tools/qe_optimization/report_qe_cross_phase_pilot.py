"""Retrospective paired-peak sanity check, NOT an annotation experiment.

Added AFTER descriptive pilot exposure; no prespecified inference/threshold
claim. Frozen measurements unchanged; no new scores, labels or accepted calls.
"""

import json
from pathlib import Path

import pandas as pd

import prepare_qe_cross_phase_candidate_pilot as prep
import diagnose_qe_cross_phase_candidate_pilot as diagnostic

OUT = prep.common.QE / "cross_phase_class_candidate_paired_review_v1"


def main():
    receipt_path = diagnostic.OUT / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["all_passed"] and receipt["datasets"] == 33
    hashes = {
        str(Path(__file__)): prep.common.sha(Path(__file__)),
        str(receipt_path): prep.common.sha(receipt_path),
    }
    tables = []
    for p in sorted((diagnostic.OUT / "private").glob("*.pkl")):
        hashes[str(p)] = prep.common.sha(p)
        tables.append(pd.read_pickle(p))
    assert len(tables) == 33
    f = pd.concat(tables, ignore_index=True)
    prefix = prep.measurement.PREFIX
    f = f.loc[f.source_is_free & f[prefix + "source_physical"].eq(1)].copy()
    f["category"] = f.diagnostic_category.replace(
        {
            "manual_absent_identity_unconfirmed": "unconfirmed",
            "alternative_label_on_manual_peak_unconfirmed": "unconfirmed",
            "different_peak_for_manually_curated_identity": "wrong_peak_option",
            "manual_correct_currently_missing": "missing_correct_option",
        }
    )
    summaries, pairs = [], []
    fields = [
        "best_raw",
        "best_normalized",
        "best_within",
        "best_leave_one_out",
        "profile_margin",
    ]
    for (s, c), part in f.groupby(["stream", "lipid_class"]):
        for category, g in part.groupby("category"):
            measured = g.loc[g[prefix + "counterpart_profiles"].gt(0)]
            summaries.append(
                dict(
                    stream=s,
                    lipid_class=c,
                    category=category,
                    free_options=len(g),
                    options_with_partner=int(
                        g[prefix + "candidate_counterparts"].gt(0).sum()
                    ),
                    options_with_profile=len(measured),
                    **{
                        "median_" + field: measured[prefix + field].median()
                        for field in fields
                    },
                )
            )
        for field in fields:
            pos = neg = tie = 0
            projects = set()
            for key, g in part.groupby(["dataset", "stream", "species", "adduct"]):
                true = g.loc[
                    g.diagnostic_category.eq("manual_correct_currently_missing")
                ]
                wrong = g.loc[
                    g.diagnostic_category.eq(
                        "different_peak_for_manually_curated_identity"
                    )
                ]
                if true.feature_id.nunique() != 1 or wrong.empty:
                    continue
                a, b = true[prefix + field].dropna(), wrong[prefix + field].dropna()
                if a.empty or b.empty:
                    continue
                assert a.nunique() == 1
                delta = float(a.iloc[0] - b.max())
                if delta > 1e-9:
                    pos += 1
                elif delta < -1e-9:
                    neg += 1
                else:
                    tie += 1
                projects.add(key[0][:7])
            pairs.append(
                dict(
                    stream=s,
                    lipid_class=c,
                    field=field,
                    true_peak_higher=pos,
                    wrong_peak_higher=neg,
                    ties=tie,
                    comparable_identities=pos + neg + tie,
                    contributing_project_groups=len(projects),
                )
            )
    OUT.mkdir(exist_ok=False)
    pd.DataFrame(summaries).to_csv(
        OUT / "free_candidate_distributions.csv", index=False
    )
    pd.DataFrame(pairs).to_csv(OUT / "paired_peak_contrasts.csv", index=False)
    assert all(prep.common.sha(Path(p)) == h for p, h in hashes.items())
    prep.common.dump(
        OUT / "receipt.json",
        dict(
            all_passed=True,
            datasets=33,
            source_input_sha256=hashes,
            posthoc_descriptive_analysis=True,
            annotation_changes=False,
            classifier_fits=0,
            thresholds_selected=False,
            note="Within-identity peak comparisons are small exploratory counts,NOT annotation accuracy/gains or independent validation. Both raw measurements/labels alreadyfrozen; no predictor or accepted call changed. Wrong-peak options distinct from unconfirmed identities.",
        ),
    )
    print(pd.DataFrame(pairs).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
