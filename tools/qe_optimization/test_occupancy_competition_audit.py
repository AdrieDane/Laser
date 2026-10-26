import numpy as np
import pandas as pd

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
from occupancy_competition_audit import measure, independent_measure


def fixture():
    rows = [
        ("A(18:0)", "unused", 0.8),
        ("B(18:0)", "unused", 0.9),
        ("B(18:0)", "called", 0.98),
        ("C(18:0)", "unused", 0.6),
    ]
    pool = pd.DataFrame(
        [
            dict(
                stream="RP_Pos",
                name=n,
                **{"class": n.split("(")[0]},
                adduct="[M+H]+",
                feat_feature_id=f,
                feat_intensity=100,
                is_is=False,
            )
            for n, f, _ in rows
        ]
    )
    previous = pool.iloc[[2]].copy()
    return (
        pool,
        pd.Series([s for _, _, s in rows]),
        previous,
        pd.Series(True, index=pool.index),
    )


def compare(p, s, a, physical):
    x = measure(p, s, a, physical)
    y = independent_measure(p, s, a, physical)
    pd.testing.assert_frame_equal(x, y)
    return x


def test_original_best_is_blocked_by_identity_already_better_elsewhere():
    result = compare(*fixture())
    assert result.relaxed_opportunity.tolist() == [True, False, False, False]
    assert result.loc[0, "blocker_score_advantage"] == 0.98 - 0.9


def test_no_certificate_from_absent_or_weaker_called_peak():
    p, s, a, phy = fixture()
    a.loc[a.index[0], "feat_feature_id"] = "outside_core"
    assert not compare(p, s, a, phy).relaxed_opportunity.any()
    p, s, a, phy = fixture()
    s.loc[2] = 0.91
    assert not compare(p, s, a, phy).relaxed_opportunity.any()


def test_preserve_occupancy_ambiguity_identity_winner_and_physical_protection():
    p, s, a, phy = fixture()
    phy.loc[0] = False
    assert not compare(p, s, a, phy).relaxed_opportunity.any()
    p, s, a, phy = fixture()
    s.loc[3] = 0.79
    assert not compare(p, s, a, phy).relaxed_opportunity.any()
    p, s, a, phy = fixture()
    a = pd.concat([a, p.iloc[[3]]])
    assert not compare(p, s, a, phy).relaxed_opportunity.any()
    p, s, a, phy = fixture()
    p = pd.concat([p, p.iloc[[0]].assign(feat_feature_id="another")], ignore_index=True)
    s.loc[4] = 0.85
    phy.loc[4] = True
    assert not compare(p, s, a, phy).relaxed_opportunity.any()


def test_duplicate_versions_do_not_create_artificial_ambiguity():
    p, s, a, phy = fixture()
    p = pd.concat([p, p.iloc[[0]]], ignore_index=True)
    s.loc[4] = s.loc[0]
    phy.loc[4] = True
    result = compare(p, s, a, phy)
    assert result.relaxed_opportunity.tolist() == [True, False, False, False, False]
    assert np.isnan(result.loc[2, "blocker_score_advantage"])
