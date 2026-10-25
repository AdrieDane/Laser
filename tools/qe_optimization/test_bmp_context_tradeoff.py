"""Fixed whole-method decision boundaries, no actual experimental outcomes."""

from context import REPO
from frozen_resources import configure_frozen_resources

configure_frozen_resources(REPO)
from report_qe_bmp_context_tradeoff import assess


def arguments():
    return dict(
        gains=100,
        extras=100,
        unsupported_extras=20,
        wrong_by_mode=dict(NP_Pos=3, NP_Neg=2),
        endogenous_by_mode=dict(NP_Pos=600, NP_Neg=500),
        old_correct_lost=0,
        changed_outside_BMP=0,
        BMP_modes_improved=2,
    )


def test_whole_budget_pass_is_not_deployment_and_no_favourable_mode_exception():
    values = arguments()
    assert assess(**values)["retain_whole_research_method"]
    for key, value in (
        ("gains", 99),
        ("extras", 101),
        ("old_correct_lost", 1),
        ("changed_outside_BMP", 1),
        ("BMP_modes_improved", 0),
    ):
        broken = dict(values, **{key: value})
        assert not assess(**broken)["retain_whole_research_method"]


def test_absolute_extra_cap_and_unsupported_composition_budget_both_apply():
    values = dict(arguments(), gains=700, extras=572)
    result = assess(**values)
    assert result["checks"]["new_total_extras_no_more_than_new_correct"]
    assert not result["checks"]["new_total_extras_no_more_than_571"]
    values = dict(arguments(), unsupported_extras=111)
    assert not assess(**values)["retain_whole_research_method"]


def test_wrong_peak_cannot_be_traded_for_other_polarity_or_total_coverage():
    values = dict(
        arguments(), gains=700, extras=500, wrong_by_mode=dict(NP_Pos=4, NP_Neg=0)
    )
    result = assess(**values)
    assert result["checks"]["new_wrong_no_more_than_five"]
    assert not result["checks"]["new_wrong_per_mode_budget"]
