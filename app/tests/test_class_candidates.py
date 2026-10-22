import numpy as np
import pandas as pd
import pytest

from matlab_annotation.class_candidates import (
    candidate_evidence, prepare_class_candidates, score_linear_evidence,
    select_class_candidates, fit_class_composition_priors, attach_composition_prior_evidence,
)
from matlab_annotation.retention_atlas import fit_retention_atlas


def inputs():
    train = pd.DataFrame([dict(cohort=str(g), stream="RP_Pos", **{"class":"PE"},
                               carbon=c, db=d, rt_seconds=300+10*(c-34)-20*d)
                          for g in range(4) for c in (34,36,38) for d in (0,1,2)])
    atlas = fit_retention_atlas(train)
    reference = pd.DataFrame({"name":["PE(36:1)"], "class":["PE"], "adduct":["[M+H]+"],
                               "carbon":[36], "db":[1], "ref_mz":[700.0], "ref_rt":[8.0]})
    features = pd.DataFrame({"feature_id":["0017","18","19"],
                              "mz":[700.0001,700.0001,700.004], "rt":[5.01,5.5,5.0],
                              "intensity":[1e5,1e6,1e7]})
    annotations = pd.DataFrame(columns=["feat_feature_id","is_is"])
    return annotations,reference,features,atlas


def test_original_feature_identity_and_strict_windows():
    annotations,reference,features,atlas=inputs()
    original=features.copy(deep=True)
    pool,audit=prepare_class_candidates(annotations,reference,features,stream="RP_Pos",atlas=atlas)
    assert pool.feat_feature_id.tolist()==["0017"]
    assert audit["atlas_targets"]==1
    assert abs(pool.rt_error.iloc[0]*60-.6)<.1
    pd.testing.assert_frame_equal(features,original)
    with pytest.raises(ValueError,match="3 ppm"):
        prepare_class_candidates(annotations,reference,features,stream="RP_Pos",atlas=atlas,ppm=6)


def test_standards_are_reserved_and_duplicate_ids_rejected():
    annotations,reference,features,atlas=inputs()
    annotations=pd.DataFrame({"feat_feature_id":["0017"],"is_is":[True]})
    pool,audit=prepare_class_candidates(annotations,reference,features,stream="RP_Pos",atlas=atlas)
    assert pool.empty and audit["protected_standard_features"]==1
    with pytest.raises(ValueError,match="unique"):
        prepare_class_candidates(annotations,reference,pd.concat([features,features]),stream="RP_Pos",atlas=atlas)


def test_evidence_has_no_feature_or_species_identifiers():
    a,r,f,atlas=inputs()
    pool,_=prepare_class_candidates(a,r,f,stream="RP_Pos",atlas=atlas)
    evidence=candidate_evidence(pool)
    assert not {"name","feat_feature_id","dataset","cohort"}&set(evidence)
    model={"format":"laser-class-evidence-logistic-v1","intercept":0,
           "coefficients":{"absolute_mass_error_ppm":-2,"class=PE":1}}
    expected=1/(1+np.exp(-(1-2*evidence.absolute_mass_error_ppm)))
    np.testing.assert_allclose(score_linear_evidence(evidence,model),expected)


def test_selection_does_not_force_ambiguous_calls_or_duplicate_features():
    a,r,f,atlas=inputs()
    pool,_=prepare_class_candidates(a,r,f,stream="RP_Pos",atlas=atlas)
    rival=pool.assign(name="rival",**{"class":"rival"})
    both=pd.concat([pool,rival],ignore_index=True)
    assert select_class_candidates(both,pd.Series([.8,.79])).empty
    selected=select_class_candidates(both,pd.Series([.95,.65]))
    assert selected.name.tolist()==["PE(36:1)"]
    assert not selected.feat_feature_id.duplicated().any()


def test_composition_prior_is_smoothed_and_counts_cohorts_not_repeated_rows():
    a,r,f,atlas=inputs()
    pool,_=prepare_class_candidates(a,r,f,stream="RP_Pos",atlas=atlas)
    data=pd.DataFrame({"cohort":["a","a","b"],"stream":["RP_Pos"]*3,
                       "class":["PE"]*3,"carbon":[36]*3,"db":[1]*3})
    priors=fit_class_composition_priors(data)
    result=attach_composition_prior_evidence(candidate_evidence(pool),priors)
    assert result.composition_training_cohorts.iloc[0]==2
    assert result.composition_joint_frequency.iloc[0]==pytest.approx(2.5/3)
    novel=candidate_evidence(pool).assign(carbon=50)
    result=attach_composition_prior_evidence(novel,priors)
    assert result.composition_joint_frequency.iloc[0]>0
