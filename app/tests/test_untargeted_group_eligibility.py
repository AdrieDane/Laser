"""QC and blank injections are displayed but never comparison groups."""
import pandas as pd
import pytest
from standard_lipidomics.sample_groups import biological_group, biological_label


@pytest.mark.parametrize('label', ['QC_BR_pool','Blanco_IS','Group_03_QC','pooled QC',
    'Blank','ProcessBlank','solventblank','quality control','QC-pool'])
def test_non_study_labels(label):
    assert not biological_label(label)


@pytest.mark.parametrize('label', ['BR_1','BR_2','healthy control','Group_01_control'])
def test_study_labels(label):
    assert biological_label(label)


def test_metadata_cannot_override_a_qc_or_blank_label_as_study():
    frame = pd.DataFrame()
    frame.attrs.update(sample_group_map={'a':'QC'},sample_type_map={'a':'study'})
    assert not biological_group(frame,'QC')
    frame.attrs.update(sample_group_map={'a':'A'},sample_group_name_map={'a':'Blanco_IS'})
    assert not biological_group(frame,'A')
    frame.attrs.update(sample_group_name_map={'a':'Cases'},sample_type_map={'a':'QC'})
    assert not biological_group(frame,'A')


@pytest.mark.parametrize('group,kind,name', [('QC','study','QC'),('B','QC','Cases'),
    ('B','blank','Cases'),('B','study','pooled QC'),('Blanco_IS','study','Blank')])
def test_terminal_analysis_rejects_non_biological_comparisons(group,kind,name):
    from standard_lipidomics.untargeted import analyse_standard_untargeted, UntargetedComparisonCriteria
    frame=pd.DataFrame({'feature_id':['x'],'mz':[500.], 'rt':[5.], 'a1':[1.], 'a2':[2.], 'b1':[20.], 'b2':[30.]})
    frame.attrs.update(sample_group_map={'a1':'A','a2':'A','b1':group,'b2':group},
        sample_type_map={'a1':'study','a2':'study','b1':kind,'b2':kind},
        sample_group_name_map={'a1':'Control','a2':'Control','b1':name,'b2':name})
    result={'streams':{'RP_Pos':{'features':frame}},'automatic_annotations':pd.DataFrame()}
    with pytest.raises(ValueError,match='biological'):
        analyse_standard_untargeted(result,UntargetedComparisonCriteria('A',group))
