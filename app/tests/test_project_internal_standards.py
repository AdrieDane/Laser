"""Project CSV ownership, metadata quantities, calibration and annotation integration."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from annotator.mass_calibrants import detect_mass_calibration_candidates
from annotator.reference import load_reference
from matlab_annotation.configuration import DEFAULT_MATLAB_PIPELINE_REFERENCE
from matlab_annotation.engine import _internal_standard_project_dir
from matlab_annotation.vendor.reference_loader import load_reference_data
from resource_paths import REFERENCE_DATA_DIRECTORY
from standard_lipidomics.internal_standards import (
    find_project_is_file, project_is_definitions, project_is_reference, project_is_signature,
)
from standard_lipidomics.project_store import save_calibration_workflow, stage_project_metadata
from standard_lipidomics.workflow import annotate_standard_batch

APP = Path(__file__).resolve().parents[1] / 'app.py'


def project(root, quantity=10000):
    (root/'metadata').mkdir(exist_ok=True)
    path=root/'metadata'/'internal_standards.csv'
    path.write_text(f'name,concentration,unit\nPC(14:0)2,{quantity},nM\nCE(16:0)-D7,2500,nM\n')
    return path


def reference():
    return load_reference(REFERENCE_DATA_DIRECTORY/'4d_lipid_reference.csv')


def test_csv_drives_native_annotation_names_quantities_and_cache_signature(tmp_path):
    path=project(tmp_path)
    assert _internal_standard_project_dir(tmp_path,Path(DEFAULT_MATLAB_PIPELINE_REFERENCE))==path.parent
    root=Path(DEFAULT_MATLAB_PIPELINE_REFERENCE)
    data=load_reference_data(code_dir=root/'Code',database_dir=root/'DataBase',rtdb_dir=root/'RTDB',project_dir=path.parent)
    assert [(item.name,item.quantity,item.unit) for item in data.internal_standards]==[
        ('PC(14:0)2',10000.0,'nM'),('CE(16:0)-D7',2500.0,'nM')]
    assert 'project/internal_standards.csv' in data.source_sha256
    before=project_is_signature(tmp_path)
    project(tmp_path,20000)
    assert project_is_signature(tmp_path)!=before


@pytest.mark.parametrize('row',[
    'PC(14:0)2,NaN,nM','PC(14:0)2,-1,nM','PC(14:0)2,0,nM',
    'PC(14:0)2,100,','Unknown standard,100,nM',
    'PC(14:0)2,100,nM\nPC(14:0)2,200,nM',
])
def test_invalid_csv_never_silently_falls_back(tmp_path,row):
    path=project(tmp_path)
    path.write_text('name,concentration,unit\n'+row+'\n')
    with pytest.raises(ValueError):
        project_is_definitions(tmp_path)


def test_calibration_search_uses_declared_labelled_mass_and_concentration(tmp_path):
    project(tmp_path)
    panel=project_is_reference(tmp_path,reference(),stream='RP_Pos')
    assert set(panel.standard)=={'PC(14:0)2','CE(16:0)-D7'}
    labelled=panel[panel.standard.eq('CE(16:0)-D7') & panel.adduct.eq('[M+NH4]+')].iloc[0]
    assert labelled.ref_mz==pytest.approx(631.628469+18.033823,abs=3e-6)
    features=pd.DataFrame({'feature_id':['is1'],'mz':[labelled.ref_mz],'rt':[labelled.ref_rt],'intensity':[1000.]})
    found=detect_mass_calibration_candidates(features,panel,ppm_tol=3,rt_tol=.1)['candidates']
    assert found.iloc[0].standard=='CE(16:0)-D7'
    assert found.iloc[0].internal_standard_concentration==2500
    assert found.iloc[0].internal_standard_unit=='nM'


def test_reference_matcher_exports_project_is_quantities_and_only_declared_spikes(tmp_path):
    project(tmp_path)
    panel=project_is_reference(tmp_path,reference(),stream='RP_Pos')
    row=panel[panel.standard.eq('PC(14:0)2') & panel.adduct.eq('[M+H]+')].iloc[0]
    source=f'name\tmzmed\trtmed\tnpeaks\tsample_a\nis1\t{row.ref_mz:.12f}\t{row.ref_rt:.12f}\t1\t1000\n'.encode()
    result=annotate_standard_batch({'RP_Pos':source},reference(),internal_standard_project=tmp_path,
                                  use_grid_coherence=False,adapt_class_rt=False)
    standards=result['streams']['RP_Pos']['reference'].query('is_is == True')
    assert set(standards.standard)=={'PC(14:0)2','CE(16:0)-D7'}
    exported=result['all_candidates']
    matching=exported[exported.internal_standard_name.eq('PC(14:0)2')]
    assert not matching.empty
    assert matching.internal_standard_quantity.eq(10000).all()
    assert matching.internal_standard_unit.eq('nM').all()


def test_calibration_archive_keeps_its_own_csv_and_metadata_remains_authoritative(tmp_path):
    path=project(tmp_path)
    source=tmp_path/'peakgrouplist_RP_Pos.txt'
    source.write_bytes(b'name\tmzmed\trtmed\n1\t500\t5\n')
    (tmp_path/'internal_standards.csv').write_text('name,concentration,unit\nPC(14:0)2,1,nM\n')
    stage_project_metadata(tmp_path,{'RP_Pos':source})
    assert project_is_definitions(tmp_path)[1][0].concentration==10000
    save_calibration_workflow(tmp_path,{'RP_Pos':source},{})
    project(tmp_path,20000)
    _,archive=save_calibration_workflow(tmp_path,{'RP_Pos':source},{})
    assert find_project_is_file(archive)==archive/'internal_standards.csv'
    assert project_is_definitions(archive)[1][0].concentration==10000
    assert project_is_definitions(tmp_path/'Calibration')[1][0].concentration==20000
    assert path.is_file()


def test_gui_calibration_defaults_to_project_list(tmp_path):
    project(tmp_path)
    (tmp_path/'peakgrouplist_RP_Pos.txt').write_text('name\tmzmed\trtmed\n1\t500\t5\n')
    app=AppTest.from_file(str(APP),default_timeout=60).run()
    app.button(key='standard-module-calibration').click().run()
    next(w for w in app.text_input if w.label=='Project folder').set_value(str(tmp_path))
    next(b for b in app.button if b.label=='Review XCMS calibration').click().run()
    assert not app.exception
    choice=next(w for w in app.selectbox if w.label=='Reference m/z calibration list')
    assert choice.value=='Project internal standards'
    assert 'Unlabelled IS' not in choice.options
