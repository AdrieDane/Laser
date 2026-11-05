"""Detached workers survive their launcher and write complete GUI projects."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from application.annotation_jobs import start_annotation_job, read_annotation_job
from resource_paths import REFERENCE_DATA_DIRECTORY
from standard_lipidomics.project_store import load_annotation_workflow

APP=Path(__file__).resolve().parents[1]


def setup(root):
    (root/'XCMS').mkdir()
    (root/'metadata').mkdir()
    (root/'XCMS'/'peakgrouplist_RP_Pos.txt').write_text('name\tmzmed\trtmed\tnpeaks\tsample_a\n1\t734.569\t7\t1\t1000\n')
    (root/'metadata'/'internal_standards.csv').write_text('name,concentration,unit\nPC(14:0)2,10000,nM\n')
    return {'engine':'laser','inputs':{'project':str(root),'reference':str(REFERENCE_DATA_DIRECTORY/'4d_lipid_reference.csv')},
            'annotation':{'disable_np_rt_reference':True,'use_grid_coherence':False,'adapt_class_rt':False}}


def wait_job(job):
    for _ in range(150):
        state=read_annotation_job(job)
        if state['status'] in {'complete','failed'}:
            return state
        time.sleep(.1)
    raise AssertionError('Worker did not finish within 15 seconds')


def test_worker_outlives_launcher_and_saves_reopenable_snapshot(tmp_path):
    config=setup(tmp_path)
    path=tmp_path/'config.json'
    path.write_text(json.dumps(config))
    env={**os.environ,'PYTHONPATH':str(APP)}
    launcher=subprocess.run([sys.executable,'-c',
        'import json,sys;from application.annotation_jobs import start_annotation_job;print(start_annotation_job(json.load(open(sys.argv[1]))))',str(path)],
        env=env,capture_output=True,text=True,check=True)
    job=Path(launcher.stdout.strip().splitlines()[-1])
    # The CSV can change after launch without altering the selected run inputs.
    (tmp_path/'metadata'/'internal_standards.csv').write_text('name,concentration,unit\nPC(14:0)2,20000,nM\n')
    state=wait_job(job)
    assert state['status']=='complete', (job/'annotation.log').read_text()
    result,_=load_annotation_workflow(tmp_path)
    assert result['internal_standards_expected'][0]['quantity']==10000
    assert (job/'inputs'/'internal_standards.csv').is_file()
    assert 'Annotation complete' in (job/'annotation.log').read_text()


def test_worker_failure_has_log_and_never_claims_completion(tmp_path):
    config=setup(tmp_path)
    config['inputs']['reference']=str(tmp_path/'missing.csv')
    job=start_annotation_job(config)
    state=wait_job(job)
    assert state['status']=='failed'
    assert 'Traceback' in (job/'annotation.log').read_text()
    assert not (tmp_path/'LASER'/'LASER_PROJECT.json').exists()


def test_duplicate_start_reattaches_to_the_active_job(tmp_path):
    config=setup(tmp_path)
    with patch('application.annotation_jobs.subprocess.Popen') as process:
        process.return_value.pid=os.getpid()
        first=start_annotation_job(config)
        second=start_annotation_job(config)
        assert first==second
        assert process.call_count==1
        if os.name!='nt':
            assert process.call_args.kwargs['start_new_session']


def test_finished_gui_has_only_completion_and_back_to_start(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps({'status':'complete','message':'Annotation complete'}))
    app=AppTest.from_file(str(APP/'app.py'),default_timeout=60).run()
    app.session_state['standard_pipeline_page']='annotation_job'
    app.session_state['standard_annotation_job_path']=str(tmp_path)
    app.run()
    assert not app.exception
    assert [item.value for item in app.success]==['Annotation complete']
    assert not app.tabs
    assert not any(button.label=='Open annotation review' for button in app.button)
    app.button(key='standard-job-back-to-start').click().run()
    assert not app.exception
    assert app.button(key='standard-module-annotation')
