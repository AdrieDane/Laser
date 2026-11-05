"""Real process activity, durable failure status and terminal output safety."""
import json
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from application.annotation_jobs import read_annotation_job
from application.job_monitor import parse_cpu_time, process_activity
from presentation.job_terminal import terminal_payload
from standard_lipidomics.untargeted import analyse_standard_untargeted, UntargetedComparisonCriteria
import test_standard_untargeted as fixtures


@pytest.mark.parametrize('text,seconds', [('1:02.50', 62.5), ('02:03:04', 7384.), ('2-01:02:03', 176523.)])
def test_native_cpu_time_formats(text, seconds):
    assert parse_cpu_time(text) == seconds


def test_current_process_reports_running_with_cpu_measurement():
    activity = process_activity(os.getpid())
    assert activity['alive']
    assert activity['cpu_seconds'] is not None


@pytest.mark.skipif(os.name == 'nt', reason='POSIX process states')
def test_zombie_worker_is_stopped_even_when_pid_exists(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps({'status': 'running', 'message': 'searching'}))
    (tmp_path/'process.json').write_text(json.dumps({'pid': 42}))
    with patch('application.job_monitor.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout='Z 0:00.00')):
        state = read_annotation_job(tmp_path)
    assert state['status'] == 'failed'
    assert not state['activity']['alive']
    assert 'stopped' in state['message']


@pytest.mark.skipif(os.name == 'nt', reason='POSIX process sampling')
def test_sampler_timeout_does_not_claim_worker_failed():
    with patch('application.job_monitor.subprocess.run', side_effect=subprocess.TimeoutExpired('ps', 1)):
        assert process_activity(os.getpid()) == {'alive': True, 'cpu_seconds': None}


def test_terminal_retains_traceback_and_completed_output(tmp_path):
    (tmp_path/'untargeted.log').write_text('Starting\nTraceback: <script>unsafe</script>\n')
    payload = terminal_payload(tmp_path, {'status': 'failed', 'message': 'worker stopped'}, 'untargeted.log')
    assert 'STOPPED / FAILED' in payload['activity']
    assert 'Traceback' in payload['output']
    # Content is sent as plain text; the renderer assigns textContent only.
    assert '<script>' in payload['output']
    complete = terminal_payload(tmp_path, {'status': 'complete'}, 'untargeted.log')
    assert complete['activity'] == 'COMPLETE'
    assert complete['output'] == payload['output']


def test_terminal_distinguishes_silent_running_worker_and_missing_log(tmp_path):
    payload = terminal_payload(tmp_path, {'status': 'running', 'activity': {'pid': 42, 'cpu_seconds': 21.5}}, 'annotation.log')
    assert 'RUNNING' in payload['activity']
    assert 'CPU time 21.5 s' in payload['activity']
    assert 'Waiting for output' in payload['activity']
    assert 'Last checked' in payload['checked']


def test_progress_counters_leave_scientific_results_identical():
    criteria = UntargetedComparisonCriteria('Group_01_control', 'Group_02_disease')
    result = fixtures.StandardUntargetedTests()._result()
    original = analyse_standard_untargeted(result, criteria)
    events = []
    monitored = analyse_standard_untargeted(result, criteria, progress=lambda value, message: events.append((value, message)))
    for key in ('features', 'raw_p_features', 'adduct_matches', 'homologous_matches'):
        pd.testing.assert_frame_equal(original[key], monitored[key])
    assert any('Adduct search:' in message and 'feature' in message for _, message in events)
    assert any('Cross-mode abundance search:' in message for _, message in events)
    assert all(a[0] <= b[0] for a, b in zip(events, events[1:]))
