"""Always-visible, read-only terminal output for detached jobs."""
from datetime import datetime
from pathlib import Path
import time

import streamlit as st


_JS = r'''
export default function({data, parentElement}) {
  let terminal = parentElement.__terminal;
  if (!terminal) {
    const shell = document.createElement('section');
    shell.className = 'terminal';
    shell.setAttribute('aria-label', 'Live process terminal');
    const header = document.createElement('div'); header.className = 'status';
    header.setAttribute('role', 'status');
    const output = document.createElement('pre');
    output.setAttribute('role', 'log');
    output.setAttribute('aria-label', 'Process output');
    output.tabIndex = 0;
    const footer = document.createElement('div'); footer.className = 'checked';
    shell.append(header, output, footer); parentElement.appendChild(shell);
    terminal = {header, output, footer, fresh:true};
    parentElement.__terminal = terminal;
  }
  const pinned = terminal.fresh || terminal.output.scrollHeight - terminal.output.scrollTop - terminal.output.clientHeight < 30;
  terminal.header.textContent = data.activity;
  terminal.header.dataset.status = data.status;
  if (terminal.output.textContent !== data.output) terminal.output.textContent = data.output;
  terminal.footer.textContent = data.checked;
  if (pinned) terminal.output.scrollTop = terminal.output.scrollHeight;
  terminal.fresh = false;
}
'''
_CSS = '''
:host { display:block; width:100%; font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
.terminal { height:310px; display:flex; flex-direction:column; background:#0d1117; color:#e6edf3; border:1px solid #30363d; border-radius:8px; overflow:hidden; }
.status { padding:10px 12px; color:#7ee787; background:#161b22; border-bottom:1px solid #30363d; }
.status[data-status="failed"] { color:#ff7b72; }
pre { margin:0; padding:12px; flex:1; min-height:0; overflow:auto; white-space:pre-wrap; overflow-wrap:anywhere; font:inherit; }
.checked { padding:6px 12px; color:#a5b2c1; border-top:1px solid #30363d; }
'''


def terminal_payload(directory, state, log_name):
    log = Path(directory) / log_name
    text = ''
    age = None
    if log.is_file():
        with log.open('rb') as handle:
            handle.seek(max(log.stat().st_size - 16000, 0))
            text = handle.read().decode('utf-8', errors='replace')
        age = max(0, int(time.time() - log.stat().st_mtime))
    status = state['status']
    activity = state.get('activity', {})
    labels = {'running': 'RUNNING', 'queued': 'STARTING', 'complete': 'COMPLETE', 'failed': 'STOPPED / FAILED'}
    parts = [labels.get(status, status.upper())]
    if status == 'failed':
        parts.append(state.get('message', 'Process stopped'))
    if status in {'running', 'queued'}:
        if activity.get('pid'):
            parts.append(f"PID {activity['pid']}")
        if activity.get('cpu_seconds') is not None:
            parts.append(f"CPU time {activity['cpu_seconds']:.1f} s")
        parts.append(f'Last output {age} s ago' if age is not None else 'Waiting for output')
    return {'status': status, 'activity': ' | '.join(parts),
            'output': text or 'Waiting for process output…',
            'checked': f"Last checked {datetime.now().astimezone().strftime('%H:%M:%S')} · refreshes every 2 seconds"}


def render_job_terminal(directory, state, log_name):
    st.markdown('**Live terminal**')
    st.components.v2.component('standard_live_job_terminal', js=_JS, css=_CSS)(
        key='standard-live-job-terminal', height=320, width='stretch',
        data=terminal_payload(directory, state, log_name), default={})
