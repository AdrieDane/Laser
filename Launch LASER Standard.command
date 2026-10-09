#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -x "$DIR/.venv/bin/python" ]; then
    python3 -m venv "$DIR/.venv"
    "$DIR/.venv/bin/python" -m pip install --upgrade pip
    "$DIR/.venv/bin/python" -m pip install -e "$DIR"
fi

cd "$DIR"
exec "$DIR/.venv/bin/python" -m streamlit run app/app.py
