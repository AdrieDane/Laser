from __future__ import annotations

from types import SimpleNamespace
import sys

import pandas as pd

import standard_cli


def test_cli_writes_result_tables_with_path_first(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "output"
    annotations = pd.DataFrame(
        {
            "standard_stream": ["RP_Pos"],
            "feat_feature_id": ["17"],
            "name": ["PC(34:1)"],
            "class": ["PC"],
            "adduct": ["[M+H]+"],
        }
    )
    result = SimpleNamespace(
        tables={
            "automatic_annotations": annotations,
            "policy_audit": pd.DataFrame({"status": ["ok"]}),
        }
    )
    monkeypatch.setattr(standard_cli.standard_annotation_service, "run", lambda request: result)
    monkeypatch.setattr(standard_cli, "discover_standard_streams", lambda path: {})
    monkeypatch.setattr(standard_cli, "write_standard_annotated_inputs", lambda *args: {})
    monkeypatch.setattr(
        sys,
        "argv",
        ["laser-standard", str(project), "--output", str(output)],
    )

    assert standard_cli.main() == 0
    assert pd.read_csv(output / "automatic_annotations.csv").to_dict("records") == [
        {
            "standard_stream": "RP_Pos",
            "feat_feature_id": 17,
            "name": "PC(34:1)",
            "class": "PC",
            "adduct": "[M+H]+",
        }
    ]
    assert pd.read_csv(output / "policy_audit.csv").to_dict("records") == [
        {"status": "ok"}
    ]
