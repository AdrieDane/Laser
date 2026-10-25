import json

import pandas as pd
import pytest

from publish_qe_aggregate_report import copy_aggregate, public_json, publish


def test_unknown_columns_and_partial_collection_cannot_be_published(tmp_path):
    source = tmp_path / "source.csv"
    destination = tmp_path / "public.csv"
    pd.DataFrame({"feat_feature_id": ["private"]}).to_csv(source, index=False)
    with pytest.raises(ValueError, match="Unapproved"):
        copy_aggregate(source, destination)
    assert not destination.exists()
    pd.DataFrame({"dataset": ["one"], "stream": ["RP_Pos"]}).to_csv(source, index=False)
    with pytest.raises(ValueError, match="Full33dataset"):
        copy_aggregate(source, destination)
    assert not destination.exists()


def test_private_inputs_and_absolute_paths_removed():
    record = {
        "input_sha256": {"/private/feature": "hash"},
        "before_checkpoint": "/private/experiment_029",
        "nested": {"resource_receipt": {"path": "/private"}, "count": 33},
    }
    assert public_json(record) == {
        "before_checkpoint": "experiment_029",
        "nested": {"count": 33},
    }


def test_no_publication_before_complete_integrity(tmp_path):
    experiment = tmp_path / "private"
    (experiment / "reports_v2").mkdir(parents=True)
    (experiment / "reports_v2/summary.json").write_text(
        json.dumps({"complete": True, "datasets_evaluated": 33})
    )
    (experiment / "integrity_receipt.json").write_text(
        json.dumps({"all_passed": False})
    )
    destination = tmp_path / "public"
    with pytest.raises(ValueError, match="passing integrity"):
        publish(experiment, tmp_path / "before", destination, "rejected")
    assert not destination.exists()
