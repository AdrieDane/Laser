import json

from verify_qe_empirical_preparation_resumable import completed_case, digest_manifest


def test_missing_partial_stale_and_wrong_scope_receipts_do_not_resume(tmp_path):
    path = tmp_path / "receipt.json"
    assert not completed_case(path, "dataset", "hash")
    path.write_text('{"all_passed":')
    assert not completed_case(path, "dataset", "hash")
    good = dict(
        all_passed=True,
        dataset="dataset",
        verification_identity_sha256="hash",
        full_masked_tables=11,
        fresh_standard_mask_checks=11,
    )
    path.write_text(json.dumps(good))
    assert completed_case(path, "dataset", "hash")
    assert not completed_case(path, "other_dataset", "hash")
    assert not completed_case(path, "dataset", "changed_hash")
    for key, value in (
        ("all_passed", False),
        ("full_masked_tables", 10),
        ("fresh_standard_mask_checks", 10),
    ):
        path.write_text(json.dumps(dict(good, **{key: value})))
        assert not completed_case(path, "dataset", "hash")


def test_manifest_order_irrelevant_but_every_input_change_invalidates_receipts():
    assert digest_manifest({"a": "1", "b": "2"}) == digest_manifest(
        {"b": "2", "a": "1"}
    )
    assert digest_manifest({"a": "1", "b": "2"}) != digest_manifest(
        {"a": "1", "b": "3"}
    )
