import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

HELPER = (
    Path(__file__).resolve().parents[2] / "tools/qe_optimization/frozen_resources.py"
)
spec = importlib.util.spec_from_file_location("frozen_resource_test", HELPER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def snapshot(tmp_path):
    root = tmp_path / "snapshot"
    (root / "app").mkdir(parents=True)
    (root / "reference_data").mkdir()
    (root / "methods").mkdir()
    (root / "reference_data/standard_lipid_composition_ranges.tsv").write_text(
        "fixture", encoding="utf-8"
    )
    shutil.copy2(
        Path(__file__).resolve().parents[1] / "resource_paths.py",
        root / "app/resource_paths.py",
    )
    return root


def test_correct_frozen_paths_are_verified_but_installed_fallback_is_rejected(tmp_path):
    root = snapshot(tmp_path)
    result = module.verify_frozen_paths(root, root / "reference_data", root / "methods")
    assert len(result["composition_ranges_sha256"]) == 64
    with pytest.raises(RuntimeError, match="escaped"):
        module.verify_frozen_paths(root, tmp_path / "installed", root / "methods")


def test_missing_bundled_assets_fail_before_module_import(tmp_path):
    root = tmp_path / "incomplete"
    (root / "app").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="missing"):
        module.configure_frozen_resources(root)


def test_fresh_interpreter_uses_only_explicit_snapshot_assets(tmp_path):
    root = snapshot(tmp_path)
    code = (
        "import importlib.util; s=importlib.util.spec_from_file_location('frozen',%r); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); r=m.configure_frozen_resources(%r); assert r['reference_data']==%r; print('frozen paths verified')"
        % (str(HELPER), str(root), str(root / "reference_data"))
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert "frozen paths verified" in result.stdout
