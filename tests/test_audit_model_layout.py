from __future__ import annotations

from pathlib import Path
import runpy


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "audit_model_layout.py"
AUDIT_MODULE = runpy.run_path(str(SCRIPT))


def test_default_model_targets_qwen_image_edit_2511() -> None:
    assert AUDIT_MODULE["DEFAULT_MODEL"].name == "qwen-edit-2511-q6"


def test_mflux_configuration_metadata_is_optional() -> None:
    requirement = next(
        item for item in AUDIT_MODULE["REQUIREMENTS"] if item.path == "configuration.json"
    )

    assert requirement.runtime == "mflux"
    assert requirement.required is False
