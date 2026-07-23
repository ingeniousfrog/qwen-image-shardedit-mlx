from __future__ import annotations

from shardedit_mlx.mlp_compile_profile import decide_mlp_compile_verdict


def test_decide_mlp_compile_verdict_helps() -> None:
    verdict, _ = decide_mlp_compile_verdict(
        eager_median=1.0,
        compiled_median=0.90,
        max_abs_error=0.0,
        all_finite=True,
        speedup_threshold=0.05,
    )
    assert verdict == "compile_helps"


def test_decide_mlp_compile_verdict_not_enough() -> None:
    verdict, _ = decide_mlp_compile_verdict(
        eager_median=1.0,
        compiled_median=0.98,
        max_abs_error=0.0,
        all_finite=True,
        speedup_threshold=0.05,
    )
    assert verdict == "compile_not_enough"


def test_decide_mlp_compile_verdict_rejects_error() -> None:
    verdict, _ = decide_mlp_compile_verdict(
        eager_median=1.0,
        compiled_median=0.50,
        max_abs_error=1e-3,
        all_finite=True,
        speedup_threshold=0.05,
        error_tolerance=0.0,
    )
    assert verdict == "invalid_compiled_output"
