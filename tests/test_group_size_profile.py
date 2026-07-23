from __future__ import annotations

from shardedit_mlx.group_size_profile import decide_group_size_verdict


def test_decide_group_size_verdict_finds_candidates() -> None:
    verdict, _, candidates = decide_group_size_verdict(
        baseline_group_size=64,
        baseline_median=1.0,
        candidates={32: 0.90, 64: 1.0, 128: 0.99},
        speedup_threshold=0.05,
    )
    assert verdict == "group_size_worth_quality_gate"
    assert candidates == (32,)


def test_decide_group_size_verdict_no_speedup() -> None:
    verdict, _, candidates = decide_group_size_verdict(
        baseline_group_size=64,
        baseline_median=1.0,
        candidates={32: 1.02, 64: 1.0, 128: 0.98},
        speedup_threshold=0.05,
    )
    assert verdict == "group_size_no_speedup"
    assert candidates == ()
