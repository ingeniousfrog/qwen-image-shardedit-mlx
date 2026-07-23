from __future__ import annotations

import json

import pytest

from shardedit_mlx.full_miss_profile import parse_timing_events
from shardedit_mlx.probe_escalation_analysis import (
    EscalationOutcome,
    probe_depths_by_step,
    simulate_escalation,
)


def _timing(event: dict) -> str:
    return "SHARDEDIT_TIMING " + json.dumps(event, sort_keys=True)


def _probe_event(
    *,
    step: int,
    block: int,
    image_relative_l1: float,
    has_previous: bool = True,
) -> dict:
    return {
        "name": "residual_probe",
        "step": step,
        "block": block,
        "has_previous": has_previous,
        "image_relative_l1": image_relative_l1,
        "text_relative_l1": image_relative_l1,
    }


def test_probe_depths_by_step_groups_by_step_and_block() -> None:
    events = parse_timing_events(
        "\n".join(
            (
                _timing(_probe_event(step=2, block=1, image_relative_l1=0.9)),
                _timing(_probe_event(step=2, block=3, image_relative_l1=0.3)),
                _timing(_probe_event(step=3, block=1, image_relative_l1=0.1)),
            )
        )
    )

    steps = probe_depths_by_step(events)

    assert [step.step for step in steps] == [2, 3]
    assert steps[0].relative_l1_by_depth == {1: 0.9, 3: 0.3}
    assert steps[1].relative_l1_by_depth == {1: 0.1}


def test_probe_depths_by_step_ignores_events_without_previous_step() -> None:
    events = parse_timing_events(
        "\n".join(
            (
                _timing(_probe_event(step=1, block=1, image_relative_l1=0.9, has_previous=False)),
                _timing(_probe_event(step=2, block=1, image_relative_l1=0.5)),
            )
        )
    )

    steps = probe_depths_by_step(events)

    assert [step.step for step in steps] == [2]


def test_probe_depths_by_step_rejects_missing_relative_l1() -> None:
    events = ({"name": "residual_probe", "step": 1, "block": 1, "has_previous": True},)

    with pytest.raises(ValueError, match="image_relative_l1"):
        probe_depths_by_step(events)


def test_simulate_escalation_marks_baseline_hit_below_threshold() -> None:
    steps = probe_depths_by_step(
        parse_timing_events(_timing(_probe_event(step=2, block=1, image_relative_l1=0.5)))
    )

    summary = simulate_escalation(steps, threshold=0.8, escalation_depths=(2, 3))

    assert summary.outcomes == (
        EscalationOutcome(
            step=2,
            baseline_depth=1,
            baseline_relative_l1=0.5,
            baseline_hit=True,
            rescue_depth=None,
            rescue_relative_l1=None,
        ),
    )
    assert summary.baseline_misses == ()
    assert summary.rescue_rate == 0.0


def test_simulate_escalation_finds_the_shallowest_rescue_depth() -> None:
    events = parse_timing_events(
        "\n".join(
            (
                _timing(_probe_event(step=2, block=1, image_relative_l1=0.9)),
                _timing(_probe_event(step=2, block=2, image_relative_l1=0.85)),
                _timing(_probe_event(step=2, block=3, image_relative_l1=0.4)),
                _timing(_probe_event(step=2, block=5, image_relative_l1=0.1)),
            )
        )
    )
    steps = probe_depths_by_step(events)

    summary = simulate_escalation(
        steps,
        threshold=0.8,
        escalation_depths=(2, 3, 5),
    )

    outcome = summary.outcomes[0]
    assert outcome.baseline_hit is False
    assert outcome.rescue_depth == 3
    assert outcome.rescue_relative_l1 == pytest.approx(0.4)
    assert summary.rescue_rate == pytest.approx(1.0)


def test_simulate_escalation_reports_no_rescue_when_all_depths_stay_over_threshold() -> None:
    events = parse_timing_events(
        "\n".join(
            (
                _timing(_probe_event(step=4, block=1, image_relative_l1=0.95)),
                _timing(_probe_event(step=4, block=2, image_relative_l1=0.9)),
            )
        )
    )
    steps = probe_depths_by_step(events)

    summary = simulate_escalation(steps, threshold=0.8, escalation_depths=(2,))

    outcome = summary.outcomes[0]
    assert outcome.baseline_hit is False
    assert outcome.rescue_depth is None
    assert outcome.rescue_relative_l1 is None
    assert summary.rescue_rate == 0.0


def test_simulate_escalation_ignores_escalation_depths_at_or_below_baseline() -> None:
    events = parse_timing_events(
        "\n".join(
            (
                _timing(_probe_event(step=1, block=1, image_relative_l1=0.9)),
                _timing(_probe_event(step=1, block=1, image_relative_l1=0.05)),
            )
        )
    )
    steps = probe_depths_by_step(events)

    summary = simulate_escalation(steps, threshold=0.8, baseline_depth=1, escalation_depths=(1,))

    assert summary.escalation_depths == ()
    assert summary.outcomes[0].rescue_depth is None


def test_simulate_escalation_rejects_non_positive_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be positive"):
        simulate_escalation((), threshold=0.0)
