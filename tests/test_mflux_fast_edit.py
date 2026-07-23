from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shardedit_mlx.mflux_fast_edit import (
    ResidualCacheState,
    TransformerCallState,
    conditioning_latents_cache_key,
    decide_residual_cache,
    decide_transformer_call,
    file_cache_signature,
    cache_threshold_adjustment,
    flow_aware_cache_threshold,
    format_timing_event,
    infer_prompt_cache_picture_prefix,
    linear_extrapolation_scale,
    normalize_image_paths,
    parse_runtime_options,
    prompt_encoding_cache_key,
    reference_conditioning_dimensions,
    replace_or_add_conditioning_size,
    select_cache_decision_metric,
    select_guided_noise,
    select_middle_anchor_outputs,
    select_predicted_anchor,
    scheduled_cache_threshold,
    should_materialize_block,
    should_materialize_residual_anchor,
    should_flow_veto_cache_hit,
    vae_encode_condition_dimensions,
)


def test_unit_guidance_returns_positive_noise_without_calling_fallback() -> None:
    positive = object()
    negative = object()

    def unexpected_fallback(*_: Any) -> object:
        raise AssertionError("fallback must not run for unit guidance")

    result = select_guided_noise(
        noise=positive,
        noise_negative=negative,
        guidance=1.0,
        fallback=unexpected_fallback,
    )

    assert result is positive


def test_non_unit_guidance_delegates_to_mflux_implementation() -> None:
    positive = object()
    negative = object()
    expected = object()
    calls: list[tuple[object, object, float]] = []

    def fallback(noise: object, noise_negative: object, guidance: float) -> object:
        calls.append((noise, noise_negative, guidance))
        return expected

    result = select_guided_noise(
        noise=positive,
        noise_negative=negative,
        guidance=1.5,
        fallback=fallback,
    )

    assert result is expected
    assert calls == [(positive, negative, 1.5)]


def test_near_but_not_equal_guidance_delegates() -> None:
    calls = 0

    def fallback(*_: Any) -> str:
        nonlocal calls
        calls += 1
        return "fallback"

    result = select_guided_noise(object(), object(), 1.000001, fallback)

    assert result == "fallback"
    assert calls == 1


def test_runtime_options_are_removed_before_upstream_cli_runs() -> None:
    options, remaining = parse_runtime_options(
        [
            "--model",
            "model-path",
            "--shardedit-eval-every-n-blocks",
            "4",
            "--shardedit-probe-blocks",
            "4,1,2,2",
            "--shardedit-profile",
            "--steps",
            "2",
        ]
    )

    assert options.eval_every_n_blocks == 4
    assert options.probe_blocks == (1, 2, 4)
    assert options.profile
    assert remaining == ["--model", "model-path", "--steps", "2"]


def test_token_redundancy_blocks_parse_and_dedupe() -> None:
    options, remaining = parse_runtime_options(
        [
            "--shardedit-token-redundancy-blocks",
            "8,1,1,30",
            "--steps",
            "8",
        ]
    )

    assert options.token_redundancy_blocks == (1, 8, 30)
    assert remaining == ["--steps", "8"]


def test_token_redundancy_blocks_can_run_alongside_cache() -> None:
    options, remaining = parse_runtime_options(
        [
            "--shardedit-token-redundancy-blocks",
            "1,3",
            "--shardedit-cache-threshold",
            "0.8",
        ]
    )

    assert options.token_redundancy_blocks == (1, 3)
    assert options.cache_threshold == 0.8
    assert remaining == []


def test_token_redundancy_heatmap_dir_parses_to_path() -> None:
    options, remaining = parse_runtime_options(
        [
            "--shardedit-token-redundancy-blocks",
            "1,30",
            "--shardedit-token-redundancy-heatmap-dir",
            "/tmp/heatmaps",
            "--steps",
            "8",
        ]
    )

    assert options.token_redundancy_heatmap_dir == Path("/tmp/heatmaps")
    assert remaining == ["--steps", "8"]


def test_bridge_error_diagnose_requires_back_blocks_and_no_cache() -> None:
    with pytest.raises(SystemExit):
        parse_runtime_options(["--shardedit-bridge-error-diagnose"])

    with pytest.raises(SystemExit):
        parse_runtime_options(
            [
                "--shardedit-bridge-error-diagnose",
                "--shardedit-cache-back-blocks",
                "2",
                "--shardedit-cache-threshold",
                "0.8",
            ]
        )

    options, _ = parse_runtime_options(
        [
            "--shardedit-bridge-error-diagnose",
            "--shardedit-cache-back-blocks",
            "2",
            "--shardedit-bridge-error-heatmap-dir",
            "/tmp/bridge-err",
        ]
    )
    assert options.bridge_error_diagnose is True
    assert options.cache_back_blocks == 2
    assert options.bridge_error_heatmap_dir == Path("/tmp/bridge-err")


def test_selective_refill_fraction_requires_cache() -> None:
    with pytest.raises(SystemExit):
        parse_runtime_options(["--shardedit-selective-refill-fraction", "0.15"])

    options, _ = parse_runtime_options(
        [
            "--shardedit-selective-refill-fraction",
            "0.15",
            "--shardedit-cache-threshold",
            "0.8",
            "--shardedit-cache-back-blocks",
            "2",
        ]
    )
    assert options.selective_refill_fraction == pytest.approx(0.15)
    assert options.selective_refill_mode == "residual-dampen"
    assert options.selective_refill_dampen == pytest.approx(1.0)
    assert options.selective_refill_min_step == 0
    assert options.cache_threshold == 0.8
    assert options.cache_back_blocks == 2


def test_selective_refill_1b_options_parse() -> None:
    options, _ = parse_runtime_options(
        [
            "--shardedit-selective-refill-fraction",
            "0.15",
            "--shardedit-selective-refill-mode",
            "residual-dampen",
            "--shardedit-selective-refill-dampen",
            "1.0",
            "--shardedit-selective-refill-min-step",
            "7",
            "--shardedit-cache-threshold",
            "0.8",
            "--shardedit-cache-back-blocks",
            "2",
        ]
    )
    assert options.selective_refill_mode == "residual-dampen"
    assert options.selective_refill_dampen == pytest.approx(1.0)
    assert options.selective_refill_min_step == 7


def test_selective_refill_subset_mode_still_available() -> None:
    options, _ = parse_runtime_options(
        [
            "--shardedit-selective-refill-fraction",
            "0.15",
            "--shardedit-selective-refill-mode",
            "subset",
            "--shardedit-cache-threshold",
            "0.8",
            "--shardedit-cache-back-blocks",
            "2",
        ]
    )
    assert options.selective_refill_mode == "subset"


def test_selective_refill_uniqueness_scale_mode_parses() -> None:
    options, _ = parse_runtime_options(
        [
            "--shardedit-selective-refill-fraction",
            "1.0",
            "--shardedit-selective-refill-mode",
            "uniqueness-scale",
            "--shardedit-selective-refill-dampen",
            "0.5",
            "--shardedit-cache-threshold",
            "0.8",
            "--shardedit-cache-back-blocks",
            "2",
        ]
    )
    assert options.selective_refill_mode == "uniqueness-scale"
    assert options.selective_refill_fraction == pytest.approx(1.0)
    assert options.selective_refill_dampen == pytest.approx(0.5)


def test_selective_refill_boost_and_subset_f1_modes_parse() -> None:
    boost, _ = parse_runtime_options(
        [
            "--shardedit-selective-refill-fraction",
            "0.15",
            "--shardedit-selective-refill-mode",
            "uniqueness-boost",
            "--shardedit-selective-refill-dampen",
            "0.5",
            "--shardedit-cache-threshold",
            "0.8",
            "--shardedit-cache-back-blocks",
            "2",
        ]
    )
    assert boost.selective_refill_mode == "uniqueness-boost"
    subset_f1, _ = parse_runtime_options(
        [
            "--shardedit-selective-refill-fraction",
            "0.15",
            "--shardedit-selective-refill-mode",
            "subset-f1",
            "--shardedit-cache-threshold",
            "0.8",
            "--shardedit-cache-back-blocks",
            "2",
        ]
    )
    assert subset_f1.selective_refill_mode == "subset-f1"


def test_cache_runtime_options_are_removed_before_upstream_cli_runs() -> None:
    options, remaining = parse_runtime_options(
        [
            "--shardedit-cache-threshold",
            "0.8",
            "--shardedit-cache-max-consecutive",
            "2",
            "--shardedit-cache-warmup-steps",
            "1",
            "--shardedit-cache-back-blocks",
            "2",
            "--steps",
            "8",
        ]
    )

    assert options.cache_threshold == 0.8
    assert options.cache_max_consecutive == 2
    assert options.cache_warmup_steps == 1
    assert options.cache_back_blocks == 2
    assert options.cache_anchor_mode == "residual"
    assert remaining == ["--steps", "8"]


def test_cache_anchor_mode_option_is_removed_before_upstream_cli_runs() -> None:
    options, remaining = parse_runtime_options(
        [
            "--shardedit-cache-threshold",
            "0.8",
            "--shardedit-cache-anchor-mode",
            "absolute",
            "--steps",
            "4",
        ]
    )

    assert options.cache_anchor_mode == "absolute"
    assert remaining == ["--steps", "4"]


def test_cache_predictor_and_threshold_schedule_are_removed_before_upstream_cli_runs() -> None:
    options, remaining = parse_runtime_options(
        [
            "--shardedit-cache-predictor",
            "quadratic",
            "--shardedit-cache-threshold-schedule",
            "flow-aware-veto",
            "--shardedit-cache-region-policy",
            "target-conservative",
            "--steps",
            "4",
        ]
    )

    assert options.cache_predictor == "quadratic"
    assert options.cache_threshold_schedule == "flow-aware-veto"
    assert options.cache_region_policy == "target-conservative"
    assert remaining == ["--steps", "4"]


def test_reference_conditioning_size_option_is_removed_before_upstream_cli_runs() -> None:
    options, remaining = parse_runtime_options(
        [
            "--shardedit-reference-conditioning-size",
            "fit-box",
            "--shardedit-reference-conditioning-short-side",
            "640",
            "--shardedit-reference-conditioning-max-width",
            "576",
            "--shardedit-reference-conditioning-max-height",
            "768",
            "--steps",
            "4",
        ]
    )

    assert options.reference_conditioning_size == "fit-box"
    assert options.reference_conditioning_short_side == 640
    assert options.reference_conditioning_max_width == 576
    assert options.reference_conditioning_max_height == 768
    assert remaining == ["--steps", "4"]


def test_fit_box_reference_conditioning_requires_grid_sized_box() -> None:
    with pytest.raises(SystemExit):
        parse_runtime_options(
            [
                "--shardedit-reference-conditioning-size",
                "fit-box",
                "--shardedit-reference-conditioning-max-width",
                "16",
                "--shardedit-reference-conditioning-max-height",
                "768",
            ]
        )


def test_shard_residency_options_are_removed_before_upstream_cli_runs() -> None:
    options, remaining = parse_runtime_options(
        [
            "--shardedit-residency",
            "shard",
            "--shardedit-residency-window-size",
            "4",
            "--steps",
            "8",
        ]
    )

    assert options.residency_mode == "shard"
    assert options.residency_window_size == 4
    assert remaining == ["--steps", "8"]


def test_dense_img_ff_window_option_requires_residency() -> None:
    options, remaining = parse_runtime_options(
        [
            "--shardedit-residency",
            "shard",
            "--shardedit-dense-img-ff-window",
            "--shardedit-dense-img-ff-cache-max-blocks",
            "60",
            "--steps",
            "8",
        ]
    )
    assert options.dense_img_ff_window is True
    assert options.dense_img_ff_cache_max_blocks == 60
    assert remaining == ["--steps", "8"]

    with pytest.raises(SystemExit):
        parse_runtime_options(
            [
                "--shardedit-residency",
                "none",
                "--shardedit-dense-img-ff-window",
            ]
        )


def test_kquant_img_ff_window_option_requires_residency_and_excludes_dense() -> None:
    options, remaining = parse_runtime_options(
        [
            "--shardedit-residency",
            "shard",
            "--shardedit-kquant-img-ff-window",
            "--shardedit-kquant-img-ff-cache-max-blocks",
            "60",
            "--shardedit-kquant-img-ff-codec",
            "q6_k",
            "--steps",
            "8",
        ]
    )
    assert options.kquant_img_ff_window is True
    assert options.kquant_img_ff_cache_max_blocks == 60
    assert options.kquant_img_ff_codec == "q6_k"
    assert remaining == ["--steps", "8"]

    with pytest.raises(SystemExit):
        parse_runtime_options(
            [
                "--shardedit-residency",
                "none",
                "--shardedit-kquant-img-ff-window",
            ]
        )

    with pytest.raises(SystemExit):
        parse_runtime_options(
            [
                "--shardedit-residency",
                "shard",
                "--shardedit-dense-img-ff-window",
                "--shardedit-kquant-img-ff-window",
            ]
        )


def test_lora_and_patched_window_cache_options_require_residency() -> None:
    options, remaining = parse_runtime_options(
        [
            "--shardedit-residency",
            "shard",
            "--shardedit-lora-tensor-cache",
            "--shardedit-lora-tensor-cache-max-windows",
            "8",
            "--shardedit-patched-window-cache-max-windows",
            "2",
            "--steps",
            "8",
        ]
    )

    assert options.lora_tensor_cache is True
    assert options.lora_tensor_cache_max_windows == 8
    assert options.patched_window_cache_max_windows == 2
    assert remaining == ["--steps", "8"]

    with pytest.raises(SystemExit):
        parse_runtime_options(
            [
                "--shardedit-residency",
                "none",
                "--shardedit-lora-tensor-cache",
            ]
        )

    with pytest.raises(SystemExit):
        parse_runtime_options(
            [
                "--shardedit-residency",
                "none",
                "--shardedit-patched-window-cache-max-windows",
                "1",
            ]
        )


def test_release_policy_option_parses() -> None:
    options, remaining = parse_runtime_options(
        [
            "--shardedit-residency",
            "shard",
            "--shardedit-release-policy",
            "step",
            "--steps",
            "8",
        ]
    )

    assert options.release_policy == "step"
    assert remaining == ["--steps", "8"]

    options, remaining = parse_runtime_options(
        [
            "--shardedit-residency",
            "shard",
            "--shardedit-release-policy",
            "none",
            "--steps",
            "8",
        ]
    )

    assert options.release_policy == "none"
    assert remaining == ["--steps", "8"]

    options, remaining = parse_runtime_options(
        [
            "--shardedit-residency",
            "shard",
            "--shardedit-release-policy",
            "keep-last",
            "--steps",
            "8",
        ]
    )

    assert options.release_policy == "keep-last"
    assert remaining == ["--steps", "8"]


def test_q6_linear_profile_option_implies_timing_profile() -> None:
    options, remaining = parse_runtime_options(
        [
            "--shardedit-q6-linear-profile",
            "--steps",
            "8",
        ]
    )

    assert options.q6_linear_profile is True
    assert options.profile is True
    assert remaining == ["--steps", "8"]


def test_condition_token_merge_options_are_removed_before_upstream_cli_runs() -> None:
    options, remaining = parse_runtime_options(
        [
            "--shardedit-condition-token-merge",
            "--shardedit-condition-token-merge-stride",
            "2",
            "--shardedit-condition-token-merge-start-block",
            "2",
            "--shardedit-condition-token-merge-back-blocks",
            "2",
            "--steps",
            "8",
        ]
    )

    assert options.condition_token_merge is True
    assert options.condition_token_merge_stride == 2
    assert options.condition_token_merge_start_block == 2
    assert options.condition_token_merge_back_blocks == 2
    assert remaining == ["--steps", "8"]


def test_text_token_merge_options_are_removed_before_upstream_cli_runs() -> None:
    options, remaining = parse_runtime_options(
        [
            "--shardedit-text-token-merge",
            "--shardedit-text-token-merge-stride",
            "2",
            "--shardedit-text-token-merge-start-block",
            "2",
            "--shardedit-text-token-merge-back-blocks",
            "2",
            "--steps",
            "8",
        ]
    )

    assert options.text_token_merge is True
    assert options.text_token_merge_stride == 2
    assert options.text_token_merge_start_block == 2
    assert options.text_token_merge_back_blocks == 2
    assert remaining == ["--steps", "8"]


def test_condition_token_merge_rejects_invalid_geometry_options() -> None:
    with pytest.raises(SystemExit):
        parse_runtime_options(
            [
                "--shardedit-condition-token-merge",
                "--shardedit-condition-token-merge-stride",
                "1",
            ]
        )

    with pytest.raises(SystemExit):
        parse_runtime_options(
            [
                "--shardedit-condition-token-merge",
                "--shardedit-condition-token-merge-start-block",
                "0",
            ]
        )


def test_text_token_merge_rejects_invalid_geometry_options() -> None:
    with pytest.raises(SystemExit):
        parse_runtime_options(
            [
                "--shardedit-text-token-merge",
                "--shardedit-text-token-merge-stride",
                "1",
            ]
        )

    with pytest.raises(SystemExit):
        parse_runtime_options(
            [
                "--shardedit-text-token-merge",
                "--shardedit-text-token-merge-start-block",
                "0",
            ]
        )


def test_residency_and_residual_cache_can_run_together() -> None:
    options, remaining = parse_runtime_options(
        [
            "--shardedit-residency",
            "shard",
            "--shardedit-cache-threshold",
            "0.8",
            "--steps",
            "8",
        ]
    )

    assert options.residency_mode == "shard"
    assert options.cache_threshold == 0.8
    assert remaining == ["--steps", "8"]


def test_residual_probe_and_cache_cannot_run_together() -> None:
    with pytest.raises(SystemExit):
        parse_runtime_options(
            [
                "--shardedit-probe-blocks",
                "1,2,4",
                "--shardedit-cache-threshold",
                "0.8",
            ]
        )


def test_runtime_options_default_to_shard_residency() -> None:
    options, remaining = parse_runtime_options(["--steps", "8"])

    assert options.eval_every_n_blocks == 0
    assert options.probe_blocks == ()
    assert options.token_redundancy_blocks == ()
    assert options.token_redundancy_heatmap_dir is None
    assert options.bridge_error_diagnose is False
    assert options.bridge_error_heatmap_dir is None
    assert options.selective_refill_fraction == 0.0
    assert options.cache_threshold == 0.0
    assert options.cache_max_consecutive == 1
    assert options.cache_warmup_steps == 1
    assert options.cache_back_blocks == 0
    assert options.cache_predictor == "last"
    assert options.cache_threshold_schedule == "fixed"
    assert options.cache_region_policy == "all"
    assert options.reference_conditioning_size == "upstream"
    assert options.reference_conditioning_short_side == 512
    assert options.reference_conditioning_max_width == 576
    assert options.reference_conditioning_max_height == 768
    assert options.residency_mode == "shard"
    assert options.residency_window_size == 8
    assert options.release_policy == "window"
    assert options.dense_img_ff_cache_max_blocks == 60
    assert options.kquant_img_ff_window is False
    assert options.kquant_img_ff_cache_max_blocks == 60
    assert options.kquant_img_ff_codec == "q6_k"
    assert options.lora_tensor_cache is False
    assert options.lora_tensor_cache_max_windows == 8
    assert options.patched_window_cache_max_windows == 0
    assert options.condition_token_merge is False
    assert options.condition_token_merge_stride == 2
    assert options.condition_token_merge_start_block == 2
    assert options.condition_token_merge_back_blocks == 2
    assert options.text_token_merge is False
    assert options.text_token_merge_stride == 2
    assert options.text_token_merge_start_block == 2
    assert options.text_token_merge_back_blocks == 2
    assert not options.profile
    assert options.q6_linear_profile is False
    assert remaining == ["--steps", "8"]


def test_runtime_options_can_disable_default_residency() -> None:
    options, remaining = parse_runtime_options(
        ["--shardedit-residency", "none", "--steps", "8"]
    )

    assert options.residency_mode == "none"
    assert remaining == ["--steps", "8"]


def test_image_path_normalization_accepts_one_or_many_paths() -> None:
    assert normalize_image_paths("one.png") == ("one.png",)
    assert normalize_image_paths(["one.png", "two.png"]) == ("one.png", "two.png")


def test_prompt_cache_picture_prefix_can_be_inferred_after_tokenizer_release() -> None:
    assert not infer_prompt_cache_picture_prefix(("one.png",), tokenizer_use_picture_prefix=None)
    assert infer_prompt_cache_picture_prefix(("one.png", "two.png"), tokenizer_use_picture_prefix=None)
    assert infer_prompt_cache_picture_prefix(("one.png",), tokenizer_use_picture_prefix=True)
    assert infer_prompt_cache_picture_prefix(("one.png", "two.png"), tokenizer_use_picture_prefix=False)


def test_file_cache_signature_changes_when_file_content_changes(tmp_path) -> None:
    image = tmp_path / "ref.png"
    image.write_bytes(b"first")
    first = file_cache_signature(str(image))

    image.write_bytes(b"second-version")
    second = file_cache_signature(str(image))

    assert first != second
    assert first.path == second.path == str(image.resolve())


def test_prompt_cache_key_includes_prompt_dimensions_and_image_signature(tmp_path) -> None:
    image = tmp_path / "ref.png"
    image.write_bytes(b"pixels")

    first = prompt_encoding_cache_key(
        prompt="coffee shop",
        negative_prompt=None,
        image_paths=[str(image)],
        vl_width=320,
        vl_height=448,
        guidance=1.0,
        use_picture_prefix=False,
    )
    different_prompt = prompt_encoding_cache_key(
        prompt="office",
        negative_prompt=None,
        image_paths=[str(image)],
        vl_width=320,
        vl_height=448,
        guidance=1.0,
        use_picture_prefix=False,
    )
    different_size = prompt_encoding_cache_key(
        prompt="coffee shop",
        negative_prompt=None,
        image_paths=[str(image)],
        vl_width=384,
        vl_height=384,
        guidance=1.0,
        use_picture_prefix=False,
    )

    assert first != different_prompt
    assert first != different_size


def test_conditioning_cache_key_invalidates_on_reference_change(tmp_path) -> None:
    image = tmp_path / "ref.png"
    image.write_bytes(b"pixels")
    first = conditioning_latents_cache_key(
        image_paths=str(image),
        height=1184,
        width=896,
        vl_width=320,
        vl_height=448,
    )

    image.write_bytes(b"changed-pixels")
    second = conditioning_latents_cache_key(
        image_paths=str(image),
        height=1184,
        width=896,
        vl_width=320,
        vl_height=448,
    )

    assert first != second


def test_materialization_interval_uses_one_based_block_boundaries() -> None:
    assert not should_materialize_block(block_index=0, every_n_blocks=4)
    assert should_materialize_block(block_index=3, every_n_blocks=4)
    assert should_materialize_block(block_index=7, every_n_blocks=4)
    assert not should_materialize_block(block_index=8, every_n_blocks=4)
    assert not should_materialize_block(block_index=3, every_n_blocks=0)


def test_select_middle_anchor_outputs_uses_residual_delta_on_first_skipped_block() -> None:
    assert select_middle_anchor_outputs(
        anchor_mode="residual",
        block_index=1,
        encoder_input=10,
        hidden_input=100,
        cached_middle_encoder_anchor=3,
        cached_middle_hidden_anchor=7,
    ) == (13, 107)
    assert select_middle_anchor_outputs(
        anchor_mode="residual",
        block_index=2,
        encoder_input=10,
        hidden_input=100,
        cached_middle_encoder_anchor=3,
        cached_middle_hidden_anchor=7,
    ) == (10, 100)


def test_select_middle_anchor_outputs_can_jump_to_absolute_middle_state() -> None:
    assert select_middle_anchor_outputs(
        anchor_mode="absolute",
        block_index=1,
        encoder_input=10,
        hidden_input=100,
        cached_middle_encoder_anchor=30,
        cached_middle_hidden_anchor=70,
    ) == (30, 70)
    assert select_middle_anchor_outputs(
        anchor_mode="absolute",
        block_index=2,
        encoder_input=10,
        hidden_input=100,
        cached_middle_encoder_anchor=30,
        cached_middle_hidden_anchor=70,
    ) == (10, 100)


def test_linear_residual_prediction_extrapolates_from_two_anchors() -> None:
    assert linear_extrapolation_scale(
        previous_coordinate=1.0,
        anchor_coordinate=3.0,
        current_coordinate=4.0,
    ) == pytest.approx(0.5)
    prediction = select_predicted_anchor(
        predictor="linear-residual",
        cached_anchor=20.0,
        previous_anchor=12.0,
        previous_coordinate=1.0,
        anchor_coordinate=3.0,
        current_coordinate=4.0,
    )

    assert prediction.scale == pytest.approx(0.5)
    assert prediction.order == 1
    assert prediction.method == "linear"
    assert prediction.value == pytest.approx(24.0)


def test_linear_residual_prediction_falls_back_without_two_anchors() -> None:
    prediction = select_predicted_anchor(
        predictor="linear-residual",
        cached_anchor=20.0,
        previous_anchor=None,
        previous_coordinate=None,
        anchor_coordinate=3.0,
        current_coordinate=4.0,
    )

    assert prediction.value == 20.0
    assert prediction.scale is None
    assert prediction.fallback_reason == "insufficient_history"


def test_quadratic_prediction_extrapolates_from_three_anchors() -> None:
    prediction = select_predicted_anchor(
        predictor="quadratic",
        older_anchor=1.0,
        previous_anchor=4.0,
        cached_anchor=9.0,
        older_coordinate=1.0,
        previous_coordinate=2.0,
        anchor_coordinate=3.0,
        current_coordinate=4.0,
    )

    assert prediction.value == pytest.approx(16.0)
    assert prediction.order == 2
    assert prediction.method == "quadratic"
    assert prediction.fallback_reason is None


def test_adams_bashforth_prediction_extrapolates_from_three_anchors() -> None:
    prediction = select_predicted_anchor(
        predictor="adams-bashforth",
        older_anchor=1.0,
        previous_anchor=2.0,
        cached_anchor=4.0,
        older_coordinate=0.0,
        previous_coordinate=1.0,
        anchor_coordinate=2.0,
        current_coordinate=3.0,
    )

    assert prediction.value == pytest.approx(6.5)
    assert prediction.order == 2
    assert prediction.method == "adams-bashforth"


def test_quadratic_prediction_falls_back_to_linear_without_three_anchors() -> None:
    prediction = select_predicted_anchor(
        predictor="quadratic",
        previous_anchor=12.0,
        cached_anchor=20.0,
        previous_coordinate=1.0,
        anchor_coordinate=3.0,
        current_coordinate=4.0,
    )

    assert prediction.value == pytest.approx(24.0)
    assert prediction.order == 1
    assert prediction.fallback_reason == "linear_fallback"


def test_vae_encode_condition_dimensions_reads_positional_arguments() -> None:
    height, width = vae_encode_condition_dimensions(
        ("vae", "reference.png", 1248, 832),
        {},
    )

    assert height == 1248
    assert width == 832


def test_vae_encode_condition_dimensions_reads_keyword_arguments() -> None:
    height, width = vae_encode_condition_dimensions(
        (),
        {"vae": "vae", "image_path": "reference.png", "height": 768, "width": 512},
    )

    assert height == 768
    assert width == 512


def test_reference_conditioning_dimensions_can_use_original_size() -> None:
    assert reference_conditioning_dimensions(
        policy="original",
        image_width=576,
        image_height=768,
    ) == (576, 768)


def test_reference_conditioning_dimensions_can_scale_short_side_to_512() -> None:
    assert reference_conditioning_dimensions(
        policy="short-side-512",
        image_width=576,
        image_height=768,
    ) == (512, 672)


def test_reference_conditioning_dimensions_can_scale_to_custom_short_side() -> None:
    assert reference_conditioning_dimensions(
        policy="short-side",
        image_width=576,
        image_height=768,
        short_side=640,
    ) == (640, 864)


def test_reference_conditioning_dimensions_can_fit_three_by_four_reference_to_box() -> None:
    assert reference_conditioning_dimensions(
        policy="fit-box",
        image_width=1080,
        image_height=1440,
        max_width=576,
        max_height=768,
    ) == (576, 768)


def test_reference_conditioning_dimensions_can_fit_tall_reference_to_box() -> None:
    assert reference_conditioning_dimensions(
        policy="fit-box",
        image_width=1200,
        image_height=2596,
        max_width=576,
        max_height=768,
    ) == (352, 768)


def test_reference_conditioning_dimensions_does_not_upscale_small_reference_for_fit_box() -> None:
    assert reference_conditioning_dimensions(
        policy="fit-box",
        image_width=320,
        image_height=448,
        max_width=576,
        max_height=768,
    ) == (320, 448)


def test_reference_conditioning_dimensions_keep_upstream_default() -> None:
    assert (
        reference_conditioning_dimensions(
            policy="upstream",
            image_width=576,
            image_height=768,
        )
        is None
    )


def test_replace_or_add_conditioning_size_updates_positional_arguments() -> None:
    args, kwargs = replace_or_add_conditioning_size(
        ("vae", 1248, 832, ["reference.png"], 320, 480),
        {},
        width=512,
        height=768,
    )

    assert args == ("vae", 1248, 832, ["reference.png"], 512, 768)
    assert kwargs == {}


def test_replace_or_add_conditioning_size_updates_keyword_arguments() -> None:
    args, kwargs = replace_or_add_conditioning_size(
        (),
        {"vae": "vae", "height": 1248, "width": 832},
        width=512,
        height=768,
    )

    assert args == ()
    assert kwargs["vl_width"] == 512
    assert kwargs["vl_height"] == 768


def test_sigma_threshold_schedule_is_stricter_at_early_high_sigma_steps() -> None:
    early, early_progress, early_coordinate = scheduled_cache_threshold(
        0.8,
        "sigma",
        step=1,
        total_steps=8,
        current_sigma=1.0,
        first_sigma=1.0,
        final_sigma=0.1,
    )
    late, late_progress, late_coordinate = scheduled_cache_threshold(
        0.8,
        "sigma",
        step=8,
        total_steps=8,
        current_sigma=0.1,
        first_sigma=1.0,
        final_sigma=0.1,
    )

    assert early == pytest.approx(0.52)
    assert late == pytest.approx(0.8)
    assert early < late
    assert early_progress == pytest.approx(0.0)
    assert late_progress == pytest.approx(1.0)
    assert early_coordinate == late_coordinate == "sigma"


def test_sigma_threshold_schedule_falls_back_to_step_progress() -> None:
    threshold, progress, coordinate = scheduled_cache_threshold(
        0.8,
        "sigma",
        step=4,
        total_steps=8,
    )

    assert threshold == pytest.approx(0.8 * (0.65 + 0.35 * (3 / 7)))
    assert progress == pytest.approx(3 / 7)
    assert coordinate == "step"


def test_flow_aware_threshold_gets_stricter_when_prediction_signals_are_bad() -> None:
    baseline, _, _ = scheduled_cache_threshold(
        0.8,
        "sigma",
        step=4,
        total_steps=8,
        current_sigma=0.55,
        first_sigma=1.0,
        final_sigma=0.1,
    )
    adjusted = flow_aware_cache_threshold(
        0.8,
        step=4,
        total_steps=8,
        current_sigma=0.55,
        first_sigma=1.0,
        final_sigma=0.1,
        prediction_cosine=0.82,
        magnitude_ratio=1.8,
        history_relative_l1=0.9,
    )

    assert adjusted.value < baseline
    assert adjusted.cosine_factor == pytest.approx(0.55)
    assert adjusted.magnitude_factor is not None
    assert adjusted.history_factor is not None


def test_flow_aware_veto_schedule_keeps_fixed_threshold_with_veto_threshold() -> None:
    adjusted = cache_threshold_adjustment(
        0.8,
        "flow-aware-veto",
        step=4,
        total_steps=8,
        current_sigma=0.55,
        first_sigma=1.0,
        final_sigma=0.1,
        prediction_cosine=0.82,
        magnitude_ratio=1.8,
        history_relative_l1=0.9,
    )

    assert adjusted.value == pytest.approx(0.8)
    assert adjusted.veto_threshold is not None
    assert adjusted.veto_threshold < adjusted.value
    assert adjusted.cosine_factor == pytest.approx(0.55)


def test_flow_aware_veto_preserves_safe_fixed_f1b2_hits() -> None:
    vetoed, reason = should_flow_veto_cache_hit(
        schedule="flow-aware-veto",
        cache_hit=True,
        relative_l1=0.37,
        base_threshold=0.8,
        veto_threshold=0.24,
        prediction_cosine=0.99,
        magnitude_ratio=1.35,
    )

    assert not vetoed
    assert reason is None


def test_flow_aware_veto_blocks_high_risk_fixed_f1b2_hits() -> None:
    vetoed, reason = should_flow_veto_cache_hit(
        schedule="flow-aware-veto",
        cache_hit=True,
        relative_l1=0.72,
        base_threshold=0.8,
        veto_threshold=0.28,
        prediction_cosine=0.88,
        magnitude_ratio=2.2,
    )

    assert vetoed
    assert reason == "flow_veto_boundary"


def test_region_policy_can_use_target_or_condition_relative_l1() -> None:
    assert (
        select_cache_decision_metric(
            policy="all",
            global_relative_l1=0.2,
            target_relative_l1=0.7,
            condition_relative_l1=0.9,
        )
        == 0.2
    )
    assert (
        select_cache_decision_metric(
            policy="target-conservative",
            global_relative_l1=0.2,
            target_relative_l1=0.7,
            condition_relative_l1=0.9,
        )
        == 0.7
    )
    assert (
        select_cache_decision_metric(
            policy="condition-conservative",
            global_relative_l1=0.2,
            target_relative_l1=0.7,
            condition_relative_l1=0.9,
        )
        == 0.9
    )


def test_residual_anchor_materialization_only_runs_on_full_cache_misses() -> None:
    assert should_materialize_residual_anchor(
        block_index=0,
        cache_enabled=True,
        cache_hit=False,
        middle_end_index=57,
    )
    assert should_materialize_residual_anchor(
        block_index=57,
        cache_enabled=True,
        cache_hit=False,
        middle_end_index=57,
    )
    assert not should_materialize_residual_anchor(
        block_index=58,
        cache_enabled=True,
        cache_hit=False,
        middle_end_index=57,
    )
    assert not should_materialize_residual_anchor(
        block_index=0,
        cache_enabled=True,
        cache_hit=True,
        middle_end_index=57,
    )
    assert not should_materialize_residual_anchor(
        block_index=0,
        cache_enabled=False,
        cache_hit=False,
        middle_end_index=57,
    )
    assert not should_materialize_residual_anchor(
        block_index=0,
        cache_enabled=True,
        cache_hit=False,
        middle_end_index=None,
    )


def test_unit_guidance_runs_positive_call_and_skips_matching_negative_call() -> None:
    initial = TransformerCallState()

    run_positive, after_positive = decide_transformer_call(initial, (10, 0), unit_guidance=True)
    run_negative, after_negative = decide_transformer_call(after_positive, (10, 0), unit_guidance=True)
    run_next_step, _ = decide_transformer_call(after_negative, (10, 1), unit_guidance=True)

    assert run_positive
    assert not run_negative
    assert run_next_step


def test_non_unit_guidance_never_skips_transformer_calls() -> None:
    initial = TransformerCallState()

    run_first, after_first = decide_transformer_call(initial, (10, 0), unit_guidance=False)
    run_second, _ = decide_transformer_call(after_first, (10, 0), unit_guidance=False)

    assert run_first
    assert run_second


def test_residual_cache_state_machine_forces_refresh_after_one_hit() -> None:
    state = ResidualCacheState()

    hit, state, reason = decide_residual_cache(
        state,
        step=1,
        warmup_steps=1,
        threshold=0.8,
        max_consecutive=1,
        relative_l1=None,
    )
    assert not hit
    assert state == ResidualCacheState(has_anchor=True, consecutive_hits=0)
    assert reason == "warmup"

    hit, state, reason = decide_residual_cache(
        state,
        step=2,
        warmup_steps=1,
        threshold=0.8,
        max_consecutive=1,
        relative_l1=1.65,
    )
    assert not hit
    assert reason == "diff_miss"

    hit, state, reason = decide_residual_cache(
        state,
        step=3,
        warmup_steps=1,
        threshold=0.8,
        max_consecutive=1,
        relative_l1=0.59,
    )
    assert hit
    assert state.consecutive_hits == 1
    assert reason == "diff_hit"

    hit, state, reason = decide_residual_cache(
        state,
        step=4,
        warmup_steps=1,
        threshold=0.8,
        max_consecutive=1,
        relative_l1=None,
    )
    assert not hit
    assert state.consecutive_hits == 0
    assert reason == "max_consecutive"


def test_timing_event_is_machine_readable_and_stable() -> None:
    assert format_timing_event("denoise_transformer", 1.23456789, step=2) == (
        'SHARDEDIT_TIMING {"name": "denoise_transformer", '
        '"seconds": 1.234568, "step": 2}'
    )
