from __future__ import annotations

import io
import json
from contextlib import AbstractContextManager
from typing import Any

import pytest

from shardedit_mlx.mflux_fast_edit import RuntimeOptions
from shardedit_mlx.warm_worker import (
    WarmEditRequest,
    WarmQwenEditWorker,
    WarmWorkerConfig,
    WarmWorkerRequestError,
    _build_parser,
    _runtime_options_from_args,
    main,
    parse_request,
    response_json,
)


class FakeRuntimeContext(AbstractContextManager[None]):
    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    def __enter__(self) -> None:
        self.entered = True

    def __exit__(self, *_: object) -> None:
        self.exited = True


class FakeGeneratedImage:
    def __init__(self) -> None:
        self.saved: list[dict[str, object]] = []

    def save(self, *, path: str, export_json_metadata: bool, overwrite: bool) -> None:
        self.saved.append(
            {
                "path": path,
                "export_json_metadata": export_json_metadata,
                "overwrite": overwrite,
            }
        )


class FakeCallbackRegistry:
    def __init__(self) -> None:
        self.registered: list[object] = []

    def register(self, callback: object) -> None:
        self.registered.append(callback)


class FakeQwenImageEdit:
    instances: list["FakeQwenImageEdit"] = []

    def __init__(
        self,
        *,
        quantize: int | None,
        model_path: str,
        lora_paths: list[str] | None,
        lora_scales: list[float] | None,
    ) -> None:
        self.init_args = {
            "quantize": quantize,
            "model_path": model_path,
            "lora_paths": lora_paths,
            "lora_scales": lora_scales,
        }
        self.callbacks = FakeCallbackRegistry()
        self.generated: list[dict[str, object]] = []
        self.image = FakeGeneratedImage()
        FakeQwenImageEdit.instances.append(self)

    def generate_image(self, **kwargs: object) -> FakeGeneratedImage:
        self.generated.append(kwargs)
        return self.image


def test_parse_request_accepts_defaults_and_one_image_path() -> None:
    request = parse_request(
        {
            "id": "first",
            "prompt": "move the portrait into a cafe",
            "image_path": "portrait.png",
            "output": "edited.png",
        }
    )

    assert request == WarmEditRequest(
        id="first",
        prompt="move the portrait into a cafe",
        image_paths=("portrait.png",),
        output="edited.png",
    )


def test_parse_request_normalizes_many_image_paths_and_options() -> None:
    request = parse_request(
        {
            "prompt": "combine references",
            "image_paths": ["one.png", "two.png"],
            "output": "edited.png",
            "seed": 7,
            "width": 512,
            "height": 640,
            "steps": 4,
            "guidance": 1.5,
            "negative_prompt": "blur",
            "scheduler": "linear",
            "metadata": True,
            "overwrite": True,
        }
    )

    assert request.image_paths == ("one.png", "two.png")
    assert request.seed == 7
    assert request.width == 512
    assert request.height == 640
    assert request.steps == 4
    assert request.guidance == 1.5
    assert request.negative_prompt == "blur"
    assert request.scheduler == "linear"
    assert request.metadata
    assert request.overwrite


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({}, "prompt is required"),
        ({"prompt": "", "image_path": "a.png", "output": "out.png"}, "prompt cannot be empty"),
        ({"prompt": "x", "image_paths": [], "output": "out.png"}, "image_paths cannot be empty"),
        ({"prompt": "x", "image_path": "a.png"}, "output is required"),
        ({"prompt": "x", "image_path": "a.png", "output": "out.png", "steps": 0}, "steps must be >= 1"),
        ({"prompt": "x", "image_path": "a.png", "output": "out.png", "metadata": "yes"}, "metadata must be a boolean"),
    ],
)
def test_parse_request_reports_invalid_requests(raw: dict[str, object], message: str) -> None:
    with pytest.raises(WarmWorkerRequestError, match=message):
        parse_request(raw)


def test_worker_config_defaults_to_shard_runtime_options() -> None:
    config = WarmWorkerConfig(model_path="model")

    assert config.runtime_options.residency_mode == "shard"
    assert config.runtime_options.residency_window_size == 8
    assert config.runtime_options.kquant_img_ff_window is False
    assert config.runtime_options.kquant_img_ff_cache_max_blocks == 60
    assert config.runtime_options.kquant_img_ff_codec == "q6_k"
    assert config.lora_paths == ()
    assert config.lora_scales == ()
    assert not config.release_encoders_after_encode


def test_warm_worker_parses_kquant_img_ff_options() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--model",
            "model",
            "--residency",
            "shard",
            "--kquant-img-ff-window",
            "--kquant-img-ff-cache-max-blocks",
            "60",
            "--kquant-img-ff-codec",
            "q6_k",
        ]
    )

    options = _runtime_options_from_args(args)

    assert options.kquant_img_ff_window is True
    assert options.kquant_img_ff_cache_max_blocks == 60
    assert options.kquant_img_ff_codec == "q6_k"


def test_response_json_is_stable_and_machine_readable() -> None:
    payload = json.loads(
        response_json(
            request_id="first",
            ok=True,
            output="edited.png",
            seconds=1.23456789,
        )
    )

    assert payload == {
        "id": "first",
        "ok": True,
        "output": "edited.png",
        "seconds": 1.234568,
    }


def test_worker_keeps_runtime_context_open_across_requests() -> None:
    FakeQwenImageEdit.instances.clear()
    runtime_context = FakeRuntimeContext()
    seen_options: list[RuntimeOptions] = []

    def override_context_factory(options: RuntimeOptions) -> FakeRuntimeContext:
        seen_options.append(options)
        return runtime_context

    worker = WarmQwenEditWorker(
        WarmWorkerConfig(
            model_path="model",
            lora_paths=("adapter.safetensors",),
            lora_scales=(1.0,),
            runtime_options=RuntimeOptions(profile=True),
        ),
        model_factory=FakeQwenImageEdit,
        override_context_factory=override_context_factory,
    )

    first = worker.generate(
        WarmEditRequest(
            prompt="cafe",
            image_paths=("portrait.png",),
            output="first.png",
            metadata=True,
            overwrite=True,
        )
    )
    second = worker.generate(
        WarmEditRequest(
            prompt="office",
            image_paths=("portrait.png",),
            output="second.png",
        )
    )
    worker.close()

    assert first["ok"] is True
    assert second["ok"] is True
    assert runtime_context.entered
    assert runtime_context.exited
    assert seen_options == [RuntimeOptions(profile=True)]
    assert len(FakeQwenImageEdit.instances) == 1
    model = FakeQwenImageEdit.instances[0]
    assert model.init_args == {
        "quantize": None,
        "model_path": "model",
        "lora_paths": ["adapter.safetensors"],
        "lora_scales": [1.0],
    }
    assert model.generated[0]["prompt"] == "cafe"
    assert model.generated[0]["image_path"] == "portrait.png"
    assert model.generated[0]["image_paths"] == ["portrait.png"]
    assert model.generated[0]["num_inference_steps"] == 8
    assert model.image.saved == [
        {"path": "first.png", "export_json_metadata": True, "overwrite": True},
        {"path": "second.png", "export_json_metadata": False, "overwrite": False},
    ]


def test_worker_can_register_encoder_release_callback_for_cache_only_warm_runs() -> None:
    FakeQwenImageEdit.instances.clear()
    created_callbacks: list[tuple[object, int | None]] = []

    def callback_factory(model: object, cache_limit_bytes: int | None) -> object:
        callback = {"model": model, "cache_limit_bytes": cache_limit_bytes}
        created_callbacks.append((model, cache_limit_bytes))
        return callback

    worker = WarmQwenEditWorker(
        WarmWorkerConfig(
            model_path="model",
            release_encoders_after_encode=True,
            mlx_cache_limit_gb=1.5,
        ),
        model_factory=FakeQwenImageEdit,
        override_context_factory=lambda _options: FakeRuntimeContext(),
        encoder_release_callback_factory=callback_factory,
    )

    worker.start()
    worker.close()

    model = FakeQwenImageEdit.instances[0]
    assert created_callbacks == [(model, 1_500_000_000)]
    assert model.callbacks.registered == [
        {"model": model, "cache_limit_bytes": 1_500_000_000}
    ]


def test_cli_processes_jsonl_requests_with_injected_worker() -> None:
    created_workers: list[FakeCliWorker] = []

    class FakeCliWorker:
        def __init__(self, config: WarmWorkerConfig) -> None:
            self.config = config
            self.closed = False
            created_workers.append(self)

        def start(self) -> None:
            pass

        def generate(self, request: WarmEditRequest) -> dict[str, object]:
            return {"id": request.id, "ok": True, "output": request.output, "seconds": 0.0}

        def close(self) -> None:
            self.closed = True

    stdin = io.StringIO(
        json.dumps(
            {
                "id": "one",
                "prompt": "cafe",
                "image_path": "portrait.png",
                "output": "edited.png",
            }
        )
        + "\n"
    )
    stdout = io.StringIO()

    exit_code = main(
        [
            "--model",
            "model",
            "--lora-paths",
            "adapter.safetensors",
            "--lora-scales",
            "1.0",
            "--release-encoders-after-encode",
            "--mlx-cache-limit-gb",
            "1.5",
            "--cache-anchor-mode",
            "absolute",
            "--cache-predictor",
            "linear-residual",
            "--cache-threshold-schedule",
            "flow-aware",
            "--cache-region-policy",
            "target-conservative",
            "--reference-conditioning-size",
            "fit-box",
            "--reference-conditioning-short-side",
            "640",
            "--reference-conditioning-max-width",
            "576",
            "--reference-conditioning-max-height",
            "768",
            "--release-policy",
            "keep-last",
            "--dense-img-ff-window",
            "--dense-img-ff-cache-max-blocks",
            "60",
            "--lora-tensor-cache",
            "--lora-tensor-cache-max-windows",
            "8",
            "--patched-window-cache-max-windows",
            "2",
            "--condition-token-merge",
            "--condition-token-merge-stride",
            "2",
            "--condition-token-merge-start-block",
            "2",
            "--condition-token-merge-back-blocks",
            "2",
            "--text-token-merge",
            "--text-token-merge-stride",
            "2",
            "--text-token-merge-start-block",
            "2",
            "--text-token-merge-back-blocks",
            "2",
            "--q6-linear-profile",
            "--profile",
        ],
        stdin=stdin,
        stdout=stdout,
        worker_factory=FakeCliWorker,
    )

    assert exit_code == 0
    assert json.loads(stdout.getvalue()) == {
        "id": "one",
        "ok": True,
        "output": "edited.png",
        "seconds": 0.0,
    }
    assert created_workers[0].config.runtime_options.profile
    assert created_workers[0].config.runtime_options.cache_anchor_mode == "absolute"
    assert created_workers[0].config.runtime_options.cache_predictor == "linear-residual"
    assert created_workers[0].config.runtime_options.cache_threshold_schedule == "flow-aware"
    assert created_workers[0].config.runtime_options.cache_region_policy == "target-conservative"
    assert created_workers[0].config.runtime_options.reference_conditioning_size == "fit-box"
    assert created_workers[0].config.runtime_options.reference_conditioning_short_side == 640
    assert created_workers[0].config.runtime_options.reference_conditioning_max_width == 576
    assert created_workers[0].config.runtime_options.reference_conditioning_max_height == 768
    assert created_workers[0].config.runtime_options.release_policy == "keep-last"
    assert created_workers[0].config.runtime_options.dense_img_ff_window is True
    assert created_workers[0].config.runtime_options.dense_img_ff_cache_max_blocks == 60
    assert created_workers[0].config.runtime_options.kquant_img_ff_window is False
    assert created_workers[0].config.runtime_options.lora_tensor_cache is True
    assert created_workers[0].config.runtime_options.lora_tensor_cache_max_windows == 8
    assert created_workers[0].config.runtime_options.patched_window_cache_max_windows == 2
    assert created_workers[0].config.runtime_options.condition_token_merge is True
    assert created_workers[0].config.runtime_options.condition_token_merge_stride == 2
    assert created_workers[0].config.runtime_options.condition_token_merge_start_block == 2
    assert created_workers[0].config.runtime_options.condition_token_merge_back_blocks == 2
    assert created_workers[0].config.runtime_options.text_token_merge is True
    assert created_workers[0].config.runtime_options.text_token_merge_stride == 2
    assert created_workers[0].config.runtime_options.text_token_merge_start_block == 2
    assert created_workers[0].config.runtime_options.text_token_merge_back_blocks == 2
    assert created_workers[0].config.runtime_options.q6_linear_profile is True
    assert created_workers[0].config.release_encoders_after_encode
    assert created_workers[0].config.mlx_cache_limit_gb == 1.5
    assert created_workers[0].config.lora_paths == ("adapter.safetensors",)
    assert created_workers[0].config.lora_scales == (1.0,)
    assert created_workers[0].closed


def test_cli_reports_bad_json_request_without_stopping_worker() -> None:
    class FakeCliWorker:
        def __init__(self, _config: WarmWorkerConfig) -> None:
            pass

        def start(self) -> None:
            pass

        def generate(self, request: WarmEditRequest) -> dict[str, object]:
            return {"id": request.id, "ok": True, "output": request.output, "seconds": 0.0}

        def close(self) -> None:
            pass

    stdin = io.StringIO(
        "{bad json}\n"
        + json.dumps(
            {
                "id": "good",
                "prompt": "cafe",
                "image_path": "portrait.png",
                "output": "edited.png",
            }
        )
        + "\n"
    )
    stdout = io.StringIO()

    exit_code = main(
        ["--model", "model"],
        stdin=stdin,
        stdout=stdout,
        worker_factory=FakeCliWorker,
    )

    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert exit_code == 0
    assert responses[0]["ok"] is False
    assert "invalid JSON" in responses[0]["error"]
    assert responses[1]["ok"] is True
