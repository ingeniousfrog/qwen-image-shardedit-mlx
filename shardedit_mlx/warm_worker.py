"""Persistent JSONL worker for warm Qwen Image Edit requests."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO

from shardedit_mlx.mflux_fast_edit import (
    CACHE_PREDICTORS,
    CACHE_REGION_POLICIES,
    CACHE_THRESHOLD_SCHEDULES,
    DEFAULT_CACHE_ANCHOR_MODE,
    DEFAULT_CACHE_PREDICTOR,
    DEFAULT_CACHE_REGION_POLICY,
    DEFAULT_CACHE_THRESHOLD_SCHEDULE,
    DEFAULT_CONDITION_TOKEN_MERGE_BACK_BLOCKS,
    DEFAULT_CONDITION_TOKEN_MERGE_START_BLOCK,
    DEFAULT_CONDITION_TOKEN_MERGE_STRIDE,
    DEFAULT_DENSE_IMG_FF_CACHE_MAX_BLOCKS,
    DEFAULT_KQUANT_IMG_FF_CACHE_MAX_BLOCKS,
    DEFAULT_KQUANT_IMG_FF_CODEC,
    DEFAULT_LORA_TENSOR_CACHE_MAX_WINDOWS,
    DEFAULT_REFERENCE_CONDITIONING_MAX_HEIGHT,
    DEFAULT_REFERENCE_CONDITIONING_MAX_WIDTH,
    DEFAULT_REFERENCE_CONDITIONING_SIZE,
    DEFAULT_REFERENCE_CONDITIONING_SHORT_SIDE,
    DEFAULT_RELEASE_POLICY,
    DEFAULT_RESIDENCY_MODE,
    DEFAULT_RESIDENCY_WINDOW_SIZE,
    DEFAULT_TEXT_TOKEN_MERGE_BACK_BLOCKS,
    DEFAULT_TEXT_TOKEN_MERGE_START_BLOCK,
    DEFAULT_TEXT_TOKEN_MERGE_STRIDE,
    REFERENCE_CONDITIONING_SIZE_POLICIES,
    RELEASE_POLICIES,
    RuntimeOptions,
    install_runtime_overrides,
)


class WarmWorkerRequestError(ValueError):
    """Raised when one JSONL request is malformed."""


class GeneratedImageLike(Protocol):
    def save(
        self,
        *,
        path: str,
        export_json_metadata: bool,
        overwrite: bool,
    ) -> None:
        ...


class QwenImageEditLike(Protocol):
    def generate_image(self, **kwargs: object) -> GeneratedImageLike:
        ...


ModelFactory = Callable[..., QwenImageEditLike]
OverrideContextFactory = Callable[[RuntimeOptions], AbstractContextManager[None]]
EncoderReleaseCallbackFactory = Callable[[QwenImageEditLike, int | None], object]


@dataclass(frozen=True)
class WarmEditRequest:
    """One JSONL edit request handled by a persistent model process."""

    prompt: str
    image_paths: tuple[str, ...]
    output: str
    id: str | None = None
    seed: int = 42
    width: int = 768
    height: int = 768
    steps: int = 8
    guidance: float = 1.0
    negative_prompt: str | None = None
    scheduler: str = "linear"
    metadata: bool = False
    overwrite: bool = False


@dataclass(frozen=True)
class WarmWorkerConfig:
    """Runtime configuration shared by all requests in one worker process."""

    model_path: str
    lora_paths: tuple[str, ...] = ()
    lora_scales: tuple[float, ...] = ()
    quantize: int | None = None
    runtime_options: RuntimeOptions = RuntimeOptions()
    release_encoders_after_encode: bool = False
    mlx_cache_limit_gb: float | None = None


def _require_string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if value is None:
        raise WarmWorkerRequestError(f"{key} is required")
    if not isinstance(value, str):
        raise WarmWorkerRequestError(f"{key} must be a string")
    if not value:
        raise WarmWorkerRequestError(f"{key} cannot be empty")
    return value


def _optional_string(raw: Mapping[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise WarmWorkerRequestError(f"{key} must be a string")
    return value


def _integer(raw: Mapping[str, Any], key: str, default: int, *, minimum: int) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise WarmWorkerRequestError(f"{key} must be an integer")
    if value < minimum:
        raise WarmWorkerRequestError(f"{key} must be >= {minimum}")
    return value


def _float(raw: Mapping[str, Any], key: str, default: float) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise WarmWorkerRequestError(f"{key} must be a number")
    return float(value)


def _boolean(raw: Mapping[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise WarmWorkerRequestError(f"{key} must be a boolean")
    return value


def _image_paths(raw: Mapping[str, Any]) -> tuple[str, ...]:
    if "image_paths" in raw:
        value = raw["image_paths"]
        if not isinstance(value, list | tuple):
            raise WarmWorkerRequestError("image_paths must be a list of strings")
        if not value:
            raise WarmWorkerRequestError("image_paths cannot be empty")
        paths = tuple(value)
        if any(not isinstance(path, str) or not path for path in paths):
            raise WarmWorkerRequestError("image_paths must contain non-empty strings")
        return paths
    return (_require_string(raw, "image_path"),)


def parse_request(raw: Mapping[str, Any]) -> WarmEditRequest:
    """Validate and normalize one JSON object from the worker input stream."""

    request_id = _optional_string(raw, "id")
    return WarmEditRequest(
        id=request_id,
        prompt=_require_string(raw, "prompt"),
        image_paths=_image_paths(raw),
        output=_require_string(raw, "output"),
        seed=_integer(raw, "seed", 42, minimum=0),
        width=_integer(raw, "width", 768, minimum=1),
        height=_integer(raw, "height", 768, minimum=1),
        steps=_integer(raw, "steps", 8, minimum=1),
        guidance=_float(raw, "guidance", 1.0),
        negative_prompt=_optional_string(raw, "negative_prompt"),
        scheduler=_optional_string(raw, "scheduler") or "linear",
        metadata=_boolean(raw, "metadata", False),
        overwrite=_boolean(raw, "overwrite", False),
    )


def request_from_json_line(line: str) -> WarmEditRequest:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise WarmWorkerRequestError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise WarmWorkerRequestError("request must be a JSON object")
    return parse_request(payload)


def response_json(
    *,
    request_id: str | None,
    ok: bool,
    output: str | None = None,
    seconds: float | None = None,
    error: str | None = None,
) -> str:
    payload: dict[str, object] = {
        "id": request_id,
        "ok": ok,
    }
    if output is not None:
        payload["output"] = output
    if seconds is not None:
        payload["seconds"] = round(seconds, 6)
    if error is not None:
        payload["error"] = error
    return json.dumps(payload, sort_keys=True)


class WarmQwenEditWorker:
    """Keep one QwenImageEdit instance and qwen-image-shardedit-mlx override context alive."""

    def __init__(
        self,
        config: WarmWorkerConfig,
        *,
        model_factory: ModelFactory | None = None,
        override_context_factory: OverrideContextFactory = install_runtime_overrides,
        encoder_release_callback_factory: EncoderReleaseCallbackFactory | None = None,
    ) -> None:
        self.config = config
        self._model_factory = model_factory
        self._override_context_factory = override_context_factory
        self._encoder_release_callback_factory = (
            encoder_release_callback_factory or self._create_encoder_release_callback
        )
        self._context: AbstractContextManager[None] | None = None
        self._model: QwenImageEditLike | None = None

    def start(self) -> None:
        if self._model is not None:
            return
        context = self._override_context_factory(self.config.runtime_options)
        context.__enter__()
        try:
            model_factory = self._model_factory or self._load_model_factory()
            with self._profile_output_context():
                self._model = model_factory(
                    quantize=self.config.quantize,
                    model_path=self.config.model_path,
                    lora_paths=list(self.config.lora_paths) or None,
                    lora_scales=list(self.config.lora_scales) or None,
                )
            self._register_encoder_release_callback(self._model)
            self._context = context
        except Exception:
            context.__exit__(*sys.exc_info())
            raise

    def close(self) -> None:
        context = self._context
        self._model = None
        self._context = None
        if context is not None:
            context.__exit__(None, None, None)

    def generate(self, request: WarmEditRequest) -> dict[str, object]:
        self.start()
        if self._model is None:
            raise RuntimeError("worker did not initialize a model")
        started_at = time.perf_counter()
        Path(request.output).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with self._profile_output_context():
            generated = self._model.generate_image(
                seed=request.seed,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                guidance=request.guidance,
                scheduler=request.scheduler,
                image_path=request.image_paths[0],
                image_paths=list(request.image_paths),
                num_inference_steps=request.steps,
            )
        with self._profile_output_context():
            generated.save(
                path=request.output,
                export_json_metadata=request.metadata,
                overwrite=request.overwrite,
            )
        return {
            "id": request.id,
            "ok": True,
            "output": request.output,
            "seconds": round(time.perf_counter() - started_at, 6),
        }

    @staticmethod
    def _load_model_factory() -> ModelFactory:
        try:
            from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
        except ImportError as exc:  # pragma: no cover - depends on runtime environment
            raise SystemExit("mflux>=0.18 is required to run shardedit-warm-edit") from exc
        return QwenImageEdit

    def _profile_output_context(self) -> AbstractContextManager[None]:
        if not self.config.runtime_options.profile:
            return nullcontext()
        return redirect_stdout(sys.stderr)

    def _register_encoder_release_callback(self, model: QwenImageEditLike) -> None:
        if not self.config.release_encoders_after_encode:
            return
        callbacks = getattr(model, "callbacks", None)
        if callbacks is None or not hasattr(callbacks, "register"):
            raise RuntimeError("model does not expose a callback registry")
        callback = self._encoder_release_callback_factory(
            model,
            self._mlx_cache_limit_bytes(),
        )
        callbacks.register(callback)

    def _mlx_cache_limit_bytes(self) -> int | None:
        if self.config.mlx_cache_limit_gb is None:
            return None
        return int(self.config.mlx_cache_limit_gb * 1_000_000_000)

    @staticmethod
    def _create_encoder_release_callback(
        model: QwenImageEditLike,
        cache_limit_bytes: int | None,
    ) -> object:
        try:
            from mflux.callbacks.instances.memory_saver import MemorySaver
        except ImportError as exc:  # pragma: no cover - depends on runtime environment
            raise SystemExit("mflux>=0.18 is required to release encoders") from exc
        return MemorySaver(
            model=model,
            keep_transformer=True,
            cache_limit_bytes=cache_limit_bytes,
            num_seeds=1,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a persistent qwen-image-shardedit-mlx Qwen Image Edit JSONL worker."
    )
    parser.add_argument("--model", required=True, help="Local Qwen Image Edit model path")
    parser.add_argument("--quantize", type=int, default=None, help="Optional mflux quantize value")
    parser.add_argument("--lora-paths", nargs="*", default=(), help="LoRA paths")
    parser.add_argument("--lora-scales", nargs="*", type=float, default=(), help="LoRA scales")
    parser.add_argument(
        "--residency",
        choices=("none", "shard", "window"),
        default=DEFAULT_RESIDENCY_MODE,
        help="Transformer residency policy",
    )
    parser.add_argument(
        "--residency-window-size",
        type=int,
        default=DEFAULT_RESIDENCY_WINDOW_SIZE,
        help="Block count for window residency",
    )
    parser.add_argument(
        "--release-policy",
        choices=RELEASE_POLICIES,
        default=DEFAULT_RELEASE_POLICY,
        help="Clear MLX cache after every residency window or once per denoise step",
    )
    parser.add_argument(
        "--dense-img-ff-window",
        action="store_true",
        help="Dequantize image MLP to dense bf16 per residency window",
    )
    parser.add_argument(
        "--dense-img-ff-cache-max-blocks",
        type=int,
        default=DEFAULT_DENSE_IMG_FF_CACHE_MAX_BLOCKS,
        help="FIFO cap for cross-step dense image MLP reuse",
    )
    parser.add_argument(
        "--kquant-img-ff-window",
        action="store_true",
        help="Diagnostic no-go probe: re-encode image MLP to mlx-kquant per residency window",
    )
    parser.add_argument(
        "--kquant-img-ff-cache-max-blocks",
        type=int,
        default=DEFAULT_KQUANT_IMG_FF_CACHE_MAX_BLOCKS,
        help="FIFO cap for cross-step K-quant image MLP reuse",
    )
    parser.add_argument(
        "--kquant-img-ff-codec",
        default=DEFAULT_KQUANT_IMG_FF_CODEC,
        help="mlx-kquant codec for image MLP weights",
    )
    parser.add_argument(
        "--lora-tensor-cache",
        action="store_true",
        help="Opt-in probe: reuse already loaded per-window LoRA tensors",
    )
    parser.add_argument(
        "--lora-tensor-cache-max-windows",
        type=int,
        default=DEFAULT_LORA_TENSOR_CACHE_MAX_WINDOWS,
        help="FIFO cap for cached per-window LoRA tensor sets",
    )
    parser.add_argument(
        "--patched-window-cache-max-windows",
        type=int,
        default=0,
        help="Opt-in probe: cache loaded+LoRA-patched Transformer windows; 0 disables",
    )
    parser.add_argument("--cache-threshold", type=float, default=0.0)
    parser.add_argument("--cache-max-consecutive", type=int, default=1)
    parser.add_argument("--cache-warmup-steps", type=int, default=1)
    parser.add_argument("--cache-back-blocks", type=int, default=0)
    parser.add_argument(
        "--cache-anchor-mode",
        choices=("residual", "absolute"),
        default=DEFAULT_CACHE_ANCHOR_MODE,
    )
    parser.add_argument(
        "--cache-predictor",
        choices=CACHE_PREDICTORS,
        default=DEFAULT_CACHE_PREDICTOR,
    )
    parser.add_argument(
        "--cache-threshold-schedule",
        choices=CACHE_THRESHOLD_SCHEDULES,
        default=DEFAULT_CACHE_THRESHOLD_SCHEDULE,
    )
    parser.add_argument(
        "--cache-region-policy",
        choices=CACHE_REGION_POLICIES,
        default=DEFAULT_CACHE_REGION_POLICY,
    )
    parser.add_argument(
        "--reference-conditioning-size",
        choices=REFERENCE_CONDITIONING_SIZE_POLICIES,
        default=DEFAULT_REFERENCE_CONDITIONING_SIZE,
    )
    parser.add_argument(
        "--reference-conditioning-short-side",
        type=int,
        default=DEFAULT_REFERENCE_CONDITIONING_SHORT_SIDE,
    )
    parser.add_argument(
        "--reference-conditioning-max-width",
        type=int,
        default=DEFAULT_REFERENCE_CONDITIONING_MAX_WIDTH,
    )
    parser.add_argument(
        "--reference-conditioning-max-height",
        type=int,
        default=DEFAULT_REFERENCE_CONDITIONING_MAX_HEIGHT,
    )
    parser.add_argument(
        "--condition-token-merge",
        action="store_true",
        help=(
            "Diagnostic/rejected V0: merge only reference-condition tokens "
            "inside eligible full-miss Transformer blocks"
        ),
    )
    parser.add_argument(
        "--condition-token-merge-stride",
        type=int,
        default=DEFAULT_CONDITION_TOKEN_MERGE_STRIDE,
        help="Local horizontal condition-token merge stride",
    )
    parser.add_argument(
        "--condition-token-merge-start-block",
        type=int,
        default=DEFAULT_CONDITION_TOKEN_MERGE_START_BLOCK,
        help="One-based first Transformer block eligible for condition-token merge",
    )
    parser.add_argument(
        "--condition-token-merge-back-blocks",
        type=int,
        default=DEFAULT_CONDITION_TOKEN_MERGE_BACK_BLOCKS,
        help="Keep this many final Transformer blocks unmerged",
    )
    parser.add_argument(
        "--text-token-merge",
        action="store_true",
        help=(
            "Diagnostic V0: merge prompt/VL text tokens inside full-miss blocks; "
            "smoke showed no speedup"
        ),
    )
    parser.add_argument(
        "--text-token-merge-stride",
        type=int,
        default=DEFAULT_TEXT_TOKEN_MERGE_STRIDE,
        help="Local text-token merge stride",
    )
    parser.add_argument(
        "--text-token-merge-start-block",
        type=int,
        default=DEFAULT_TEXT_TOKEN_MERGE_START_BLOCK,
        help="One-based first Transformer block eligible for text-token merge",
    )
    parser.add_argument(
        "--text-token-merge-back-blocks",
        type=int,
        default=DEFAULT_TEXT_TOKEN_MERGE_BACK_BLOCKS,
        help="Keep this many final Transformer blocks text-unmerged",
    )
    parser.add_argument(
        "--q6-linear-profile",
        action="store_true",
        help=(
            "Diagnostic: synchronize and aggregate QuantizedLinear calls inside "
            "executed Qwen Transformer blocks"
        ),
    )
    parser.add_argument(
        "--release-encoders-after-encode",
        action="store_true",
        help=(
            "Release VLM/text encoders after each prompt encode. This lowers denoise "
            "memory pressure, but later cache misses require a fresh worker."
        ),
    )
    parser.add_argument(
        "--mlx-cache-limit-gb",
        type=float,
        default=None,
        help="Optional MLX cache limit for encoder-release mode",
    )
    parser.add_argument("--profile", action="store_true")
    return parser


def _runtime_options_from_args(args: argparse.Namespace) -> RuntimeOptions:
    if args.residency_window_size < 1:
        raise SystemExit("--residency-window-size must be >= 1")
    if args.cache_threshold < 0.0 or args.cache_threshold > 1.0:
        raise SystemExit("--cache-threshold must be between 0 and 1")
    if args.cache_max_consecutive < 1:
        raise SystemExit("--cache-max-consecutive must be >= 1")
    if args.cache_warmup_steps < 0:
        raise SystemExit("--cache-warmup-steps must be >= 0")
    if args.cache_back_blocks < 0:
        raise SystemExit("--cache-back-blocks must be >= 0")
    if args.reference_conditioning_short_side < 1:
        raise SystemExit("--reference-conditioning-short-side must be >= 1")
    if args.reference_conditioning_max_width < 1:
        raise SystemExit("--reference-conditioning-max-width must be >= 1")
    if args.reference_conditioning_max_height < 1:
        raise SystemExit("--reference-conditioning-max-height must be >= 1")
    if args.reference_conditioning_size == "fit-box" and (
        args.reference_conditioning_max_width < 32 or args.reference_conditioning_max_height < 32
    ):
        raise SystemExit("fit-box reference conditioning max dimensions must be >= 32")
    if args.mlx_cache_limit_gb is not None and args.mlx_cache_limit_gb <= 0:
        raise SystemExit("--mlx-cache-limit-gb must be > 0")
    if args.dense_img_ff_window and args.residency == "none":
        raise SystemExit("--dense-img-ff-window requires --residency shard or window")
    if args.dense_img_ff_cache_max_blocks < 1:
        raise SystemExit("--dense-img-ff-cache-max-blocks must be >= 1")
    if args.kquant_img_ff_window and args.residency == "none":
        raise SystemExit("--kquant-img-ff-window requires --residency shard or window")
    if args.dense_img_ff_window and args.kquant_img_ff_window:
        raise SystemExit("--dense-img-ff-window and --kquant-img-ff-window are mutually exclusive")
    if args.kquant_img_ff_cache_max_blocks < 1:
        raise SystemExit("--kquant-img-ff-cache-max-blocks must be >= 1")
    if args.lora_tensor_cache and args.residency == "none":
        raise SystemExit("--lora-tensor-cache requires --residency shard or window")
    if args.lora_tensor_cache_max_windows < 1:
        raise SystemExit("--lora-tensor-cache-max-windows must be >= 1")
    if args.patched_window_cache_max_windows < 0:
        raise SystemExit("--patched-window-cache-max-windows must be >= 0")
    if args.patched_window_cache_max_windows > 0 and args.residency == "none":
        raise SystemExit(
            "--patched-window-cache-max-windows requires --residency shard or window"
        )
    if args.condition_token_merge_stride < 2:
        raise SystemExit("--condition-token-merge-stride must be >= 2")
    if args.condition_token_merge_start_block < 1:
        raise SystemExit("--condition-token-merge-start-block must be >= 1")
    if args.condition_token_merge_back_blocks < 0:
        raise SystemExit("--condition-token-merge-back-blocks must be >= 0")
    if args.text_token_merge_stride < 2:
        raise SystemExit("--text-token-merge-stride must be >= 2")
    if args.text_token_merge_start_block < 1:
        raise SystemExit("--text-token-merge-start-block must be >= 1")
    if args.text_token_merge_back_blocks < 0:
        raise SystemExit("--text-token-merge-back-blocks must be >= 0")
    return RuntimeOptions(
        cache_threshold=args.cache_threshold,
        cache_max_consecutive=args.cache_max_consecutive,
        cache_warmup_steps=args.cache_warmup_steps,
        cache_back_blocks=args.cache_back_blocks,
        cache_anchor_mode=args.cache_anchor_mode,
        cache_predictor=args.cache_predictor,
        cache_threshold_schedule=args.cache_threshold_schedule,
        cache_region_policy=args.cache_region_policy,
        reference_conditioning_size=args.reference_conditioning_size,
        reference_conditioning_short_side=args.reference_conditioning_short_side,
        reference_conditioning_max_width=args.reference_conditioning_max_width,
        reference_conditioning_max_height=args.reference_conditioning_max_height,
        residency_mode=args.residency,
        residency_window_size=args.residency_window_size,
        release_policy=args.release_policy,
        dense_img_ff_window=args.dense_img_ff_window,
        dense_img_ff_cache_max_blocks=args.dense_img_ff_cache_max_blocks,
        kquant_img_ff_window=args.kquant_img_ff_window,
        kquant_img_ff_cache_max_blocks=args.kquant_img_ff_cache_max_blocks,
        kquant_img_ff_codec=args.kquant_img_ff_codec,
        lora_tensor_cache=args.lora_tensor_cache,
        lora_tensor_cache_max_windows=args.lora_tensor_cache_max_windows,
        patched_window_cache_max_windows=args.patched_window_cache_max_windows,
        condition_token_merge=args.condition_token_merge,
        condition_token_merge_stride=args.condition_token_merge_stride,
        condition_token_merge_start_block=args.condition_token_merge_start_block,
        condition_token_merge_back_blocks=args.condition_token_merge_back_blocks,
        text_token_merge=args.text_token_merge,
        text_token_merge_stride=args.text_token_merge_stride,
        text_token_merge_start_block=args.text_token_merge_start_block,
        text_token_merge_back_blocks=args.text_token_merge_back_blocks,
        q6_linear_profile=args.q6_linear_profile,
        profile=args.profile or args.q6_linear_profile,
    )


def _config_from_args(args: argparse.Namespace) -> WarmWorkerConfig:
    if args.lora_scales and len(args.lora_scales) != len(args.lora_paths):
        raise SystemExit("--lora-scales must match --lora-paths length")
    return WarmWorkerConfig(
        model_path=args.model,
        lora_paths=tuple(args.lora_paths),
        lora_scales=tuple(args.lora_scales),
        quantize=args.quantize,
        runtime_options=_runtime_options_from_args(args),
        release_encoders_after_encode=args.release_encoders_after_encode,
        mlx_cache_limit_gb=args.mlx_cache_limit_gb,
    )


def _write_response(stdout: TextIO, payload: Mapping[str, object]) -> None:
    stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    stdout.flush()


def main(
    argv: list[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    worker_factory: Callable[[WarmWorkerConfig], WarmQwenEditWorker] = WarmQwenEditWorker,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    input_stream = stdin or sys.stdin
    output_stream = stdout or sys.stdout
    worker = worker_factory(config)
    worker.start()
    try:
        for line in input_stream:
            if not line.strip():
                continue
            try:
                request = request_from_json_line(line)
                _write_response(output_stream, worker.generate(request))
            except WarmWorkerRequestError as exc:
                _write_response(
                    output_stream,
                    {
                        "id": None,
                        "ok": False,
                        "error": str(exc),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                _write_response(
                    output_stream,
                    {
                        "id": None,
                        "ok": False,
                        "error": str(exc),
                    },
                )
    finally:
        worker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
