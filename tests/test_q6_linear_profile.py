from __future__ import annotations

from shardedit_mlx.q6_linear_profile import (
    Q6LinearProfiler,
    qwen_block_linear_call_sites,
    summarize_q6_linear_events,
)


class FakeArray:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class FakeLinear:
    def __init__(self, *, bits: int = 6, group_size: int = 64, mode: str = "affine") -> None:
        self.bits = bits
        self.group_size = group_size
        self.mode = mode


class FakeAttention:
    def __init__(self) -> None:
        self.to_q = FakeLinear()
        self.to_k = FakeLinear()
        self.to_v = FakeLinear()
        self.add_q_proj = FakeLinear()
        self.add_k_proj = FakeLinear()
        self.add_v_proj = FakeLinear()
        self.attn_to_out = [FakeLinear()]
        self.to_add_out = FakeLinear()


class FakeFeedForward:
    def __init__(self) -> None:
        self.mlp_in = FakeLinear()
        self.mlp_out = FakeLinear()


class FakeLoRALinear:
    def __init__(self, linear: FakeLinear) -> None:
        self.linear = linear


class FakeFusedLoRALinear:
    def __init__(self, base_linear: FakeLinear) -> None:
        self.base_linear = base_linear
        self.loras = []


class FakeBlock:
    def __init__(self) -> None:
        self.img_mod_linear = FakeLinear()
        self.txt_mod_linear = FakeLinear()
        self.attn = FakeAttention()
        self.img_ff = FakeFeedForward()
        self.txt_ff = FakeFeedForward()


def test_qwen_block_linear_call_sites_label_q6_hotspots() -> None:
    block = FakeBlock()

    labels = qwen_block_linear_call_sites(block, block_index=2)

    by_site = {label.site: label for label in labels.values()}
    assert by_site["img_ff.mlp_in"].category == "img_mlp"
    assert by_site["img_ff.mlp_in"].block == 3
    assert by_site["attn.to_q"].category == "img_attn_qkv"
    assert by_site["attn.add_q_proj"].category == "txt_attn_qkv"
    assert by_site["txt_mod_linear"].category == "txt_mod"
    assert len(by_site) == 14


def test_q6_linear_profiler_aggregates_same_callsite_across_blocks() -> None:
    first = FakeBlock()
    second = FakeBlock()
    profiler = Q6LinearProfiler()
    profiler.register_block(first, block_index=0)
    profiler.register_block(second, block_index=1)

    with profiler.block_context(step=4, block=1):
        profiler.record_call(
            first.img_ff.mlp_in,
            input_value=FakeArray((1, 3456, 3072)),
            output_value=FakeArray((1, 3456, 12288)),
            seconds=0.25,
        )
    with profiler.block_context(step=4, block=2):
        profiler.record_call(
            second.img_ff.mlp_in,
            input_value=FakeArray((1, 3456, 3072)),
            output_value=FakeArray((1, 3456, 12288)),
            seconds=0.35,
        )

    events = profiler.drain_step(step=4, cache_hit=False, cache_reason="threshold")

    assert len(events) == 1
    event = events[0]
    assert event.seconds == 0.6
    assert event.category == "img_mlp"
    assert event.site == "img_ff.mlp_in"
    assert event.call_count == 2
    assert event.blocks == "1-2"
    assert event.input_shape == (1, 3456, 3072)
    assert event.output_shape == (1, 3456, 12288)
    assert event.bits == 6
    assert event.group_size == 64
    assert event.cache_hit is False


def test_q6_linear_profiler_records_lora_wrapped_base_linears() -> None:
    block = FakeBlock()
    to_q_base = FakeLinear()
    mlp_base = FakeLinear()
    block.attn.to_q = FakeLoRALinear(to_q_base)
    block.img_ff.mlp_in = FakeFusedLoRALinear(mlp_base)
    profiler = Q6LinearProfiler()
    profiler.register_block(block, block_index=0)

    with profiler.block_context(step=4, block=1):
        profiler.record_call(
            to_q_base,
            input_value=FakeArray((1, 3456, 3072)),
            output_value=FakeArray((1, 3456, 3072)),
            seconds=0.25,
        )
        profiler.record_call(
            mlp_base,
            input_value=FakeArray((1, 3456, 3072)),
            output_value=FakeArray((1, 3456, 12288)),
            seconds=0.35,
        )

    events = profiler.drain_step(step=4, cache_hit=False, cache_reason="threshold")

    by_site = {event.site: event for event in events}
    assert by_site["attn.to_q"].category == "img_attn_qkv"
    assert by_site["attn.to_q"].seconds == 0.25
    assert by_site["img_ff.mlp_in"].category == "img_mlp"
    assert by_site["img_ff.mlp_in"].seconds == 0.35


def test_q6_linear_profiler_ignores_unregistered_or_out_of_context_calls() -> None:
    profiler = Q6LinearProfiler()
    line = FakeLinear()
    profiler.record_call(
        line,
        input_value=FakeArray((1, 1, 1)),
        output_value=FakeArray((1, 1, 1)),
        seconds=1.0,
    )

    assert profiler.drain_step(step=1, cache_hit=None, cache_reason=None) == ()


def test_q6_linear_profiler_ignores_stale_module_id_entries() -> None:
    block = FakeBlock()
    profiler = Q6LinearProfiler()
    profiler.register_block(block, block_index=0)

    unrelated = FakeLinear()
    profiler._sites_by_module_id[id(unrelated)] = profiler._sites_by_module_id[  # noqa: SLF001
        id(block.img_ff.mlp_in)
    ]

    with profiler.block_context(step=2, block=1):
        assert profiler.should_record(unrelated) is False
        profiler.record_call(
            unrelated,
            input_value=FakeArray((1, 3456, 3072)),
            output_value=FakeArray((1, 3456, 12288)),
            seconds=1.0,
        )

    assert profiler.drain_step(step=2, cache_hit=False, cache_reason=None) == ()


def test_summarize_q6_linear_events_keeps_full_miss_rows() -> None:
    summaries = summarize_q6_linear_events(
        [
            {
                "name": "q6_linear_profile",
                "step": 2,
                "cache_hit": False,
                "category": "img_mlp",
                "site": "img_ff.mlp_in",
                "seconds": 0.5,
                "call_count": 2,
                "blocks": "1-2",
                "input_shape": [1, 3456, 3072],
                "output_shape": [1, 3456, 12288],
                "bits": 6,
                "group_size": 64,
                "mode": "affine",
            },
            {
                "name": "q6_linear_profile",
                "step": 3,
                "cache_hit": True,
                "category": "img_mlp",
                "site": "img_ff.mlp_in",
                "seconds": 0.1,
                "call_count": 1,
                "blocks": "1",
                "input_shape": [1, 3456, 3072],
                "output_shape": [1, 3456, 12288],
                "bits": 6,
                "group_size": 64,
                "mode": "affine",
            },
        ]
    )

    assert len(summaries) == 1
    assert summaries[0].site == "img_ff.mlp_in"
    assert summaries[0].total_seconds == 0.5
    assert summaries[0].call_count == 2
    assert summaries[0].steps == (2,)
    assert summaries[0].blocks == "1-2"
