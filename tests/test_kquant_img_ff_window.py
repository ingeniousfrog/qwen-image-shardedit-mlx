"""Lightweight tests for K-quant img_ff spike helpers."""

from __future__ import annotations

from types import SimpleNamespace

from shardedit_mlx.kquant_img_ff_window import kquant_cache_bytes


def test_kquant_cache_bytes_sums_linear_payloads() -> None:
    cache = {
        (0, "q6_k"): SimpleNamespace(
            mlp_in=SimpleNamespace(bytes_materialized=11),
            mlp_out=SimpleNamespace(bytes_materialized=13),
        ),
        (1, "q6_k"): SimpleNamespace(
            mlp_in=SimpleNamespace(bytes_materialized=17),
            mlp_out=SimpleNamespace(bytes_materialized=19),
        ),
    }

    assert kquant_cache_bytes(cache) == 60
