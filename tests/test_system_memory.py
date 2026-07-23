from __future__ import annotations

import pytest

from shardedit_mlx.system_memory import (
    counter_deltas,
    parse_ioreg_gpu_performance_statistics,
    parse_memory_pressure_free_percentage,
    parse_swap_usage,
    parse_vm_stat,
)


VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   507749.
\"Translation faults\":                      175043958.
Pages stored in compressor:                   450097.
Pageins:                                    39783748.
Pageouts:                                      94873.
Swapins:                                    57004059.
Swapouts:                                   63471224.
"""


def test_parse_vm_stat_normalizes_names_and_preserves_page_size() -> None:
    snapshot = parse_vm_stat(VM_STAT)

    assert snapshot.page_size_bytes == 16384
    assert snapshot.value("pages_free") == 507749
    assert snapshot.value("translation_faults") == 175043958
    assert snapshot.value("pages_stored_in_compressor") == 450097


def test_parse_vm_stat_rejects_invalid_output() -> None:
    with pytest.raises(ValueError, match="page size"):
        parse_vm_stat("not vm_stat output")


def test_parse_memory_pressure_free_percentage() -> None:
    output = "System-wide memory free percentage: 87%\n"

    assert parse_memory_pressure_free_percentage(output) == 87
    assert parse_memory_pressure_free_percentage("no percentage") is None


def test_parse_swap_usage_supports_megabytes_and_gigabytes() -> None:
    usage = parse_swap_usage(
        "total = 8.00G  used = 7168.00M  free = 1.00G  (encrypted)"
    )

    assert usage is not None
    assert usage.total_bytes == 8 * 1024**3
    assert usage.used_bytes == 7168 * 1024**2
    assert usage.free_bytes == 1024**3
    assert parse_swap_usage("permission denied") is None


def test_counter_deltas_report_pages_and_bytes() -> None:
    before = parse_vm_stat(VM_STAT)
    after = parse_vm_stat(
        VM_STAT.replace("39783748", "39783758")
        .replace("94873", "94875")
        .replace("57004059", "57004062")
        .replace("63471224", "63471228")
    )

    deltas = counter_deltas(before, after)

    assert deltas["pageins_pages"] == 10
    assert deltas["pageins_bytes"] == 10 * 16384
    assert deltas["pageouts_pages"] == 2
    assert deltas["swapins_pages"] == 3
    assert deltas["swapouts_pages"] == 4


IOREG_GPU = """+-o AGXAcceleratorG14G  <class AGXAcceleratorG14G>
    {
      "PerformanceStatistics" = {"In use system memory (driver)"=0,"Alloc system memory"=3085729792,"Tiler Utilization %"=40,"recoveryCount"=0,"lastRecoveryTime"=0,"Renderer Utilization %"=39,"TiledSceneBytes"=819200,"Device Utilization %"=40,"SplitSceneCount"=0,"Allocated PB Size"=70385664,"In use system memory"=696860672}
      "model" = "Apple M2"
    }
"""


def test_parse_ioreg_gpu_performance_statistics() -> None:
    snapshot = parse_ioreg_gpu_performance_statistics(IOREG_GPU)

    assert snapshot.device_utilization_percent == 40
    assert snapshot.renderer_utilization_percent == 39
    assert snapshot.tiler_utilization_percent == 40
    assert snapshot.alloc_system_memory_bytes == 3085729792
    assert snapshot.in_use_system_memory_bytes == 696860672


def test_parse_ioreg_gpu_performance_statistics_rejects_missing_block() -> None:
    with pytest.raises(ValueError, match="PerformanceStatistics"):
        parse_ioreg_gpu_performance_statistics('+-o AGXAcceleratorG14G\n{\n  "model" = "Apple M2"\n}')


def test_parse_ioreg_gpu_performance_statistics_rejects_missing_fields() -> None:
    output = (
        '"PerformanceStatistics" = {"Device Utilization %"=10,'
        '"Renderer Utilization %"=9,"Alloc system memory"=100}'
    )
    with pytest.raises(ValueError, match="missing required fields"):
        parse_ioreg_gpu_performance_statistics(output)