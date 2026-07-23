#!/usr/bin/env python3
"""Build the MLX Swift Metal library for a local qwen.image.swift checkout."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_QWEN_IMAGE_SWIFT = Path("external/qwen.image.swift")

KERNELS = [
    "arg_reduce",
    "conv",
    "gemv",
    "layer_norm",
    "random",
    "rms_norm",
    "rope",
    "scaled_dot_product_attention",
    "fence",
    "steel/attn/kernels/steel_attention",
    "arange",
    "binary",
    "binary_two",
    "copy",
    "fft",
    "reduce",
    "quantized",
    "fp4_quantized",
    "scan",
    "softmax",
    "logsumexp",
    "sort",
    "ternary",
    "unary",
    "steel/conv/kernels/steel_conv",
    "steel/conv/kernels/steel_conv_general",
    "steel/gemm/kernels/steel_gemm_fused",
    "steel/gemm/kernels/steel_gemm_gather",
    "steel/gemm/kernels/steel_gemm_masked",
    "steel/gemm/kernels/steel_gemm_splitk",
    "steel/gemm/kernels/steel_gemm_segmented",
    "gemv_masked",
]


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def run(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True, env=env)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def xcrun_find(tool: str) -> Path | None:
    result = run(["xcrun", "-find", tool])
    if result.returncode != 0:
        return None
    path = Path(result.stdout.strip())
    return path if path.exists() else None


def installed_dir_from_metal_version() -> Path | None:
    result = run(["xcrun", "-sdk", "macosx", "metal", "-v"])
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines() + result.stderr.splitlines():
        if line.startswith("InstalledDir:"):
            path = Path(line.split(":", 1)[1].strip())
            if path.exists():
                return path
    return None


def find_tool(tool: str) -> Path:
    candidates: list[Path] = []

    metal_installed_dir = installed_dir_from_metal_version()
    if metal_installed_dir:
        candidates.append(metal_installed_dir / tool)

    path = shutil.which(tool)
    if path:
        candidates.append(Path(path))

    xcrun_path = xcrun_find(tool)
    if xcrun_path:
        candidates.append(xcrun_path)

    found = first_existing(candidates)
    if found:
        return found
    fail(f"could not find {tool}; install full Xcode/Metal toolchain or pass --{tool}-tool")


def kernel_target_name(kernel: str) -> str:
    return Path(kernel).name


def build_metallib(
    mlx_swift_checkout: Path,
    output: Path,
    air_dir: Path,
    metal_tool: Path,
    metallib_tool: Path,
    force: bool,
) -> None:
    project_source = mlx_swift_checkout / "Source" / "Cmlx" / "mlx"
    kernel_dir = project_source / "mlx" / "backend" / "metal" / "kernels"
    version_include = kernel_dir / "metal_3_1"

    if not kernel_dir.exists():
        fail(f"MLX kernel directory does not exist: {kernel_dir}")
    if not version_include.exists():
        fail(f"Metal 3.1 include directory does not exist: {version_include}")

    output.parent.mkdir(parents=True, exist_ok=True)
    air_dir.mkdir(parents=True, exist_ok=True)
    module_cache = air_dir / "clang-module-cache"
    module_cache.mkdir(parents=True, exist_ok=True)
    compile_env = os.environ.copy()
    compile_env["CLANG_MODULE_CACHE_PATH"] = str(module_cache)
    if output.exists() and not force:
        fail(f"output already exists: {output} (pass --force to replace)")
    if output.exists():
        output.unlink()

    air_files: list[Path] = []
    metal_flags = ["-Wall", "-Wextra", "-fno-fast-math", "-Wno-c++17-extensions"]
    for kernel in KERNELS:
        source = kernel_dir / f"{kernel}.metal"
        if not source.exists():
            fail(f"kernel source does not exist: {source}")
        target = air_dir / f"{kernel_target_name(kernel)}.air"
        command = [
            str(metal_tool),
            *metal_flags,
            "-c",
            str(source),
            f"-I{project_source}",
            f"-I{version_include}",
            "-o",
            str(target),
        ]
        print(f"compile {source.relative_to(kernel_dir)}")
        result = run(command, env=compile_env)
        if result.returncode != 0:
            raise SystemExit(result.stderr or result.stdout)
        air_files.append(target)

    command = [str(metallib_tool), *[str(path) for path in air_files], "-o", str(output)]
    print(f"link {output}")
    result = run(command)
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout)

    print(f"metallib: {output}")
    print("next:")
    print(f"  {output.parent / 'QwenImageCLI'} --help")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-image-swift", type=Path, default=DEFAULT_QWEN_IMAGE_SWIFT)
    parser.add_argument("--mlx-swift-checkout", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--air-dir", type=Path)
    parser.add_argument("--metal-tool", type=Path)
    parser.add_argument("--metallib-tool", type=Path)
    parser.add_argument("--force", action="store_true", help="Replace an existing metallib")
    args = parser.parse_args()

    qwen_image_swift = args.qwen_image_swift.expanduser().resolve()
    mlx_swift_checkout = (
        args.mlx_swift_checkout.expanduser().resolve()
        if args.mlx_swift_checkout
        else qwen_image_swift / ".build" / "checkouts" / "mlx-swift"
    )
    release_dir = qwen_image_swift / ".build" / "release"
    output = args.output.expanduser().resolve() if args.output else release_dir / "mlx.metallib"
    air_dir = args.air_dir.expanduser().resolve() if args.air_dir else qwen_image_swift / ".build" / "mlx-metallib-air"
    metal_tool = args.metal_tool.expanduser().resolve() if args.metal_tool else find_tool("metal")
    metallib_tool = args.metallib_tool.expanduser().resolve() if args.metallib_tool else find_tool("metallib")

    build_metallib(
        mlx_swift_checkout=mlx_swift_checkout,
        output=output,
        air_dir=air_dir,
        metal_tool=metal_tool,
        metallib_tool=metallib_tool,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
