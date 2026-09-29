#!/usr/bin/env python3
import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizeEcho(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, dim=1)


def fail(message):
    print(f"[FAIL] {message}")
    return False


def ok(message):
    print(f"[ OK ] {message}")
    return True


def parse_devices(value, device_count):
    if ":" in value:
        start_text, end_text = value.split(":", 1)
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else device_count
        devices = list(range(start, end))
    else:
        devices = [int(part) for part in value.split(",") if part.strip()]

    if not devices:
        raise ValueError("device list is empty")
    if min(devices) < 0 or max(devices) >= device_count:
        raise ValueError(f"requested devices {devices}, but only {device_count} CUDA devices exist")
    if len(set(devices)) != len(devices):
        raise ValueError(f"duplicate devices are not allowed: {devices}")
    return devices


def check_direct_copies(devices, rows, dim, dtype):
    print("\n== Direct GPU copy checks ==")
    passed = True
    output_device = devices[0]
    base = torch.randn(rows, dim, device=f"cuda:{output_device}", dtype=dtype)

    for src in devices[1:]:
        src_tensor = base.to(f"cuda:{src}")
        copied = src_tensor.to(f"cuda:{output_device}")
        torch.cuda.synchronize(output_device)
        torch.cuda.synchronize(src)

        same = torch.equal(base, copied)
        max_abs = (base - copied).abs().max().item()
        if same:
            passed &= ok(f"cuda:{src} -> cuda:{output_device} exact copy match")
        else:
            passed &= fail(
                f"cuda:{src} -> cuda:{output_device} copy mismatch, max_abs_diff={max_abs:g}"
            )

    return passed


def check_dataparallel(devices, rows_per_gpu, dim, dtype):
    print("\n== nn.DataParallel gather check ==")
    output_device = devices[0]
    total_rows = rows_per_gpu * len(devices)
    x = torch.randn(total_rows, dim, device=f"cuda:{output_device}", dtype=dtype)
    expected = F.normalize(x, dim=1)

    model = NormalizeEcho(dim).cuda(output_device)
    dp = nn.DataParallel(model, device_ids=devices, output_device=output_device)

    with torch.no_grad():
        actual = dp(x)
    torch.cuda.synchronize()

    passed = True
    for chunk_index, gpu in enumerate(devices):
        start = chunk_index * rows_per_gpu
        end = start + rows_per_gpu
        chunk = actual[start:end]
        ref = expected[start:end]

        norms = chunk.norm(dim=1)
        max_abs = (chunk - ref).abs().max().item()
        finite = torch.isfinite(chunk).all().item()
        close = torch.allclose(chunk, ref, atol=1e-6, rtol=1e-5)

        if finite and close:
            passed &= ok(
                f"DataParallel chunk from cuda:{gpu} matches; "
                f"norm min/max={norms.min().item():.6f}/{norms.max().item():.6f}"
            )
        else:
            passed &= fail(
                f"DataParallel chunk from cuda:{gpu} corrupted; "
                f"finite={finite}, max_abs_diff={max_abs:g}, "
                f"norm min/max={norms.min().item():.6f}/{norms.max().item():.6f}"
            )

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Check whether CUDA peer copies and nn.DataParallel gather are reliable."
    )
    parser.add_argument("--rows-per-gpu", type=int, default=8)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument(
        "--devices",
        default=None,
        help="CUDA devices to test, e.g. '0,1,2,3' or Python-style range '1:5'. "
        "Default: all devices.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[FAIL] CUDA is not available")
        return 1

    num_gpus = torch.cuda.device_count()
    print(f"torch: {torch.__version__}")
    print(f"cuda build: {torch.version.cuda}")
    print(f"gpu count: {num_gpus}")
    for i in range(num_gpus):
        print(f"cuda:{i}: {torch.cuda.get_device_name(i)}")

    try:
        devices = parse_devices(args.devices or f"0:{num_gpus}", num_gpus)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 1

    if num_gpus < 2:
        print("[FAIL] Need at least 2 GPUs to test nn.DataParallel gather/copies")
        return 1
    if len(devices) < 2:
        print("[FAIL] Need at least 2 selected GPUs to test nn.DataParallel gather/copies")
        return 1

    print(f"selected devices: {devices}")
    print(f"gather/output device: cuda:{devices[0]}")

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    copy_ok = check_direct_copies(devices, args.rows_per_gpu, args.dim, dtype)
    dp_ok = check_dataparallel(devices, args.rows_per_gpu, args.dim, dtype)

    print("\n== Result ==")
    if copy_ok and dp_ok:
        print("[ OK ] CUDA peer copies and nn.DataParallel gather look healthy")
        return 0

    print("[FAIL] Multi-GPU copies/gather are not reliable in this environment")
    return 2


if __name__ == "__main__":
    sys.exit(main())
