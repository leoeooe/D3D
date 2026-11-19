"""Device selection for training and layer pruning."""

import os

import torch


def default_device_type() -> str:
    from accelerate.utils import is_npu_available

    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or is_npu_available():
        return "npu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def bind_device(name: str, *, local_world_size: int = 1) -> torch.device:
    if name.split(":")[0] == "npu":
        try:
            import torch_npu  # noqa: F401 - registers the npu device and HCCL
        except ImportError as exc:
            raise RuntimeError(
                "Ascend training requires torch_npu and the CANN environment"
            ) from exc

    device = torch.device(name)
    if device.type not in {"cuda", "npu"}:
        return device
    backend = getattr(torch, device.type)
    count = backend.device_count() if backend.is_available() else 0
    index = device.index if device.index is not None else 0
    if index >= count or local_world_size > count:
        raise ValueError(
            f"Requested {device.type}:{index} with {local_world_size} local rank(s), "
            f"but only {count} {device.type} device(s) are visible"
        )
    device = torch.device(device.type, index)
    backend.set_device(device)
    return device


def training_accelerator(requested: str, **kwargs):
    from accelerate import Accelerator
    from accelerate.utils import parse_flag_from_env

    if requested == "auto":
        if parse_flag_from_env("ACCELERATE_USE_CPU") or default_device_type() != "npu":
            return Accelerator(**kwargs)
        requested = "npu"
    if requested not in {"cuda", "npu"}:
        raise ValueError(f"Unknown training device: {requested}")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = bind_device(
        f"{requested}:{local_rank}",
        local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", 1)),
    )
    accelerator = Accelerator(**kwargs)
    actual = accelerator.device
    if actual.type != device.type or actual.index not in (None, device.index):
        raise RuntimeError(f"Requested {device} but Accelerate selected {actual}")
    return accelerator


def training_attention(requested: str, device_type: str) -> str:
    if requested == "auto":
        return "sdpa" if device_type == "npu" else "flash_attention_2"
    if device_type == "npu" and requested != "sdpa":
        raise ValueError("Ascend training supports --attn-implementation sdpa")
    return requested
