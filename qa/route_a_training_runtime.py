"""Bounded Torch CPU parallelism shared by Route A research runners."""
from __future__ import annotations

import os
from typing import Any


def configure_torch_threads(torch_module: Any) -> dict[str, int]:
    """Honor an explicit CPU thread budget; otherwise use up to 8 cores.

    Small Route A recurrent models benchmark faster at 8 threads than at 16;
    callers can override this with ROUTE_A_TORCH_NUM_THREADS.
    """
    cpu_count = os.cpu_count() or 1
    default_threads = max(1, min(8, cpu_count))
    threads = int(os.environ.get("ROUTE_A_TORCH_NUM_THREADS", str(default_threads)))
    if not 1 <= threads <= cpu_count:
        raise ValueError("ROUTE_A_TORCH_NUM_THREADS is outside the available CPU range")
    default_interop = max(1, min(4, threads))
    interop = int(os.environ.get(
        "ROUTE_A_TORCH_INTEROP_THREADS", str(default_interop)
    ))
    if not 1 <= interop <= cpu_count:
        raise ValueError("ROUTE_A_TORCH_INTEROP_THREADS is outside the available CPU range")
    torch_module.set_num_threads(threads)
    try:
        torch_module.set_num_interop_threads(interop)
    except RuntimeError:
        # Torch inter-op pools can only be configured before first use.
        pass
    return {
        "cpu_threads": torch_module.get_num_threads(),
        "interop_threads": torch_module.get_num_interop_threads(),
        "logical_cpu_count": cpu_count,
    }
