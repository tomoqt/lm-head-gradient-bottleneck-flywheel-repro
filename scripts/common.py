import json
import os
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class RunMeta:
    started_at_unix: float
    ended_at_unix: float
    duration_sec: float
    device: str
    seed: int


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def finalize_payload(payload: dict, start_time: float, device: str, seed: int) -> dict:
    end_time = time.time()
    payload["run_meta"] = asdict(
        RunMeta(
            started_at_unix=start_time,
            ended_at_unix=end_time,
            duration_sec=end_time - start_time,
            device=device,
            seed=seed,
        )
    )
    return payload
