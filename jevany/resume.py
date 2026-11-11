"""State for continuing a training schedule from an optimizer-step boundary."""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import uuid

import numpy as np
import torch

from .suite import digest, write_json

VERSION = 1
MUTABLE_SETTINGS = {
    "config", "dry_run", "out", "resume", "auto_resume", "base_load_path", "checkpoint_every_steps",
    "wandb_project", "wandb_name", "wandb_group", "wandb_entity", "wandb_mode",
}


def prepare_training_output(args, inputs, resources, rank, world_size):
    """主进程锁定输出并选择恢复点，所有 worker 复用同一尝试目录。"""
    import fcntl
    import torch.distributed as distributed
    from .evaluation import commit_json

    output = Path(args.out).resolve()
    state = output.parent / f".{output.name}.training"
    selection = [None]
    if rank == 0:
        try:
            state.mkdir(parents=True, exist_ok=True)
            lock = resources.enter_context((state / "writer.lock").open("a"))
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            configuration = {key: value for key, value in vars(args).items()
                             if key not in {"out", "config", "dry_run", "auto_resume"}}
            identity = {"args": configuration, "world_size": world_size,
                        "inputs": {name: dataset_fingerprint(records) for name, records in inputs.items()}}
            manifest = state / "execution.json"
            if manifest.exists():
                if json.loads(manifest.read_text()) != identity:
                    raise ValueError("training output belongs to a different configuration or dataset; use a new --out")
            else:
                if output.exists():
                    raise ValueError("--auto-resume requires a new output directory for its first execution")
                commit_json(manifest, identity)
            completed = next((path for path in sorted(state.glob("attempt-*"))
                              if (path / "training_metrics.json").is_file()), None)
            if completed is not None:
                selection[0] = {"attempt": str(completed), "completed": True}
            else:
                checkpoints = []
                # 最后一步已保存但最终发布中断时，同样恢复该步以完成发布。
                for attempt in state.glob("attempt-*"):
                    for directory in [attempt, *attempt.glob("checkpoints/step-*")]:
                        if not (directory / "trainer_state.json").is_file() or not (directory / "trainer_state.pt").is_file():
                            continue
                        header = read_resume_config(directory)
                        checkpoints.append((header["optimizer_step"], str(directory)))
                resume = max(checkpoints, default=(0, args.resume))[1]
                selection[0] = {"attempt": str(state / f"attempt-{uuid.uuid4().hex}"),
                                "resume": resume, "completed": False}
        except Exception as error:
            selection[0] = {"error": str(error)}
    if world_size > 1:
        device = torch.device("cuda", torch.cuda.current_device()) if distributed.get_backend() == "nccl" else None
        distributed.broadcast_object_list(selection, src=0, device=device)
    chosen = selection[0]
    if "error" in chosen:
        raise ValueError(chosen["error"])
    if rank == 0:
        print(json.dumps({"output": str(output), **chosen}), flush=True)
    return chosen


def publish_training_output(attempt, output):
    """完成后原子发布稳定模型目录，恢复状态与检查点保留在原尝试中。"""
    output = Path(output).resolve()
    if output.exists():
        return
    staging = output.parent / f".{output.name}-publish-{uuid.uuid4().hex}"
    shutil.copytree(attempt, staging)
    os.rename(staging, output)


def read_resume_config(directory: str | Path) -> dict:
    """Read a local training checkpoint's configuration without loading weights."""
    directory = Path(directory)
    filename = directory / "trainer_state.json"
    if not filename.is_file():
        raise ValueError(f"{directory}: no complete trainer state; use --init-from for a weights-only warm start")
    try:
        header = json.loads(filename.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"{filename}: invalid trainer state metadata") from error
    if header.get("version") != VERSION or not isinstance(header.get("args"), dict):
        raise ValueError(f"{filename}: unsupported trainer state format")
    return header


def validate_resume_settings(args: dict, saved: dict) -> None:
    """A resume continues the original schedule; recipe changes need a warm start."""
    changed = sorted(key for key in saved if key not in MUTABLE_SETTINGS and args.get(key) != saved[key])
    if changed:
        raise ValueError(f"--resume must keep the original training settings: {', '.join(changed)}; "
                         "use --init-from to start a different experiment")


def dataset_fingerprint(records: list[dict]) -> str:
    """Identify admitted records, labels, ordering and the local media they use."""
    fingerprint = hashlib.sha256()
    media = set()
    for record in records:
        # Question and option insertion order affects encoding and seeded
        # augmentation, so it must be part of the resume identity.
        fingerprint.update(json.dumps(record, ensure_ascii=False, allow_nan=False).encode())
        fingerprint.update(b"\n")
        for item in record.get("media", []):
            media.add(item["uri"])
    for uri in sorted(media):
        fingerprint.update(uri.encode())
        if "://" not in uri:
            fingerprint.update(digest(uri).encode())
    return fingerprint.hexdigest()


def capture_rng(shuffle: random.Random, device: str) -> dict:
    """Capture the random generators used by one rank after its optimizer step."""
    numpy = np.random.get_state()
    result = {
        "shuffle": shuffle.getstate(), "python": random.getstate(),
        "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]],
        "torch": torch.get_rng_state(),
    }
    if device == "cuda":
        result["device"] = torch.cuda.get_rng_state()
    elif device == "mps":
        result["device"] = torch.mps.get_rng_state()
    return result


def restore_rng(state: dict, shuffle: random.Random, device: str) -> None:
    """Restore generators after model/optimizer construction has consumed randomness."""
    shuffle.setstate(state["shuffle"])
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if device == "cuda":
        torch.cuda.set_rng_state(state["device"])
    elif device == "mps":
        torch.mps.set_rng_state(state["device"])


def save_trainer_state(directory: str | Path, state: dict) -> None:
    """Write state last, so an interrupted model save cannot appear resumable."""
    directory = Path(directory)
    state["version"] = VERSION
    state["checkpoint_sha256"] = {
        name: digest(directory / name) for name in ("adapter_model.safetensors", "head.pt")
    }
    temporary = directory / ".trainer_state.pt.tmp"
    torch.save(state, temporary)
    os.replace(temporary, directory / "trainer_state.pt")
    header = {key: state[key] for key in ("version", "args", "optimizer_step", "world_size", "dataset_sha256")}
    temporary = directory / ".trainer_state.json.tmp"
    write_json(temporary, header)
    os.replace(temporary, directory / "trainer_state.json")


def load_trainer_state(directory: str | Path, args: dict, dataset_sha256: str,
                       world_size: int, device: str) -> dict:
    """Load tensor/basic-type state and validate its data, schedule and weight files."""
    directory = Path(directory)
    header = read_resume_config(directory)
    validate_resume_settings(args, header["args"])
    state_file = directory / "trainer_state.pt"
    if not state_file.is_file():
        raise ValueError(f"{directory}: missing trainer_state.pt")
    state = torch.load(state_file, map_location="cpu", weights_only=True)
    if (state.get("version") != VERSION or state.get("args") != header["args"]
            or state.get("optimizer_step") != header["optimizer_step"]):
        raise ValueError(f"{directory}: trainer state and metadata disagree")
    if state["dataset_sha256"] != dataset_sha256:
        raise ValueError("--resume training data changed since this checkpoint "
                         "(record content/order, media paths or local media contents)")
    if state["world_size"] != world_size or state["device"] != device:
        raise ValueError("--resume requires the original world size and device type")
    required_rng = {"shuffle", "python", "numpy", "torch"}
    if device in ("cuda", "mps"):
        required_rng.add("device")
    rank_rng = state.get("rank_rng")
    if (not isinstance(rank_rng, list) or len(rank_rng) != world_size
            or any(not isinstance(rng, dict) or not required_rng <= rng.keys() for rng in rank_rng)):
        raise ValueError(f"{directory}: incomplete checkpoint RNG state; use a complete checkpoint "
                         "or --init-from for a weights-only warm start")
    for name, expected in state["checkpoint_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"{directory}: {name} differs from the saved trainer state")
    return state
