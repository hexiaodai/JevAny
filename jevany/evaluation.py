"""独立分片评测协议：稳定分组、原子提交和全局报告汇总。"""

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from .data import api_request
from .model import ContextLengthError
from .suite import ENCODING, digest, record_digest, write_json


PROTOCOL_VERSION = 1


def distributed_group_slice(records, index, count):
    """按完整样本组稳定分片，保持对比和排列样本在同一分片。"""
    slots = {}
    for record in records:
        group = record["_meta"]["group_id"]
        if group not in slots:
            slots[group] = len(slots) % count
    return [record for record in records if slots[record["_meta"]["group_id"]] == index]


def execution_identity(args, records, source_hash):
    """固定数据、检查点内容和评测选项，防止不同实验的分片被混合。"""
    run = Path(args.run)
    if not run.is_dir():
        raise ValueError("sharded evaluation requires a local, immutable checkpoint directory")
    suffixes = {".json", ".safetensors", ".pt", ".txt", ".jinja", ".model", ".tiktoken"}
    files = {path.name: digest(path) for path in sorted(run.iterdir()) if path.is_file() and path.suffix in suffixes}
    if not files:
        raise ValueError("checkpoint directory contains no model files")
    # 所有节点共享只读基础模型 Dataset，配置文件摘要额外校验模型配置。
    base = os.environ.get("JEVANY_BASE_LOAD_PATH")
    base_config = digest(Path(base) / "config.json") if base else None
    options = {key: value for key, value in vars(args).items()
               if key not in {"aggregate", "shard_index", "out", "execution_id", "shard_wait_seconds", "device"}}
    environment = {key: value for key, value in os.environ.items() if key.startswith("JEVANY_") and "KEY" not in key}
    value = {"version": PROTOCOL_VERSION, "suite_sha256": source_hash,
             "records_sha256": record_digest(records), "model": files, "base_config_sha256": base_config,
             "options": options, "environment": environment}
    value["implementation"] = {name: digest(Path(__file__).with_name(name)) for name in ("benchmark.py", "evaluation.py", "metrics.py")}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def execution_directory(output, execution_id):
    """执行 ID 作为单级目录，隔离不同 JobSet 实例和用户重试。"""
    if not execution_id or execution_id in {".", ".."} or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in execution_id):
        raise ValueError("execution ID must contain only letters, digits, hyphens and underscores")
    return Path(output) / "shards" / execution_id


def commit_json(path, value):
    """在同一文件系统内先完成写入，再原子替换提交清单。"""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding=ENCODING) as stream:
        json.dump(value, stream, allow_nan=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def evaluate_shard(records, predictor, args, identity, metadata):
    """逐条同步预测，重启校验身份并复用完整提交的记录。"""
    import fcntl
    from .benchmark import prediction_rows

    directory = execution_directory(args.out, args.execution_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{args.shard_index}-checkpoint.jsonl"
    manifest_path = directory / f"{args.shard_index}-checkpoint.json"
    shard = distributed_group_slice(records, args.shard_index, args.shard_count)
    expected = {"version": PROTOCOL_VERSION, "execution_id": args.execution_id, "identity": identity,
                "shard_index": args.shard_index, "shard_count": args.shard_count}
    with (directory / f"{args.shard_index}.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != expected:
                raise ValueError("evaluation checkpoint identity mismatch")
        else:
            if path.exists():
                raise ValueError("evaluation journal has no execution identity")
            commit_json(manifest_path, expected)
        committed = 0
        with path.open("a+b") as stream:
            stream.seek(0)
            while line := stream.readline():
                # 只回收末尾未提交的半行；完整但损坏的记录必须报错。
                if not line.endswith(b"\n"):
                    stream.seek(-len(line), os.SEEK_CUR)
                    stream.truncate()
                    break
                item = json.loads(line)
                if committed >= len(shard):
                    raise ValueError("evaluation checkpoint exceeds shard coverage")
                record = shard[committed]
                if item["id"] != record["_meta"]["id"] or item["request_sha256"] != record_digest(api_request(record)):
                    raise ValueError("evaluation checkpoint record mismatch")
                if ("prediction" in item) == ("context_error" in item):
                    raise ValueError("invalid evaluation checkpoint result")
                committed += 1
            stream.seek(0, os.SEEK_END)
            for record in shard[committed:]:
                item = {"id": record["_meta"]["id"], "request_sha256": record_digest(api_request(record))}
                try:
                    prediction = predictor(record)
                    prediction_rows(record, prediction)
                    item["prediction"] = prediction
                except ContextLengthError as error:
                    if not args.data:
                        raise
                    item["context_error"] = str(error)
                stream.write((json.dumps(item, allow_nan=False) + "\n").encode(ENCODING))
                stream.flush()
                os.fsync(stream.fileno())
        manifest = {**expected, "records": len(shard), "file": path.name, "sha256": digest(path),
                    "temperature": predictor.temperature, "metadata": metadata}
        commit_json(directory / f"{args.shard_index}.json", manifest)
        print(json.dumps({"shard_index": args.shard_index, "records": len(shard),
                          "reused": committed, "committed": True}), flush=True)


def read_committed_shard(directory, index, deadline):
    """按确定路径读取清单和产物，有限等待 NFS 跨客户端可见性。"""
    while True:
        try:
            manifest = json.loads((directory / f"{index}.json").read_text(encoding=ENCODING))
            filename = manifest["file"]
            if Path(filename).name != filename or not filename.endswith(".jsonl"):
                raise ValueError("invalid shard artifact path")
            data = (directory / filename).read_bytes()
            if hashlib.sha256(data).hexdigest() != manifest["sha256"]:
                raise ValueError(f"shard {index} checksum mismatch")
            return manifest, data
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"committed shard {index} is not visible") from None
            time.sleep(min(1, max(0, deadline - time.monotonic())))


class CachedPredictor:
    """从已验证分片读取原始预测，保留推理端记录的真实延迟。"""

    def __init__(self, predictions, temperature):
        self.predictions = predictions
        self.temperature = temperature

    def __call__(self, record):
        item = self.predictions[record["_meta"]["id"]]
        if "context_error" in item:
            raise ContextLengthError(item["context_error"])
        return item["prediction"]


def aggregate_shards(records, args, identity, heldout_sources):
    """校验所有分片覆盖和配置，再复用单机指标逻辑发布唯一报告。"""
    from .benchmark import evaluate_records

    directory = execution_directory(args.out, args.execution_id)
    predictions = {}
    metadata = temperature = None
    deadline = time.monotonic() + args.shard_wait_seconds
    for index in range(args.shard_count):
        manifest, data = read_committed_shard(directory, index, deadline)
        expected = {"version": PROTOCOL_VERSION, "execution_id": args.execution_id, "identity": identity,
                    "shard_index": index, "shard_count": args.shard_count}
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(f"shard {index} execution identity mismatch")
        if metadata is None:
            metadata, temperature = manifest["metadata"], manifest["temperature"]
        elif manifest["metadata"] != metadata or manifest["temperature"] != temperature:
            raise ValueError(f"shard {index} model metadata mismatch")
        shard = {r["_meta"]["id"]: r for r in distributed_group_slice(records, index, args.shard_count)}
        seen = set()
        for line in data.splitlines():
            item = json.loads(line)
            identifier = item["id"]
            if identifier not in shard or identifier in predictions:
                raise ValueError(f"unexpected or duplicate evaluation record: {identifier}")
            if item["request_sha256"] != record_digest(api_request(shard[identifier])):
                raise ValueError(f"request digest mismatch: {identifier}")
            if ("prediction" in item) == ("context_error" in item):
                raise ValueError(f"invalid prediction result: {identifier}")
            predictions[identifier] = item
            seen.add(identifier)
        if seen != shard.keys() or len(seen) != manifest["records"]:
            raise ValueError(f"shard {index} coverage mismatch")
    if predictions.keys() != {r["_meta"]["id"] for r in records}:
        raise ValueError("evaluation coverage mismatch")

    final = Path(args.out) / "final"
    if final.exists():
        completed = json.loads((final / "manifest.json").read_text(encoding=ENCODING))
        if completed["execution_id"] != args.execution_id or completed["identity"] != identity:
            raise ValueError("output already contains another evaluation; use a new output directory")
        for name, checksum in completed["files"].items():
            if Path(name).name != name or digest(final / name) != checksum:
                raise ValueError("published report checksum mismatch")
        return json.loads((final / "report.json").read_text(encoding=ENCODING))

    staging = Path(args.out) / ".staging" / uuid.uuid4().hex
    report, _ = evaluate_records(records, CachedPredictor(predictions, temperature), staging,
                                 heldout_sources=heldout_sources, skip_overlong=bool(args.data))
    metadata = dict(metadata)
    report["calibration"].update(metadata.pop("calibration"))
    report.update(metadata)
    report["distributed"] = {"execution_id": args.execution_id, "shard_count": args.shard_count, "identity": identity}
    write_json(staging / "report.json", report)
    files = {path.name: digest(path) for path in staging.iterdir() if path.is_file()}
    write_json(staging / "manifest.json", {"execution_id": args.execution_id, "identity": identity, "files": files})
    os.rename(staging, final)
    return report
