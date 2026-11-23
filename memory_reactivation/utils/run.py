"""实验运行身份、输出目录和可复现快照管理。

训练启动时会先把“静态配置”解析为带 task、lag、模型模式和 seed 的唯一运行
身份，再创建时间戳目录。运行目录同时保存输入配置、实际命令、Python 解释器、
源码快照，以及 manifest/split/normalization 三个小型数据描述文件。训练随后
改用这些快照路径，避免长任务运行期间外部文件被覆盖而无法审计。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence


SNAPSHOT_IGNORE_PATTERNS = (
    # 只快照可执行源码和配置；历史结果材料不属于训练实现，复制它们会让每个
    # run 携带近百 MB 的旧 CSV/日志，且不会提高当前 run 的可复现性。
    ".git",
    ".pytest_cache",
    "__pycache__",
    "*.pyc",
    "*.orig",
    "*.rej",
    "artifacts",
    "valid_format",
)


def data_artifact_keys(data: Mapping[str, object]) -> tuple[str, ...]:
    keys = ("manifest_csv", "split_csv", "normalization_json")
    return keys + (("graph_laplacian_json",) if data.get("graph_laplacian_json") else ())


@dataclass(frozen=True)
class PreparedRun:
    """训练开始前准备好的运行目录信息。

    output_dir 是本次运行的唯一根目录；timestamp 是目录名使用的本地时间；
    started_at 是带时区的 ISO 时间；snapshot_dir 指向该次运行冻结的源码副本。
    """

    output_dir: Path
    timestamp: str
    started_at: str
    snapshot_dir: Path


def create_timestamped_output_dir(base_output_dir: Path, timestamp: str) -> Path:
    """原子创建带时间戳的唯一运行目录。

    base_output_dir 本身是逻辑前缀，实际目录为“前缀_时间戳”。若同一秒并行
    启动多个任务，则尝试 _01、_02 等后缀；直接依赖 mkdir 的 FileExistsError
    判定竞争结果，因此不会出现先检查后创建的竞态窗口。
    """
    base_output_dir = base_output_dir.expanduser()
    base_output_dir.parent.mkdir(parents=True, exist_ok=True)
    stem = f"{base_output_dir.name}_{timestamp}"
    for collision_index in range(1000):
        suffix = "" if collision_index == 0 else f"_{collision_index:02d}"  # 同秒并行时加后缀
        candidate = base_output_dir.parent / f"{stem}{suffix}"
        try:
            candidate.mkdir()  # 原子占位，避免 check-then-create 竞态
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"Could not create a unique run directory for {base_output_dir}")


def snapshot_project_source(project_root: Path, output_dir: Path) -> Path:
    """把当前项目树复制到运行目录的 code_snapshot。

    忽略规则只排除 Git 元数据、解释器/测试缓存、可重建 artifacts 及历史结果
    材料。返回的是快照目录而非源目录，供运行元数据和后续审计引用。
    """
    snapshot_dir = output_dir / "code_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    for name in ("memory_reactivation", "configs", "scripts"):
        source = project_root / name
        if source.is_dir():
            shutil.copytree(source, snapshot_dir / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    metadata = project_root / "pyproject.toml"
    if metadata.is_file():
        shutil.copy2(metadata, snapshot_dir / metadata.name)
    return snapshot_dir


def prepare_run_directory(
    base_output_dir: Path,
    project_root: Path,
    source_config: Path,
    command: Sequence[str] | None = None,
    now: datetime | None = None,
) -> PreparedRun:
    """在读取数据和构建模型前冻结本次运行的入口状态。

    处理顺序是：生成带时区时间、创建唯一目录、复制项目源码、复制原始输入
    配置，再写 run_metadata.json。metadata 中 command 默认取 sys.argv，也可
    由调用方显式传入，便于测试或包装脚本保存真实调用参数。
    """
    started = now if now is not None else datetime.now().astimezone()
    if started.tzinfo is None:
        started = started.astimezone()
    timestamp = started.strftime("%Y%m%d_%H%M%S")
    output_dir = create_timestamped_output_dir(base_output_dir, timestamp)
    skip_code_snapshot = os.environ.get("PB2023_SKIP_CODE_SNAPSHOT", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if skip_code_snapshot:
        # High-concurrency launches can otherwise make dozens of simultaneous
        # recursive NFS copies before a single model reaches CUDA.  Keep a
        # marker and metadata so this deliberate trade-off is visible; normal
        # runs still take the complete source snapshot.
        snapshot_dir = output_dir / "code_snapshot"
        snapshot_dir.mkdir()
        (snapshot_dir / "SNAPSHOT_SKIPPED.txt").write_text(
            "Code snapshot skipped by PB2023_SKIP_CODE_SNAPSHOT=1 for a high-concurrency run.\n"
            f"project_root={project_root.resolve()}\n"
        )
    else:
        snapshot_dir = snapshot_project_source(project_root.resolve(), output_dir)

    source_config = source_config.resolve()
    shutil.copy2(source_config, output_dir / "input_config.yaml")  # 用户原始 YAML
    metadata = {
        "started_at": started.isoformat(),
        "timestamp": timestamp,
        "base_output_dir": str(base_output_dir.expanduser().resolve()),
        "output_dir": str(output_dir.resolve()),
        "project_root": str(project_root.resolve()),
        "source_config": str(source_config),
        "python": sys.executable,
        "command": list(command if command is not None else sys.argv),
        "code_snapshot_skipped": skip_code_snapshot,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return PreparedRun(
        output_dir=output_dir,
        timestamp=timestamp,
        started_at=metadata["started_at"],
        snapshot_dir=snapshot_dir,
    )


def _run_name_component(value: object, name: str) -> str:
    """把一个运行身份字段规范成稳定、适合路径和平台展示的短横线名称。"""
    normalized = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    if not normalized:
        raise ValueError(f"{name} must contain at least one letter or number")
    return normalized


def _string_sequence(value: object, name: str) -> list[str]:
    """校验标签配置必须是非字符串序列，并逐项转换为字符串。"""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return [str(item) for item in value]


def _is_dynamic_run_tag(tag: str) -> bool:
    """判断标签是否由本次 task/lag/model/seed 动态身份生成。

    重新配置同一个字典时会剔除旧动态标签，再加入当前身份，避免配置从 pitch
    切到 choice 后仍残留 pitch、旧 lag 或旧 seed 标签；用户静态标签则保留。
    """
    normalized = str(tag).strip().lower().replace("_", "-")
    reserved = {
        "target",
        "time-only",
        "pitch",
        "choice",
        "motor",
        "attention",
        "no-attention",
        "scorer",
        "query",
        "gated-scorer",
        "gated-query",
        "fusion",
        "no-fusion",
        "current-trial",
    }
    return (
        normalized in reserved
        or normalized.startswith("previous-trial-lag")
        or re.fullmatch(r"cv-fold[0-9]+-of-[0-9]+", normalized) is not None
        or re.fullmatch(r"(?:lag|seed)[0-9]+", normalized) is not None
        or normalized == "graph-laplacian"
        or normalized.startswith("graph-laplacian-")
    )


def configure_runtime_metadata(
    config: MutableMapping[str, object],
    *,
    mode: str,
    task: str,
    lag: int,
    cv_fold: int | None = None,
    cv_folds: int | None = None,
) -> Path:
    """解析一次参数化运行的输出路径与 SwanLab 身份。

    函数会原地补全 config：写入 run_mode、task、lag、output_root，生成包含
    模式/任务/lag/注意力/融合/seed 的 experiment_name，并同步 run 与 swanlab
    子映射。CV 的各个 fold 共享同一个 experiment_name 目录；fold 号只进入
    ``run-cv-foldN-of-K`` 运行子目录、动态标签和配置/checkpoint 身份。返回路径
    固定落在 base_output_dir / experiment_name / run[-cv-foldN-of-K]，最后一级
    会由时间戳函数扩展。

    旧的动态 SwanLab 标签会被清理，base_tags 和非保留标签继续保留；这样同一
    模板配置可安全用于多个 task/lag，而不会把实验身份混在一起。
    """
    normalized_mode = str(mode).lower().replace("_", "-")
    if normalized_mode not in {"target", "time-only"}:
        raise ValueError("mode must be target or time-only")
    lag_value = int(lag)
    if lag_value < 0:
        raise ValueError("lag must be non-negative")

    raw_run = config.get("run", {})
    if raw_run is None:
        raw_run = {}
    if not isinstance(raw_run, Mapping):
        raise ValueError("run must be a mapping")
    run_config = dict(raw_run)

    raw_logger = config.get("swanlab", {})
    if raw_logger is None:
        raw_logger = {}
    if not isinstance(raw_logger, Mapping):
        raise ValueError("swanlab must be a mapping")
    logger_config = dict(raw_logger)

    base_output_value = run_config.get(
        "base_output_dir",
        config.get("output_root", config.get("output_dir")),  # 兼容旧键
    )
    if base_output_value is None or not str(base_output_value).strip():
        raise ValueError("config must define run.base_output_dir or output_root")
    base_output_dir = Path(str(base_output_value)).expanduser().resolve()

    base_name_value = run_config.get(
        "base_experiment_name",
        logger_config.get("base_experiment_name", logger_config.get("experiment_name")),
    )
    base_name = (
        None
        if base_name_value is None
        else _run_name_component(base_name_value, "base_experiment_name")
    )
    task_name = _run_name_component(task, "task")
    history_tag = "current-trial" if lag_value == 0 else f"previous-trial-lag{lag_value}"

    raw_model = config.get("model", {})
    if raw_model is None:
        raw_model = {}
    if not isinstance(raw_model, Mapping):
        raise ValueError("model must be a mapping")
    raw_fusion = raw_model.get("fusion", {})
    if raw_fusion is None:
        raw_fusion = {}
    if not isinstance(raw_fusion, Mapping):
        raise ValueError("model.fusion must be a mapping")
    fusion_tag = "fusion" if bool(raw_fusion.get("enabled", False)) else "no-fusion"

    if cv_fold is not None:
        cv_fold = int(cv_fold)
        cv_folds = 5 if cv_folds is None else int(cv_folds)
        if cv_folds < 2 or cv_fold < 0 or cv_fold >= cv_folds:
            raise ValueError("cv_fold must be in [0, cv_folds - 1]")

    # 实验目录名和 SwanLab 实验名不包含 fold；所有 fold 共享同一实验目录。
    # fold 仍保留在 tags、config_snapshot 和 checkpoint 中，并单独进入运行子目录。
    identity_parts = [normalized_mode, task_name, f"lag{lag_value}"]
    dynamic_tags = [normalized_mode, task_name, f"lag{lag_value}", history_tag]
    if config.get("cv_strategy", (config.get("data") or {}).get("cv_strategy", "block")) == "subject":
        identity_parts.append("subject-cv")
        dynamic_tags.append("subject-cv")
    fold_run_suffix: str | None = None
    if cv_fold is not None:
        fold_run_suffix = f"cv-fold{cv_fold}-of-{cv_folds}"
        dynamic_tags.append(fold_run_suffix)
    if normalized_mode == "target":
        attention_mode = _run_name_component(
            raw_model.get("attention_mode", "scorer"),
            "model.attention_mode",
        )
        identity_parts.append(attention_mode)
        dynamic_tags.extend(["attention", attention_mode])
    identity_parts.append(fusion_tag)
    dynamic_tags.append(fusion_tag)
    graph_sha = (config.get("data") or {}).get("graph_laplacian_sha256")
    if graph_sha:
        from memory_reactivation.data.spatial import normalize_graph_input_mode
        graph_mode = normalize_graph_input_mode(
            (config.get("data") or {}).get("graph_laplacian_mode")
        )
        # Keep old replace-mode names stable; concat is a different experiment
        # even when the base name and the graph matrix are the same.
        graph_tag = "graph-laplacian-concat" if graph_mode == "concat" else "graph-laplacian"
        identity_parts.append(f"{graph_tag}-{str(graph_sha)[:12]}")
        dynamic_tags.append("graph-laplacian")
        if graph_mode == "concat":
            dynamic_tags.append(graph_tag)

    seed = int(config.get("seed", 20260530))
    seed_tag = f"seed{seed}"
    identity_parts.append(seed_tag)
    dynamic_tags.append(seed_tag)
    name_suffix = "-".join(identity_parts)
    experiment_name = f"{base_name}-{name_suffix}" if base_name else name_suffix
    group_suffix = "-".join(identity_parts[:3])
    default_group = f"{base_name}-{group_suffix}" if base_name else group_suffix

    config["run_mode"] = normalized_mode
    config["task"] = str(task)
    config["lag"] = lag_value
    if cv_fold is not None:
        config["cv_fold"] = cv_fold
        config["cv_folds"] = cv_folds
    config["output_root"] = str(base_output_dir)
    run_config.update(
        {
            "base_output_dir": str(base_output_dir),
            "name_suffix": name_suffix,
            "experiment_name": experiment_name,
        }
    )
    if base_name is not None:
        run_config["base_experiment_name"] = base_name
    config["run"] = run_config

    base_tags = _string_sequence(logger_config.get("base_tags", []), "swanlab.base_tags")
    configured_tags = _string_sequence(logger_config.get("tags", []), "swanlab.tags")
    # 丢掉旧动态标签，避免 pitch→choice 残留
    static_tags = [
        tag
        for tag in [*base_tags, *configured_tags]
        if not _is_dynamic_run_tag(tag)
    ]
    tags = list(dict.fromkeys([*static_tags, *dynamic_tags]))
    logger_config["base_tags"] = base_tags
    if base_name is not None:
        logger_config["base_experiment_name"] = base_name
    logger_config["experiment_name"] = experiment_name
    logger_config["group"] = logger_config.get("group") or default_group
    logger_config["tags"] = tags
    logger_config["description"] = logger_config.get("description") or (
        f"PB2023 {experiment_name} ({history_tag})."
    )
    config["swanlab"] = logger_config
    run_stem = "run" if fold_run_suffix is None else f"run-{fold_run_suffix}"
    return base_output_dir / experiment_name / run_stem  # 上层再加时间戳


def resolve_snapshot_data_artifacts(
    data: Mapping[str, object], run_dir: Path,
) -> dict[str, object]:
    """Prefer this run's frozen artifacts, including after a move or copy.

    Keep legacy runs without local copies usable via their configured paths.
    Callers must validate the resolved files against artifact_sources.json.
    """
    resolved = dict(data)
    artifact_dir = Path(run_dir).resolve() / "artifacts"
    for key in data_artifact_keys(data):
        if data.get(key) is None:
            raise ValueError(f"Snapshot config is missing data.{key}")
        configured = Path(str(data[key])).expanduser()
        candidates = (
            artifact_dir / f"{key}{''.join(configured.suffixes)}",
            artifact_dir / configured.name,
            configured,
        )
        path = next((path for path in candidates if path.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Snapshot artifact {key} not found: {candidates}")
        resolved[key] = str(path.resolve())
    return resolved


def validate_snapshot_data_artifacts(
    data: Mapping[str, object], run_dir: Path, *, required: bool = False,
    expected_split_sha256: str | None = None,
) -> None:
    """Check the actual inputs against hashes frozen when the run was created.

    Older fixed-split runs may predate artifact_sources.json. CV runs require it;
    otherwise a replaced normalization or manifest could silently change a fold.
    """
    sources_path = Path(run_dir) / "artifacts" / "artifact_sources.json"
    required = required or bool(data.get("graph_laplacian_json"))
    if not sources_path.is_file():
        if required:
            raise FileNotFoundError(f"Missing snapshot artifact hashes: {sources_path}")
        from .source_files import warn_legacy_source_files
        warn_legacy_source_files()
        return
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    if not isinstance(sources, Mapping):
        raise ValueError(f"Invalid snapshot artifact hashes: {sources_path}")
    if bool(sources.get("graph_laplacian_json")) != bool(data.get("graph_laplacian_json")):
        raise ValueError("Snapshot graph Laplacian preprocessing mismatch")
    for key in data_artifact_keys(data):
        entry = sources.get(key)
        if not isinstance(entry, Mapping) or not entry.get("sha256"):
            raise ValueError(f"Missing snapshot artifact hash: {key}")
        path = Path(str(data[key])).expanduser().resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Snapshot artifact changed: {key} ({path})")
        if key == "split_csv" and expected_split_sha256 is not None and digest != expected_split_sha256:
            raise ValueError("Snapshot split does not match the saved CV fold")
        if key == "graph_laplacian_json" and digest != data.get("graph_laplacian_sha256"):
            raise ValueError("Snapshot graph Laplacian identity mismatch")
    from .source_files import validate_record_files
    validate_record_files(Path(str(data["manifest_csv"])), sources["manifest_csv"])


def snapshot_data_artifacts(
    config: MutableMapping[str, object],
    output_dir: Path,
) -> dict[str, dict[str, object]]:
    """快照关键数据描述文件，并把运行配置重定向到快照。

    必须存在 manifest_csv、split_csv、normalization_json；启用图拉普拉斯时还会
    快照 graph_laplacian_json。每个源文件复制到 output_dir/artifacts 后计算 SHA-256，artifact_sources.json
    同时记录原路径、快照绝对路径和摘要。config["data"] 会原地改为快照路径，
    因而后续 Dataset 与运行审计看到的是完全相同的输入描述。

    这里只复制体积小且决定样本选择/归一化的数据协议文件，不复制大型 EEG
    NPZ；其内容与行为 CSV 内容通过 manifest_csv 条目内的 SHA-256 绑定。
    历史快照没有该扩展时仍可读取，但恢复校验会明确提示无法验证原始数据身份。
    """
    raw_data = config.get("data")
    if not isinstance(raw_data, Mapping):
        raise ValueError("data must be a mapping")
    data = dict(raw_data)
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, object]] = {}
    # 只快照协议小文件；EEG NPZ 仍走 manifest 里的路径
    for key in data_artifact_keys(data):
        if key not in data:
            raise ValueError(f"data.{key} is required")
        source = Path(str(data[key])).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        suffix = "".join(source.suffixes)  # 保留 .json.gz 等复合后缀
        destination = artifact_dir / f"{key}{suffix}"
        shutil.copy2(source, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if key == "split_csv" and config.get("cv_split_sha256") is not None:
            if digest != config["cv_split_sha256"]:
                raise ValueError("CV split changed while creating the run snapshot")
        if key == "graph_laplacian_json" and digest != data.get("graph_laplacian_sha256"):
            raise ValueError("Graph Laplacian changed while creating the run snapshot")
        manifest[key] = {
            "source": str(source),
            "snapshot": str(destination.resolve()),
            "sha256": digest,
        }
        data[key] = str(destination.resolve())  # 训练后只读快照，防外部文件被改
    from .source_files import snapshot_record_files
    manifest["manifest_csv"].update(
        record_files_schema=1,
        record_files=snapshot_record_files(Path(str(data["manifest_csv"]))),
    )
    config["data"] = data
    with (artifact_dir / "artifact_sources.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
