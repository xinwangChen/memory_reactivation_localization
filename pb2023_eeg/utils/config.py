"""YAML 配置文件的最小读写接口。

训练入口先读取用户配置，补全运行时字段后再将最终配置写入运行目录。这里
只约束顶层必须是映射，不在工具层解释各业务字段；字段级校验由对应模型、
数据集或训练模块负责。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """读取一个 YAML 配置并返回顶层键值映射。

    yaml.safe_load 对空文件会返回 None，对列表式 YAML 会返回列表；两者都不是
    项目约定的配置 schema，因此在这里尽早拒绝。
    """
    # 兼容 str 类型路径
    path = Path(path) if not isinstance(path, Path) else path
    try:
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML config file {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")
    return config


def save_yaml(path: Path, config: Dict[str, Any]) -> None:
    """将已经解析并补全的运行时配置保存为 YAML 快照。

    保留字典插入顺序而不按键排序，使输出结构尽量贴近输入配置，方便人工
    对照；父目录会在写入前自动创建。
    """
    # 兼容 str 类型路径
    path = Path(path) if not isinstance(path, Path) else path
    if not isinstance(config, dict):
        raise ValueError(
            f"save_yaml expects a dict, got {type(config).__name__}. "
            "Only top-level mappings are supported."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
