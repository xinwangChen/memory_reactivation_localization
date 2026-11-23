"""PB2023 EEG 的样本索引、按需加载、条件平衡与 batch 采样。

本模块把两个坐标系保持为显式字段：row_index 是清洗后 NPZ 数组 X 的连续行号，
trial0 是原始实验 trial 编号。EEG 必须按 row_index 读取，行为标签和历史 trial
必须按 trial0 回查。第一阶段 EEGCleanDataset 只返回当前 trial 的干净 EEG；
第二阶段 LagDecodingDataset 用当前 trial EEG 预测当前或历史 lag 的行为标签。

NPZ 的 X 形状约定为 [trial, channel, time]，单样本 eeg 为 [channel, time]。
大型压缩 NPZ 由 SubjectArrayStore 按实验-被试记录惰性解压并用 LRU 缓存。
Choice 任务还支持条件内平衡、逆频率权重、逐 epoch 动态下采样和条件内标签
置乱；这些操作始终保留原始 EEG 定位信息，只改变选择、权重或监督标签。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .manifest import SubjectRecord, load_split_map, metadata_by_trial0
from .normalization import load_normalization_stats
from .offline_cache import load_offline_cache, offline_cache_path
from ..utils.source_files import SourceFileBinding
from .spatial import (
    GraphLaplacian,
    apply_graph_input,
    normalize_graph_input_mode,
)


# classes=元数据原始取值；Dataset 的 target 会映射成 0..K-1
TASK_SPECS = {
    "pitch": {"column": "pitch", "classes": [1, 2, 3, 4, 5], "task_id": 0},
    "choice": {"column": "choice", "classes": [0, 1], "task_id": 1},
    "motor": {"column": "motor", "classes": [1, 2], "task_id": 2},
}
DEFAULT_PITCH_CLASS_INDICES = tuple(
    range(len(TASK_SPECS["pitch"]["classes"]))
)


def normalize_lags(value: Sequence[int]) -> list[int]:
    """Validate lag values before conversion can change their label meaning."""
    if isinstance(value, (str, bytes)):
        raise ValueError("lags must contain non-negative integers")
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError("lags must contain non-negative integers") from exc
    if not values or any(isinstance(lag, (bool, np.bool_))
                         or not isinstance(lag, (int, np.integer)) or lag < 0
                         for lag in values):
        raise ValueError("lags must contain non-negative integers")
    return sorted({int(lag) for lag in values})


def normalize_pitch_class_indices(value: object) -> Tuple[int, ...]:
    """规范化需要保留的 Pitch 原始类别下标。

    配置使用模型侧的 ``0..4`` 下标，而元数据保存 ``1..5``。该字段表示一个
    类别集合，因此返回值固定升序；缺省保留全部五类。至少保留两个类别，避免
    构造退化的单类别分类任务。
    """
    if value is None:
        return DEFAULT_PITCH_CLASS_INDICES
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("pitch_class_indices must be a sequence of integers in [0, 4]")
    normalized: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise ValueError(
                "pitch_class_indices must contain only integers in [0, 4]"
            )
        normalized.append(int(item))
    if len(normalized) < 2:
        raise ValueError("pitch_class_indices must retain at least two classes")
    if len(set(normalized)) != len(normalized):
        raise ValueError("pitch_class_indices must not contain duplicates")
    if any(index not in DEFAULT_PITCH_CLASS_INDICES for index in normalized):
        raise ValueError("pitch_class_indices values must be between 0 and 4")
    return tuple(sorted(normalized))

# 兼容修复前已经生成的 NPZ。值为 (起始秒, 时间点数)；采样率仍从 NPZ 读取。
# 新 NPZ 会直接保存作者原始 Matdata.time，不再依赖这份实验特定回退表。
LEGACY_PB2023_TIME_AXES = {
    1: (-0.496, 275),
    2: (-0.996, 300),
}


@dataclass(frozen=True)
class EEGExample:
    """第一阶段一个干净 trial 的轻量索引。

    subject_index 实际指向 records 中的一条实验-被试记录，因而也用作 record_id；
    它不等于跨实验合并后的 subject_id。row_index 定位 NPZ 的 X 行，trial0 用于
    输出与原始元数据对齐，exp 记录该行来自哪个实验。对象本身不持有 EEG。
    """

    subject_index: int  # records 列表下标 = record_id（≠ subject_id）
    row_index: int  # NPZ["X"] 的行 0..n_clean-1，读 EEG 只用它
    trial0: int  # 原始实验编号 0..1999，查 meta/历史只用它
    exp: int  # 1 或 2


@dataclass(frozen=True)
class LagExample:
    """第二阶段一条“当前 EEG、指定 lag 标签”的轻量索引。

    row_index/trial0 始终定位当前 EEG；lag=0 时 target_index 来自当前 trial，
    lag>0 时来自 trial0-lag 的历史 trial。target_index 是连续类别下标。
    current_stimulus 固定为当前 trial 的 pitch；prev_* 是当前 trial 前一试次的
    nuisance 条件值，不随要预测的 lag 改变。choice_condition_values 按
    (current_pitch, 配置的历史字段...) 保存，用于 Choice 条件内采样和指标。

    source_target_index 仅在标签置乱控制中记录置乱前类别；sample_weight 是
    样本损失权重，默认 1，不参与 EEG 或标签索引。
    """

    subject_index: int  # = record_id，进 SubjectArrayStore 的键
    row_index: int  # 当前 trial 的 NPZ 行；EEG 永远是「当前」
    trial0: int  # 当前 trial 的原始编号
    exp: int  # 实验号 1/2
    lag: int  # 0→标签=当前；k→标签=meta[trial0-k]
    current_stimulus: int  # 当前 pitch∈{1..5}，控制变量，不是 target
    target_index: int  # 监督 y：pitch 0..4 / choice 0..1 / motor 0..1
    prev_pitch: int = -1  # trial0-1 的 pitch；-1=没有前试
    prev_motor: int = -1  # trial0-1 的按键 1/2
    prev_mapping: int = -1  # trial0-1 的刺激-反应映射 1/2
    prev_choice: int = -1  # trial0-1 的 choice 0/1
    choice_condition_values: Tuple[int, ...] = ()  # (current_pitch, *配置的 prev_*)
    source_target_index: int | None = None  # 置乱前的 target；未置乱则 None
    sample_weight: float = 1.0  # 损失权重；inverse_frequency 时 ≠1


@dataclass(frozen=True)
class LagBalanceReport:
    """描述 lag 样本选择前后的数量与条件覆盖。

    candidate_count 是通过 split/历史连续性/干净历史筛选后的数量，
    selected_count 是平衡、丢弃不完整条件或动态采样后的当前数量。condition
    元组以 record_id 开头，随后是当前 pitch 和配置的 Choice 历史条件字段。
    missing_candidate_conditions 从未形成候选；dropped_conditions 有候选但未
    进入当前选择。报告用于审计，不反向影响采样。
    """

    candidate_count: int
    selected_count: int
    expected_condition_count: int
    candidate_condition_count: int
    selected_condition_count: int
    missing_candidate_conditions: Tuple[Tuple[int, ...], ...]
    dropped_conditions: Tuple[Tuple[int, ...], ...]


CELL_WEIGHTING_MODES = {"none", "inverse_frequency"}
CHOICE_CELL_HISTORY_FIELDS = (
    "prev_pitch",
    "prev_motor",
    "prev_mapping",
    "prev_choice",
)
CHOICE_CELL_FIELD_VALUES = {
    # 每个可选历史字段的合法“原始标签值”，顺序也定义混合进制条件 ID 的位序。
    "prev_pitch": tuple(int(value) for value in TASK_SPECS["pitch"]["classes"]),
    "prev_motor": tuple(int(value) for value in TASK_SPECS["motor"]["classes"]),
    "prev_mapping": (1, 2),
    "prev_choice": tuple(int(value) for value in TASK_SPECS["choice"]["classes"]),
}


def normalize_choice_cell_history_fields(value: object) -> Tuple[str, ...]:
    """校验 Choice 条件中追加在 current pitch 后的历史 nuisance 字段。

    None 表示只按当前 pitch 分条件。字段顺序会进入条件元组和稳定 ID，因此
    保留用户顺序；重复项或协议外字段会导致同一语义出现歧义，必须拒绝。
    """
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("choice_cell_history_fields must be a sequence")
    fields = tuple(str(field) for field in value)
    if len(set(fields)) != len(fields):
        raise ValueError("choice_cell_history_fields must not contain duplicates")
    unknown = [field for field in fields if field not in CHOICE_CELL_HISTORY_FIELDS]
    if unknown:
        supported = ", ".join(CHOICE_CELL_HISTORY_FIELDS)
        raise ValueError(
            "choice_cell_history_fields contains unsupported fields "
            f"{unknown}; supported fields: {supported}"
        )
    return fields


def normalize_choice_label_shuffle(
    enabled: object,
    seed: object,
) -> tuple[bool, int | None]:
    """校验 Choice 训练集“条件内固定标签置乱”的开关与种子。

    开启时必须提供非负整数种子，以保证同一候选集合得到同一置乱分配；关闭时
    可以不提供种子。显式拒绝 bool 充当整数，防止 YAML true 被解释为种子 1。
    """
    if not isinstance(enabled, bool):
        raise ValueError("train_label_shuffle_within_cells must be a boolean")
    if seed is None:
        if enabled:
            raise ValueError(
                "train_label_shuffle_within_cells=true requires a non-negative "
                "train_label_shuffle_seed"
            )
        return False, None
    if isinstance(seed, bool):
        raise ValueError("train_label_shuffle_seed must be a non-negative integer")
    normalized_seed = int(seed)
    if normalized_seed < 0 or float(seed) != float(normalized_seed):
        raise ValueError("train_label_shuffle_seed must be a non-negative integer")
    return enabled, normalized_seed


def normalize_time_shuffle(
    enabled: object,
    seed: object,
) -> tuple[bool, int | None]:
    """校验训练集「每试次独立时间置乱」的开关与种子。

    开启时必须提供非负整数种子，以保证同一 trial 在不同 epoch 和 DataLoader
    worker 中得到相同置乱顺序；关闭时可以不提供种子。显式拒绝 bool 充当整数，
    防止 YAML true 被解释为种子 1。
    """
    if not isinstance(enabled, bool):
        raise ValueError("train_time_shuffle_within_train must be a boolean")
    if seed is None:
        if enabled:
            raise ValueError(
                "train_time_shuffle_within_train=true requires a non-negative "
                "train_time_shuffle_seed"
            )
        return False, None
    if isinstance(seed, bool):
        raise ValueError("train_time_shuffle_seed must be a non-negative integer")
    normalized_seed = int(seed)
    if normalized_seed < 0 or float(seed) != float(normalized_seed):
        raise ValueError("train_time_shuffle_seed must be a non-negative integer")
    return enabled, normalized_seed


def resolve_balance_seed(data_config: Mapping[str, object], default_seed: int) -> int:
    """解析独立的非负平衡种子，不让样本选择依赖优化器随机状态。

    data.balance_seed 缺省时使用实验默认 seed；显式配置后，修改模型初始化或
    batch 顺序不会改变静态平衡所选的 trial。
    """
    seed = int(data_config.get("balance_seed", default_seed))
    if seed < 0:
        raise ValueError("data.balance_seed must be non-negative")
    return seed


def normalize_cell_weighting(value: object) -> str:
    """规范化逐样本条件权重模式，并兼容短横线/下划线写法。"""
    mode = "none" if value is None else str(value).strip().lower().replace("-", "_")
    if mode not in CELL_WEIGHTING_MODES:
        supported = ", ".join(sorted(CELL_WEIGHTING_MODES))
        raise ValueError(f"train_cell_weighting must be one of: {supported}")
    return mode


def normalize_cell_weight_max(value: object) -> float | None:
    """校验逆频率权重的可选上限；None 表示不截断。"""
    if value is None:
        return None
    maximum = float(value)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("train_cell_weight_max must be a positive finite number")
    return maximum


def normalize_dynamic_choice_sampling(
    enabled: object,
    min_cell_class_count: object,
    majority_ratio_cap: object,
) -> tuple[bool, int | None, float | None]:
    """校验逐 epoch Choice 条件内动态采样参数。

    开启后，每个保留条件的两个类别都至少要有 min_cell_class_count 个候选，
    majority_ratio_cap 则限制多数类相对少数类最多保留的倍数。三项必须成组
    出现；关闭时仍配置阈值通常意味着拼错配置，因此直接报错。
    """
    if not isinstance(enabled, bool):
        raise ValueError("train_dynamic_cell_sampling must be a boolean")
    if not enabled:
        if min_cell_class_count is not None or majority_ratio_cap is not None:
            raise ValueError(
                "train_min_cell_class_count and train_majority_ratio_cap require "
                "train_dynamic_cell_sampling=true"
            )
        return False, None, None
    if min_cell_class_count is None or majority_ratio_cap is None:
        raise ValueError(
            "train_dynamic_cell_sampling=true requires train_min_cell_class_count "
            "and train_majority_ratio_cap"
        )
    if isinstance(min_cell_class_count, bool):
        raise ValueError("train_min_cell_class_count must be a positive integer")
    minimum = int(min_cell_class_count)
    if minimum < 1 or float(min_cell_class_count) != float(minimum):
        raise ValueError("train_min_cell_class_count must be a positive integer")
    if isinstance(majority_ratio_cap, bool):
        raise ValueError(
            "train_majority_ratio_cap must be a finite number greater than or equal to 1"
        )
    ratio_cap = float(majority_ratio_cap)
    if not math.isfinite(ratio_cap) or ratio_cap < 1.0:
        raise ValueError(
            "train_majority_ratio_cap must be a finite number greater than or equal to 1"
        )
    return True, minimum, ratio_cap


def _choice_condition_values(example: LagExample) -> Tuple[int, ...]:
    """取得不含 record_id 的 Choice 条件值，并兼容旧样本的单一当前 pitch。"""
    values = tuple(int(value) for value in example.choice_condition_values)
    return values or (int(example.current_stimulus),)


def _choice_condition(example: LagExample) -> Tuple[int, ...]:
    """返回完整条件键：(record_id, current_pitch, 可选历史字段...)。"""
    return int(example.subject_index), *_choice_condition_values(example)


def choice_condition_id(
    values: Sequence[int],
    history_fields: Sequence[str],
) -> int:
    """把一个 Choice 条件值元组编码成稳定的非负混合进制 ID。

    values 不含 record_id，顺序必须是 current_pitch 后接 history_fields。没有
    历史字段时直接返回 1..5 的 current pitch，以保持旧输出协议；有字段时，
    依次按各字段合法值表的基数追加其位置。该 ID 只描述条件，不包含 target。
    """
    fields = normalize_choice_cell_history_fields(history_fields)
    normalized_values = tuple(int(value) for value in values)
    if len(normalized_values) != 1 + len(fields):
        raise ValueError(
            "Choice condition values must contain current_pitch followed by each "
            "configured history field"
        )
    current_pitch = normalized_values[0]
    if current_pitch not in TASK_SPECS["pitch"]["classes"]:
        raise ValueError(f"Invalid current_pitch in Choice condition: {current_pitch}")
    # 从原始 pitch 值而非零开始的下标起算，以保持无历史字段时的旧输出协议。
    packed = current_pitch
    for field, value in zip(fields, normalized_values[1:]):
        allowed = CHOICE_CELL_FIELD_VALUES[field]
        if value not in allowed:
            raise ValueError(f"Invalid {field} in Choice condition: {value}")
        packed = packed * len(allowed) + allowed.index(value)
    return int(packed)


def _lag_weight_cell(example: LagExample) -> Tuple[int, ...]:
    """逆频率权重单元：(Choice 条件, lag, target)。"""
    return (
        *_choice_condition(example),
        int(example.lag),
        int(example.target_index),
    )


def _choice_weight_cell(example: LagExample) -> Tuple[int, ...]:
    """动态 Choice 权重单元：(Choice 条件, target)，单 lag 配置无需重复 lag。"""
    return (*_choice_condition(example), int(example.target_index))


def shuffle_choice_labels_within_conditions(
    examples: Sequence[LagExample],
    *,
    seed: int,
) -> List[LagExample]:
    """在每个扩展条件内部做一次由 seed 固定的普通随机标签排列。

    分组键包含 record_id、current pitch、配置的历史条件字段和 lag，但不包含
    target。含 lag 是为了防止多 lag 训练时跨 lag 交换标签、破坏
    每个 (condition, lag) 的标签边际；单 lag 配置下分组等价，行为不变。
    严格保留条件内的类别数量与 EEG/lag 定位，允许部分标签随机保持原值。
    强制每个标签改变会在平衡二分类中退化为类别互换，仍保留全部类别信息。
    返回新 LagExample 列表，source_target_index 保存原标签，输入对象不被修改。
    """
    _, normalized_seed = normalize_choice_label_shuffle(True, seed)
    assert normalized_seed is not None
    # 保存原列表位置，置乱后仍维持候选样本的整体顺序。
    grouped: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[(*_choice_condition(example), int(example.lag))].append(index)
    rng = np.random.default_rng(normalized_seed)
    shuffled = list(examples)
    single_class_cells: list[tuple[int, ...]] = []
    # 固定遍历条件顺序，使同一 seed 不依赖字典插入顺序。
    for condition in sorted(grouped):
        indices = grouped[condition]
        targets = np.asarray(
            [int(examples[index].target_index) for index in indices],
            dtype=np.int64,
        )
        # 单类别 cell 置换是恒等操作（标签全相同，置换结果不变）；
        # 记录并在循环结束后统一警告，以便用户察觉负控制实验覆盖不完整。
        if len(np.unique(targets)) < 2:
            single_class_cells.append(condition)
        permuted = rng.permutation(targets)
        for index, target in zip(indices, permuted):
            original = examples[index]
            shuffled[index] = replace(
                original,
                target_index=int(target),
                source_target_index=int(original.target_index),
            )
    if single_class_cells:
        import warnings
        warnings.warn(
            f"label_shuffle_within_cells: {len(single_class_cells)} condition cell(s) "
            f"contain only one class — shuffling these cells is a no-op and they will "
            f"contribute unshuffled examples to the negative control. "
            f"First affected cell key: {single_class_cells[0]}",
            UserWarning,
            stacklevel=2,
        )
    return shuffled


def filter_dynamic_choice_conditions(
    examples: Sequence[LagExample],
    *,
    min_cell_class_count: int,
) -> tuple[List[LagExample], Tuple[Tuple[int, ...], ...]]:
    """保留两个类别都达到最小样本数的 Choice 条件。

    筛选以完整 Choice 条件为单位：任一类别不足就丢弃该条件的全部样本，避免
    动态 epoch 采样遇到无法定义的单类条件。返回保留样本和排序后的被丢条件键；
    不在此处做随机下采样或修改权重。
    """
    _, minimum, _ = normalize_dynamic_choice_sampling(
        True,
        min_cell_class_count,
        1.0,
    )
    assert minimum is not None
    by_condition: Dict[Tuple[int, ...], Dict[int, List[LagExample]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for example in examples:
        target = int(example.target_index)
        if target not in (0, 1):
            raise ValueError(
                "dynamic Choice sampling requires binary target indices 0 and 1"
            )
        by_condition[_choice_condition(example)][target].append(example)
    dropped = tuple(
        sorted(
            condition
            for condition, by_target in by_condition.items()
            if any(len(by_target.get(target, ())) < minimum for target in (0, 1))
        )
    )
    dropped_set = set(dropped)
    retained = [
        example
        for example in examples
        if _choice_condition(example) not in dropped_set
    ]
    return retained, dropped


def sample_dynamic_choice_epoch(
    examples: Sequence[LagExample],
    *,
    majority_ratio_cap: float,
    seed: int,
    epoch: int,
) -> List[LagExample]:
    """为一个 epoch 生成可复现的 Choice 条件内动态子集。

    每个条件完整保留少数类；多数类最多保留
    floor(majority_ratio_cap * minority_count) 条，且不做有放回过采样。随机源
    使用 seed + epoch，因此同一 epoch 可复现、不同 epoch 可轮换多数类 trial。

    选择后设置 sample_weight，使每个条件内两个类别的总权重各占该条件样本量
    的一半，但不同条件之间仍按各自保留样本量贡献，不会被强行等权。
    """
    _, _, ratio_cap = normalize_dynamic_choice_sampling(
        True,
        1,
        majority_ratio_cap,
    )
    assert ratio_cap is not None
    if seed < 0:
        raise ValueError("dynamic Choice sampling seed must be non-negative")
    if epoch < 0:
        raise ValueError("dynamic Choice sampling epoch must be non-negative")
    if not examples:
        return []

    by_condition: Dict[Tuple[int, ...], Dict[int, List[LagExample]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for example in examples:
        target = int(example.target_index)
        if target not in (0, 1):
            raise ValueError(
                "dynamic Choice sampling requires binary target indices 0 and 1"
            )
        by_condition[_choice_condition(example)][target].append(example)

    # epoch 只改变多数类的具体入选 trial，不改变候选池和条件过滤结果。
    rng = np.random.default_rng(seed + epoch)
    selected: List[LagExample] = []
    for condition in sorted(by_condition):
        by_target = by_condition[condition]
        if any(not by_target.get(target) for target in (0, 1)):
            raise ValueError(
                "dynamic Choice sampling received an incomplete configured condition"
            )
        minority_count = min(len(by_target[target]) for target in (0, 1))
        majority_limit = max(
            minority_count,
            int(math.floor(ratio_cap * minority_count)),
        )
        for target in (0, 1):
            values = by_target[target]
            n_take = min(len(values), majority_limit)
            if n_take == len(values):
                selected.extend(values)
                continue
            # 排序抽中的局部下标，避免随机抽样额外扰乱条件内原始顺序。
            chosen = np.sort(rng.choice(len(values), size=n_take, replace=False))
            selected.extend(values[int(index)] for index in chosen)

    # 权重归一后，每个条件的总权重仍等于其样本数，仅平衡条件内部的两个类别。
    condition_counts = Counter(_choice_condition(example) for example in selected)
    cell_counts = Counter(_choice_weight_cell(example) for example in selected)
    return [
        replace(
            example,
            sample_weight=(
                condition_counts[_choice_condition(example)]
                / (2.0 * cell_counts[_choice_weight_cell(example)])
            ),
        )
        for example in selected
    ]


def drop_incomplete_lag_cells(
    examples: Sequence[LagExample],
    *,
    task: str,
) -> tuple[List[LagExample], Tuple[Tuple[int, ...], ...]]:
    """删除无法在条件内部定义完整分类的 Choice 条件-lag 组。

    组键为 (record_id, current pitch, 可选历史条件..., lag)。二分类 Choice
    要求每组同时出现 target 0 和 1；缺任一类时删除整组，避免后续逆频率权重
    为不可比较的单类条件赋予过高权重。该策略目前只为 Choice 定义。
    """
    if task != "choice":
        raise ValueError("dropping incomplete cells is only supported for choice")
    required_targets = set(range(len(TASK_SPECS[task]["classes"])))
    group_targets: Dict[Tuple[int, ...], set[int]] = defaultdict(set)
    for example in examples:
        group = (*_choice_condition(example), int(example.lag))
        group_targets[group].add(int(example.target_index))
    incomplete_groups = tuple(
        sorted(group for group, targets in group_targets.items() if targets != required_targets)
    )
    if not incomplete_groups:
        return list(examples), ()
    incomplete_set = set(incomplete_groups)
    retained = [
        example
        for example in examples
        if (*_choice_condition(example), int(example.lag)) not in incomplete_set
    ]
    return retained, incomplete_groups


def apply_lag_sample_weights(
    examples: Sequence[LagExample],
    *,
    task: str,
    mode: str,
    max_weight: float | None = None,
) -> List[LagExample]:
    """为 lag 样本附加归一化的条件单元逆频率权重。

    Choice/Pitch 的 inverse_frequency 单元键为
    (record_id, current pitch, 可选历史条件..., lag, target)。未截断时每个单元
    的权重总和都等于 样本总数/单元数，因此所有单元等贡献，且全体权重总和
    等于样本数。max_weight 在归一化后逐样本截断，会相应降低稀有单元总权重。
    none 模式则显式把所有样本恢复为权重 1。
    """
    normalized_mode = normalize_cell_weighting(mode)
    normalized_max = normalize_cell_weight_max(max_weight)
    if normalized_mode == "none":
        if normalized_max is not None:
            raise ValueError("train_cell_weight_max requires inverse_frequency weighting")
        return [replace(example, sample_weight=1.0) for example in examples]
    if task not in ("choice", "pitch"):
        raise ValueError("inverse_frequency cell weighting is only supported for choice/pitch")
    if not examples:
        return []

    counts = Counter(_lag_weight_cell(example) for example in examples)
    total_weight_per_cell = len(examples) / len(counts)
    weighted: List[LagExample] = []
    for example in examples:
        weight = total_weight_per_cell / counts[_lag_weight_cell(example)]
        if normalized_max is not None:
            weight = min(weight, normalized_max)
        weighted.append(replace(example, sample_weight=weight))
    return weighted


def _clipped_lag_sample_count(dataset: "LagDecodingDataset") -> int:
    """统计按当前权重上限会被截断的样本数，供审计摘要使用。"""
    if dataset.cell_weighting != "inverse_frequency" or dataset.cell_weight_max is None:
        return 0
    if not dataset.examples:
        return 0
    counts = Counter(_lag_weight_cell(example) for example in dataset.examples)
    total_weight_per_cell = len(dataset.examples) / len(counts)
    return sum(
        count
        for count in counts.values()
        if total_weight_per_cell / count > dataset.cell_weight_max
    )


def write_lag_example_manifest(
    dataset: "LagDecodingDataset",
    csv_path: Path,
    summary_path: Path,
) -> dict[str, object]:
    """保存当前精确样本选择清单及机器可审计摘要。

    CSV 每行对应 dataset.examples 的一个位置，example_index 可直接与 Dataset
    下标对齐；record_id 是实验-被试记录索引，subject_id 则把同一 SubXX 跨
    Exp1/Exp2 合并。row_index 定位 NPZ，trial0 定位原始元数据，target 为实际
    训练标签。开启标签置乱时额外写 source_target。

    summary JSON 包含 CSV 的 SHA-256、候选/保留数量、条件覆盖、权重范围、动态
    采样状态和置乱种子。对动态采样数据集，该函数反映调用当时 epoch 的
    examples，而 dynamic_choice_pool_count 另行记录固定候选池规模。
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    history_fields = tuple(
        getattr(dataset, "choice_cell_history_fields", ())
    )
    label_shuffle_enabled = bool(
        getattr(dataset, "label_shuffle_within_cells", False)
    )
    time_shuffle_enabled = bool(
        getattr(dataset, "time_shuffle_within_train", False)
    )
    default_class_indices = tuple(
        range(len(TASK_SPECS[str(dataset.task)]["classes"]))
    )
    class_indices = tuple(
        int(index)
        for index in getattr(dataset, "class_indices", default_class_indices)
    )
    # 字段顺序是输出协议的一部分；可选历史字段保持配置顺序插入。
    fields = [
        "example_index",
        "record_id",
        "subject_id",
        "exp",
        "row_index",
        "trial0",
        "lag",
        "current_stimulus",
        *history_fields,
        "target",
        "target_original",
        *(["source_target"] if label_shuffle_enabled else []),
        "sample_weight",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for example_index, example in enumerate(dataset.examples):
            writer.writerow(
                {
                    "example_index": example_index,
                    "record_id": int(example.subject_index),
                    "subject_id": int(dataset.participant_indices[example.subject_index]),
                    "exp": int(example.exp),
                    "row_index": int(example.row_index),
                    "trial0": int(example.trial0),
                    "lag": int(example.lag),
                    "current_stimulus": int(example.current_stimulus),
                    **{
                        field: int(getattr(example, field))
                        for field in history_fields
                    },
                    "target": int(example.target_index),
                    "target_original": int(class_indices[example.target_index]),
                    **(
                        {
                            "source_target": int(example.source_target_index)
                            if example.source_target_index is not None
                            else int(example.target_index)
                        }
                        if label_shuffle_enabled
                        else {}
                    ),
                    "sample_weight": float(example.sample_weight),
                }
            )

    # 摘要针对落盘字节计算，可同时发现样本集合、顺序或格式发生变化。
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    weights = np.asarray(
        [float(example.sample_weight) for example in dataset.examples],
        dtype=np.float64,
    )
    summary: dict[str, object] = {
        "path": str(csv_path),
        "sha256": digest,
        "example_count": len(dataset.examples),
        "candidate_count": int(dataset.balance_report.candidate_count),
        "selected_count": int(dataset.balance_report.selected_count),
        "task": dataset.task,
        "split": dataset.split,
        "class_count": len(class_indices),
        "class_index_to_original": list(class_indices),
        "pitch_class_indices": list(
            getattr(dataset, "pitch_class_indices", DEFAULT_PITCH_CLASS_INDICES)
        ),
        "pitch_filter_dropped_count": int(
            getattr(dataset, "pitch_filter_dropped_count", 0)
        ),
        "balance_cells": dataset.balance_cells,
        "balance_seed": dataset.balance_seed,
        "choice_cell_history_fields": list(history_fields),
        "cell_weighting": dataset.cell_weighting,
        "cell_weight_max": dataset.cell_weight_max,
        "drop_incomplete_cells": dataset.drop_incomplete_cells,
        "incomplete_cell_dropped_group_count": len(dataset.incomplete_cell_dropped_groups),
        "incomplete_cell_dropped_example_count": dataset.incomplete_cell_dropped_example_count,
        "incomplete_cell_dropped_groups": [
            {
                "record_id": int(group[0]),
                "current_stimulus": int(group[1]),
                **{
                    field: int(value)
                    for field, value in zip(history_fields, group[2:-1])
                },
                "lag": int(group[-1]),
            }
            for group in dataset.incomplete_cell_dropped_groups
        ],
        "weight_cell_count": len({_lag_weight_cell(example) for example in dataset.examples}),
        "sample_weight_clipped_count": _clipped_lag_sample_count(dataset),
        "sample_weight_min": float(weights.min()) if weights.size else None,
        "sample_weight_max": float(weights.max()) if weights.size else None,
        "sample_weight_mean": float(weights.mean()) if weights.size else None,
        "sample_weight_sum": float(weights.sum()) if weights.size else 0.0,
        "dynamic_choice_sampling": bool(
            getattr(dataset, "dynamic_choice_sampling", False)
        ),
        "dynamic_choice_epoch": int(getattr(dataset, "dynamic_choice_epoch", 0)),
        "train_min_cell_class_count": getattr(
            dataset, "min_cell_class_count", None
        ),
        "train_majority_ratio_cap": getattr(dataset, "majority_ratio_cap", None),
        "dynamic_choice_pool_count": int(
            len(getattr(dataset, "dynamic_choice_pool", ()))
        ),
        "dynamic_choice_dropped_conditions": [
            {
                "record_id": int(condition[0]),
                "current_stimulus": int(condition[1]),
                **{
                    field: int(value)
                    for field, value in zip(history_fields, condition[2:])
                },
            }
            for condition in getattr(
                dataset, "dynamic_choice_dropped_conditions", ()
            )
        ],
        "label_shuffle_within_cells": label_shuffle_enabled,
        "label_shuffle_seed": getattr(dataset, "label_shuffle_seed", None),
        "label_shuffle_method": "within_condition_permutation_v1" if label_shuffle_enabled else None,
        "label_shuffle_target_sha256": getattr(
            dataset, "label_shuffle_target_sha256", None
        ),
        "time_shuffle_within_train": time_shuffle_enabled,
        "time_shuffle_seed": getattr(dataset, "time_shuffle_seed", None),
        "time_shuffle_assignment_sha256": getattr(
            dataset, "time_shuffle_assignment_sha256", None
        ),
    }
    if label_shuffle_enabled:
        summary["label_shuffle_manifest_sha256"] = digest
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def participant_indices_for_records(
    records: Sequence[SubjectRecord],
) -> tuple[List[str], List[int]]:
    """把实验-被试记录索引映射为跨实验共享的参与者索引。

    participant_names 按 SubXX 排序，第二个返回列表与 records 等长；其第 i 项
    是 records[i] 对应的 subject_id。于是 Exp1_Sub01 与 Exp2_Sub01 拥有不同
    record_id，却共享同一个 subject_id。
    """
    participant_names = sorted({record.subject for record in records})
    participant_to_index = {
        subject: index for index, subject in enumerate(participant_names)
    }
    return participant_names, [
        participant_to_index[record.subject] for record in records
    ]


class SubjectArrayStore:
    """实验-被试 EEG 数组的惰性加载、LRU 缓存、归一化与时间裁剪层。

    records 中每项对应一个压缩 NPZ。缓存键是 record_id（变量名沿用
    subject_index），值是完整 X 数组；这样按被试分组取 batch 时通常只需解压
    一次。归一化统计按 subject_uid 查找，作用于裁剪前的单 trial 全时间轴。

    时间选择有三种互斥模式：取前 time_samples 点、按物理秒闭区间选择，或按
    样本数做动态窗口。静态模式要求所有记录选出的物理时间轴一致；动态模式只
    要求长度足够，可用于 Exp1/Exp2 原始时间轴不同但输入长度固定的训练。

    offline_cache_dir 启用时，首次加载会把解压后的 float32 X 数组以 .npy 格式
    写入该目录（以源内容和图变换为指纹，原子写入并加 POSIX lockf 锁防并发），
    后续用 mmap_mode='r' 映射，OS page cache 在同机所有进程间共享，避免多进程
    同时解压同一 NFS 压缩 NPZ 造成 IO 竞争。
    """

    def __init__(
        self,
        records: Sequence[SubjectRecord],
        normalization_json: Path,
        time_samples: int | None = None,
        time_window_start_s: float | None = None,
        time_window_end_s: float | None = None,
        cache_size: int = 1,
        offline_cache_dir: Path | str | None = None,
        dynamic_time_window_samples: int | None = None,
        dynamic_time_window_sample_options: Sequence[int] | None = None,
        dynamic_time_window_random: bool = False,
        dynamic_time_window_seed: int = 20260530,
        dynamic_time_window_deterministic: bool = False,
        graph_laplacian_json: Path | str | None = None,
        graph_laplacian_sha256: str | None = None,
        graph_laplacian_mode: str = "replace",
    ) -> None:
        """建立记录索引和时间轴选择，但不立即解压所有 EEG。

        dynamic_time_window_sample_options 可提供多个候选长度，由 set_epoch 为
        整个 epoch 选择一个，保证 DataLoader 能堆叠同 batch 张量。窗口起点由
        dynamic_time_window_random 控制：关闭时居中，开启时可用随 epoch 推进
        的 RNG；deterministic=True 则按 record/row 哈希成固定起点，适合验证。
        cache_size 至少按 1 处理，避免每次取样都丢弃刚加载的数组。
        offline_cache_dir 非 None 时先检查本地 .npy 缓存，未命中则解压 NPZ 并
        原子写入；多进程并发由 lockf 序列化构建，校验摘要后以 mmap_mode='r' 映射。
        """
        if time_samples is not None and (
            time_window_start_s is not None or time_window_end_s is not None
        ):
            raise ValueError("Use either time_samples or a physical time window, not both")
        if (
            dynamic_time_window_samples is not None
            and dynamic_time_window_sample_options is not None
        ):
            raise ValueError(
                "Use either dynamic_time_window_samples or "
                "dynamic_time_window_sample_options, not both"
            )
        dynamic_window_sample_options: tuple[int, ...] | None = None
        if dynamic_time_window_sample_options is not None:
            dynamic_window_sample_options = tuple(
                dict.fromkeys(int(value) for value in dynamic_time_window_sample_options)
            )
            if len(dynamic_window_sample_options) == 0:
                raise ValueError("dynamic_time_window_sample_options must not be empty")
        elif dynamic_time_window_samples is not None:
            dynamic_window_sample_options = (int(dynamic_time_window_samples),)
        if dynamic_window_sample_options is not None and (
            time_samples is not None
            or time_window_start_s is not None
            or time_window_end_s is not None
        ):
            raise ValueError(
                "Use dynamic time windows by themselves, not with time_samples or a physical time window"
            )
        if dynamic_window_sample_options is not None and any(
            value < 1 for value in dynamic_window_sample_options
        ):
            raise ValueError("dynamic time window sample counts must be positive")
        if (time_window_start_s is None) != (time_window_end_s is None):
            raise ValueError("time_window_start_s and time_window_end_s must be configured together")
        if time_window_start_s is not None and time_window_start_s > time_window_end_s:
            raise ValueError("time_window_start_s must not exceed time_window_end_s")

        self.records = list(records)
        self.participant_names, self.participant_indices = participant_indices_for_records(
            records
        )
        if not records:
            raise ValueError("records must not be empty")
        if (graph_laplacian_json is None) != (graph_laplacian_sha256 is None):
            raise ValueError("Graph Laplacian requires both artifact path and checksum")
        self.graph_laplacian_mode = normalize_graph_input_mode(graph_laplacian_mode)
        if self.graph_laplacian_mode == "concat" and graph_laplacian_json is None:
            raise ValueError("graph_laplacian_mode=concat requires graph Laplacian artifacts")
        source_paths = [normalization_json, *(record.npz_path for record in records)]
        if graph_laplacian_json is not None:
            source_paths.append(graph_laplacian_json)
        self.source_files = SourceFileBinding(source_paths)
        self.graph = (GraphLaplacian.load(Path(graph_laplacian_json), graph_laplacian_sha256)
                      if graph_laplacian_json is not None else None)
        if self.graph is not None and any(record.exp != self.graph.experiment for record in records):
            raise ValueError("EEG experiment does not match graph montage")
        self.stats = load_normalization_stats(
            normalization_json,
            graph_sha256=graph_laplacian_sha256,
            graph_mode=self.graph_laplacian_mode,
        )
        missing_stats = sorted(
            record.subject_uid
            for record in records
            if record.subject_uid not in self.stats
        )
        if missing_stats:
            raise ValueError(
                f"Normalization JSON missing records: {missing_stats}"
            )
        channel_counts = {
            int(self.stats[record.subject_uid]["mean"].shape[0])
            for record in records
        }
        if len(channel_counts) != 1:
            raise ValueError(
                "Normalization statistics must use one common input channel count; "
                f"got {sorted(channel_counts)}"
            )
        self.input_channels = next(iter(channel_counts))
        if self.graph is not None:
            expected = len(self.graph.channel_names) * (
                2 if self.graph_laplacian_mode == "concat" else 1
            )
            if self.input_channels != expected:
                raise ValueError("Normalization channel count does not match graph input mode")
        self.cache_size = max(1, int(cache_size))
        self.offline_cache_dir = Path(offline_cache_dir) if offline_cache_dir is not None else None
        # LRU 中只存完整 X；元数据和时间轴体积小，始终常驻在独立字典中。
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        # Dataset inputs are immutable for the lifetime of a store. Keep verified
        # read-only mmap handles across LRU eviction, so global sampling does not
        # rehash multi-GB files per trial. New stores verify the files afresh.
        # Uncached/readonly-fallback arrays still obey cache_size and are not held here.
        self._verified_offline_cache: Dict[int, np.memmap] = {}
        self.time_indices: Dict[int, np.ndarray] = {}
        self.time_axes: Dict[int, np.ndarray] = {}
        self.dynamic_time_window_sample_options = dynamic_window_sample_options
        self.dynamic_time_window_samples = (
            dynamic_window_sample_options[0]
            if dynamic_window_sample_options is not None
            else None
        )
        self.dynamic_time_window_random = bool(dynamic_time_window_random)
        self.dynamic_time_window_seed = int(dynamic_time_window_seed)
        self.dynamic_time_window_deterministic = bool(
            dynamic_time_window_deterministic
        )
        self._current_epoch = 0
        selected_time_axes = []
        for subject_index, record in enumerate(self.records):
            time_s = self._load_time_axis(record)
            self.time_axes[subject_index] = time_s
            if self.dynamic_time_window_sample_options is not None:
                max_dynamic_samples = max(self.dynamic_time_window_sample_options)
                if time_s.shape[0] < max_dynamic_samples:
                    raise ValueError(
                        f"{record.subject_uid} has only {time_s.shape[0]} time samples; "
                        f"requested dynamic window of up to {max_dynamic_samples}"
                    )
                # 初始化先用居中窗口建立名义 time_s；逐样本随机起点在读取时决定。
                indices = self._dynamic_window_indices(subject_index, randomize=False)
            elif time_window_start_s is not None:
                # 端点按闭区间选择；微小容差吸收浮点时间轴的表示误差。
                tolerance = 1e-6
                indices = np.flatnonzero(
                    (time_s >= time_window_start_s - tolerance)
                    & (time_s <= time_window_end_s + tolerance)
                )
            elif time_samples is not None:
                if time_s.shape[0] < time_samples:
                    raise ValueError(
                        f"{record.subject_uid} has only {time_s.shape[0]} time samples; "
                        f"requested {time_samples}"
                    )
                # time_samples 的协议是从原始时间轴起点取连续前 N 点。
                indices = np.arange(int(time_samples), dtype=int)
            else:
                indices = np.arange(time_s.shape[0], dtype=int)
            if indices.size == 0:
                raise ValueError(f"Configured time window contains no samples for {record.subject_uid}")
            self.time_indices[subject_index] = indices
            selected_time_axes.append(time_s[indices])

        # 静态 batch 必须共享同一物理时间坐标，否则相同列会代表不同实验时刻。
        reference_time_s = selected_time_axes[0]
        if self.dynamic_time_window_sample_options is None:
            for record, time_s in zip(self.records[1:], selected_time_axes[1:]):
                if time_s.shape != reference_time_s.shape or not np.allclose(
                    time_s, reference_time_s, atol=1e-6, rtol=0.0
                ):
                    raise ValueError(
                        "Selected EEG time axes differ across records. "
                        "Configure time_window_start_s/time_window_end_s for aligned Exp1+Exp2 training, "
                        "or dynamic_time_window_samples for fixed-length crops; "
                        f"first record has {reference_time_s.shape[0]} samples but {record.subject_uid} "
                        f"has {time_s.shape[0]}."
                    )
        self.time_s = np.asarray(reference_time_s, dtype=np.float32)
        self.source_files.check_stamps()

    def __getstate__(self):
        # numpy.memmap pickles its bytes as a writable in-memory copy, losing
        # the filename and content verification. Spawned workers must reopen
        # and verify their own read-only mappings instead of trusting that copy.
        state = dict(self.__dict__)
        state["cache"] = OrderedDict()
        state["_verified_offline_cache"] = {}
        return state

    @staticmethod
    def _load_time_axis(record: SubjectRecord) -> np.ndarray:
        """Load a finite, strictly increasing one-dimensional time axis."""
        with np.load(record.npz_path) as data:
            if "time_s" in data:
                raw_time_s = np.asarray(data["time_s"])
                if raw_time_s.ndim != 1:
                    raise ValueError(
                        f"EEG time_s must be one-dimensional in {record.npz_path}"
                    )
                time_s = np.asarray(raw_time_s, dtype=np.float64)
            else:
                if record.exp not in LEGACY_PB2023_TIME_AXES:
                    raise ValueError(f"{record.npz_path} does not contain time_s")
                if "fsample" not in data:
                    raise ValueError(f"{record.npz_path} does not contain fsample")
                time_start_s, n_times = LEGACY_PB2023_TIME_AXES[record.exp]
                sample_rate_values = np.asarray(data["fsample"]).reshape(-1)
                if sample_rate_values.size != 1:
                    raise ValueError(f"Invalid fsample in {record.npz_path}")
                sample_rate_hz = float(sample_rate_values[0])
                if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
                    raise ValueError(f"Invalid fsample in {record.npz_path}")
                time_s = time_start_s + np.arange(
                    n_times, dtype=np.float64
                ) / sample_rate_hz
        if time_s.size == 0 or not np.all(np.isfinite(time_s)):
            raise ValueError(f"Invalid EEG time axis in {record.npz_path}")
        if np.any(np.diff(time_s) <= 0.0):
            raise ValueError(
                f"EEG time axis must be strictly increasing in {record.npz_path}"
            )
        return time_s

    def set_dynamic_time_window_samples(self, samples: int) -> None:
        """切换当前动态窗口长度并刷新所有记录的居中索引。

        samples 必须来自初始化时配置的候选集合。self.time_s 取第一条记录的
        居中窗口，作为模型输出/绘图的名义时间轴；若启用了逐样本随机起点，
        实际某个样本的物理起点由 get_normalized 读取时另行决定。
        """
        if self.dynamic_time_window_sample_options is None:
            raise ValueError("dynamic time windows are not configured")
        samples = int(samples)
        if samples not in self.dynamic_time_window_sample_options:
            allowed = ", ".join(str(value) for value in self.dynamic_time_window_sample_options)
            raise ValueError(
                f"dynamic window sample count {samples} is not in configured options [{allowed}]"
            )
        self.dynamic_time_window_samples = samples
        selected_time_axes = []
        for subject_index in range(len(self.records)):
            indices = self._dynamic_window_indices(subject_index, randomize=False)
            self.time_indices[subject_index] = indices
            selected_time_axes.append(self.time_axes[subject_index][indices])
        self.time_s = np.asarray(selected_time_axes[0], dtype=np.float32)

    def set_epoch(self, epoch: int) -> None:
        """Record the crop epoch and deterministically choose any length option.

        Per-trial crop starts derive their own local seed. When multiple window lengths
        are configured, a separate epoch-derived RNG selects one length without coupling
        that choice to sample access order.
        """
        epoch = int(epoch)
        self._current_epoch = epoch
        if self.dynamic_time_window_sample_options is None:
            return
        if len(self.dynamic_time_window_sample_options) == 1:
            self.set_dynamic_time_window_samples(
                self.dynamic_time_window_sample_options[0]
            )
            return
        rng = np.random.default_rng(self.dynamic_time_window_seed + epoch * 104_729)
        selected_index = int(
            rng.integers(0, len(self.dynamic_time_window_sample_options))
        )
        self.set_dynamic_time_window_samples(
            self.dynamic_time_window_sample_options[selected_index]
        )

    def _dynamic_window_indices(
        self,
        subject_index: int,
        row_index: int | None = None,
        randomize: bool = False,
    ) -> np.ndarray:
        """返回一条记录上连续固定长度的时间下标。

        randomize=False uses a centered window. randomize=True always derives a local
        seed from the record, row, source length, and window length, so worker count
        and access order do not matter. Non-deterministic mode also includes epoch;
        deterministic mode intentionally keeps the same crop across epochs.
        """
        if self.dynamic_time_window_samples is None:
            raise ValueError("dynamic_time_window_samples is not configured")
        window_samples = int(self.dynamic_time_window_samples)
        n_times = int(self.time_axes[subject_index].shape[0])
        max_start = n_times - window_samples
        if max_start < 0:
            record = self.records[subject_index]
            raise ValueError(
                f"{record.subject_uid} has only {n_times} time samples; "
                f"requested dynamic window of {window_samples}"
            )
        if randomize and max_start > 0:
            row_component = 0 if row_index is None else int(row_index)
            epoch_component = (
                0 if self.dynamic_time_window_deterministic else self._current_epoch
            )
            seed = (
                self.dynamic_time_window_seed
                + subject_index * 1_000_003
                + row_component * 9_176
                + epoch_component * 131
                + n_times * 37
                + window_samples * 53
            )
            start = int(np.random.default_rng(seed).integers(0, max_start + 1))
        else:
            # 奇数差值时整数除法选择靠左的两个“中心窗口”之一。
            start = max_start // 2
        return np.arange(start, start + window_samples, dtype=int)

    def _offline_cache_path(self, record: SubjectRecord) -> Path:
        """按 NPZ 内容、图变换和缓存版本定位本地缓存。"""
        assert self.offline_cache_dir is not None
        return offline_cache_path(record, self.offline_cache_dir, self.graph, self.graph_laplacian_mode)

    def _load_or_build_offline_cache(self, subject_index: int, *, readonly: bool = False) -> np.ndarray:
        """校验来源与缓存摘要；缺失时原子构建，readonly 时仅在内存读取。"""
        assert self.offline_cache_dir is not None
        if subject_index not in self._verified_offline_cache:
            source = self.records[subject_index].npz_path
            self.source_files.check_stamps([source])
            values = load_offline_cache(self.records[subject_index], self.offline_cache_dir,
                                        self.graph, self.graph_laplacian_mode, readonly=readonly)
            self.source_files.check_stamps([source])
            if not isinstance(values, np.memmap):
                return values
            self._verified_offline_cache[subject_index] = values
        return self._verified_offline_cache[subject_index]

    def _load_subject(self, subject_index: int) -> np.ndarray:
        """按需加载完整 X，并按最近使用顺序维护定长 LRU。

        命中时先弹出再放到末尾；超限时从开头淘汰最久未使用记录。返回数组形状
        为 [trial, channel, time]，统一为 float32，但不在缓存内做归一化。
        offline_cache_dir 启用时走离线 .npy 路径，否则直接解压 NPZ。
        """
        if subject_index in self.cache:
            x = self.cache.pop(subject_index)
            self.cache[subject_index] = x
            return x
        if self.offline_cache_dir is not None:
            x = self._load_or_build_offline_cache(subject_index)
        else:
            record = self.records[subject_index]
            self.source_files.check_stamps([record.npz_path])
            with np.load(record.npz_path) as data:
                x = np.asarray(data["X"], dtype=np.float32)
            self.source_files.check_stamps([record.npz_path])
            x = apply_graph_input(x, self.graph, self.graph_laplacian_mode)
        record = self.records[subject_index]
        if x.ndim != 3:
            raise ValueError(
                f"EEG X must have shape [trial, channel, time] in {record.npz_path}; "
                f"got {tuple(x.shape)}"
            )
        if x.shape[0] != record.n_trials:
            raise ValueError(
                f"{record.subject_uid} manifest reports {record.n_trials} trials, "
                f"but EEG X contains {x.shape[0]}"
            )
        if x.shape[2] != self.time_axes[subject_index].shape[0]:
            raise ValueError(
                f"{record.subject_uid} EEG X has {x.shape[2]} time samples, but "
                f"its time axis has {self.time_axes[subject_index].shape[0]}"
            )
        stats = self.stats[record.subject_uid]
        expected_stats_shape = (x.shape[1],)
        if stats["mean"].shape != expected_stats_shape or stats["std"].shape != expected_stats_shape:
            raise ValueError(
                f"{record.subject_uid} normalization stats must have shape "
                f"{expected_stats_shape}; got mean={stats['mean'].shape}, "
                f"std={stats['std'].shape}"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError(f"EEG X contains non-finite values in {record.npz_path}")
        self.cache[subject_index] = x
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return x

    def get_normalized(self, subject_index: int, row_index: int) -> torch.Tensor:
        """读取、逐通道标准化并裁剪一个 trial。

        row_index 是清洗 NPZ 的行号。取出的 x 形状为 [channel, time]，通道均值
        和标准差通过 [channel, 1] 广播；随后按静态索引或动态连续窗口裁剪，
        最终返回 float32 CPU Tensor [channel, selected_time]。
        """
        record = self.records[subject_index]
        # .copy() 防止调用方对 view 原地修改时静默污染缓存数组
        x = self._load_subject(subject_index)[row_index].copy()  # np [C,T_full] 该 trial 全时间轴
        stats = self.stats[record.subject_uid]  # mean/std: [C]，仅 train block 估计
        normalized = (x - stats["mean"][:, None]) / stats["std"][:, None]  # 广播到 [C,T]
        indices = (
            self._dynamic_window_indices(
                subject_index,
                row_index=row_index,
                randomize=self.dynamic_time_window_random,
            )
            if self.dynamic_time_window_samples is not None
            else self.time_indices[subject_index]  # 静态窗：各 record 同一物理时间下标
        )  # indices: 1D int，选中的时间点
        if indices[-1] >= normalized.shape[1]:
            raise ValueError(
                f"{record.subject_uid} has only {normalized.shape[1]} EEG time samples, "
                f"but its configured time axis selects index {indices[-1]}"
            )
        normalized = normalized[:, indices]  # [C,T_selected]，T_selected 常为 275
        return torch.from_numpy(np.asarray(normalized, dtype=np.float32))


class EEGCleanDataset(Dataset[Dict[str, torch.Tensor]]):
    """第一阶段干净 EEG 数据集。

    NPZ 已只包含非 outlier trial；初始化再按当前 trial 所属 block 选择 split，
    只建立 EEGExample 索引，不复制 EEG。每个样本返回标准化 eeg、record_id、
    跨实验 participant 的 subject_id、该记录的通道统计及 exp/trial0。
    """

    def __init__(
        self,
        records: Sequence[SubjectRecord],
        split_csv: Path,
        normalization_json: Path,
        split: str,
        time_samples: int | None = None,
        time_window_start_s: float | None = None,
        time_window_end_s: float | None = None,
        cache_size: int = 1,
        offline_cache_dir: Path | str | None = None,
        dynamic_time_window_samples: int | None = None,
        dynamic_time_window_sample_options: Sequence[int] | None = None,
        dynamic_time_window_random: bool = False,
        dynamic_time_window_seed: int = 20260530,
        dynamic_time_window_deterministic: bool = False,
        graph_laplacian_json: Path | str | None = None,
        graph_laplacian_sha256: str | None = None,
        graph_laplacian_mode: str = "replace",
    ) -> None:
        """为指定 split 建立清洗 NPZ 行号到原始 trial 的轻量索引。

        时间裁剪和缓存参数原样交给 SubjectArrayStore。split 归属按元数据 block
        查询，而不是按 NPZ 行号切分，确保同一 block 的 trial 不跨集合。
        """
        self.records = list(records)
        self.store = SubjectArrayStore(
            records,
            normalization_json,
            time_samples=time_samples,
            time_window_start_s=time_window_start_s,
            time_window_end_s=time_window_end_s,
            cache_size=cache_size,
            offline_cache_dir=offline_cache_dir,
            dynamic_time_window_samples=dynamic_time_window_samples,
            dynamic_time_window_sample_options=dynamic_time_window_sample_options,
            dynamic_time_window_random=dynamic_time_window_random,
            dynamic_time_window_seed=dynamic_time_window_seed,
            dynamic_time_window_deterministic=dynamic_time_window_deterministic,
            graph_laplacian_json=graph_laplacian_json,
            graph_laplacian_sha256=graph_laplacian_sha256,
            graph_laplacian_mode=graph_laplacian_mode,
        )
        self.participant_indices = self.store.participant_indices
        self.input_channels = self.store.input_channels
        self.time_s = self.store.time_s
        split_map = load_split_map(split_csv)
        # store 负责按需加载、标准化和裁剪；此处只建立轻量索引，不预读全部 EEG。
        examples: List[EEGExample] = []
        for subject_index, record in enumerate(self.records):
            # 元数据以原始 trial0 为键；NPZ 的 trial0 数组把清洗行映射回该坐标。
            metadata = metadata_by_trial0(Path(record.meta_path))
            with np.load(record.npz_path) as data:
                trial0 = np.asarray(data["trial0"], dtype=int)
            for row_index, trial in enumerate(trial0):
                block = int(metadata[int(trial)]["block"])
                if split_map[(record.subject_uid, block)] == split:
                    examples.append(
                        EEGExample(
                            subject_index=subject_index,
                            row_index=row_index,
                            trial0=int(trial),
                            exp=record.exp,
                        )
                    )
        self.examples = examples

    def __len__(self) -> int:
        """返回当前 split 中清洗 trial 的数量。"""
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """返回一条第一阶段样本字典。

        eeg 为 [channel, time]；channel_mean/channel_std 为 [channel]，是该条
        实验-被试记录从训练 block 估计的原始量纲统计。所有 ID、exp 和 trial0
        均为 long 标量张量，便于默认 collate 堆成 batch。
        """
        example = self.examples[index]
        record = self.records[example.subject_index]
        stats = self.store.stats[record.subject_uid]
        return {
            "eeg": self.store.get_normalized(example.subject_index, example.row_index),
            "record_id": torch.tensor(example.subject_index, dtype=torch.long),
            "subject_id": torch.tensor(
                self.participant_indices[example.subject_index], dtype=torch.long
            ),
            "channel_mean": torch.from_numpy(stats["mean"]),
            "channel_std": torch.from_numpy(stats["std"]),
            "exp": torch.tensor(example.exp, dtype=torch.long),
            "trial0": torch.tensor(example.trial0, dtype=torch.long),
        }

    @property
    def current_dynamic_time_window_samples(self) -> int | None:
        """当前 epoch 使用的动态窗口长度；非动态裁剪时返回 None。"""
        return self.store.dynamic_time_window_samples

    def set_dynamic_time_window_samples(self, samples: int) -> None:
        """手动切换动态窗口长度，主要用于测试和消融。"""
        self.store.set_dynamic_time_window_samples(samples)
        self.time_s = self.store.time_s

    def set_epoch(self, epoch: int) -> None:
        """让数据集按 epoch 更新多尺度窗口长度和随机裁剪种子。"""
        self.store.set_epoch(epoch)
        self.time_s = self.store.time_s


class LagDecodingDataset(Dataset[Dict[str, torch.Tensor]]):
    """用当前 trial EEG 解码当前或历史行为标签的第二阶段数据集。

    一个干净当前 trial 可按 lags 展开成多条 LagExample；这些样本共享 EEG，
    但 target 指向不同历史距离。lag=0 只允许单独训练，正 lag 可同时配置。
    pitch 原始值 1..5、motor 原始值 1..2 等都会映射成从 0 开始的 target。

    balance_cells 是有损的等量下采样；Choice/Pitch 的 cell_weighting 则保留
    样本、通过损失权重调整贡献。Choice 还可删除单类条件、逐 epoch 轮换多数类
    样本，以及在固定条件内置乱标签。各策略的互斥关系在初始化时显式校验。
    """

    def __init__(
        self,
        records: Sequence[SubjectRecord],
        split_csv: Path,
        normalization_json: Path,
        split: str,
        task: str,
        lags: Sequence[int],
        pitch_class_indices: Sequence[int] | None = None,
        require_clean_history: bool = True,
        balance_cells: bool = True,
        time_samples: int | None = None,
        time_window_start_s: float | None = None,
        time_window_end_s: float | None = None,
        cache_size: int = 1,
        offline_cache_dir: Path | str | None = None,
        seed: int = 20260530,
        balance_seed: int | None = None,
        cell_weighting: str = "none",
        cell_weight_max: float | None = None,
        drop_incomplete_cells: bool = False,
        dynamic_cell_sampling: bool = False,
        min_cell_class_count: int | None = None,
        majority_ratio_cap: float | None = None,
        choice_cell_history_fields: Sequence[str] | None = None,
        label_shuffle_within_cells: bool = False,
        label_shuffle_seed: int | None = None,
        time_shuffle_within_train: bool = False,
        time_shuffle_seed: int | None = None,
        dynamic_time_window_samples: int | None = None,
        dynamic_time_window_sample_options: Sequence[int] | None = None,
        dynamic_time_window_random: bool = False,
        dynamic_time_window_seed: int | None = None,
        dynamic_time_window_deterministic: bool = False,
        graph_laplacian_json: Path | str | None = None,
        graph_laplacian_sha256: str | None = None,
        graph_laplacian_mode: str = "replace",
    ) -> None:
        """为指定 split、task 和 lag 构建第二阶段样本索引。

        当前 EEG 只能来自清洗 NPZ；历史标签来自完整元数据。正 lag 候选要求从
        trial0-1 到 trial0-max(lags) 的原始 trial 连续存在且不跨 block，
        require_clean_history=True 时还要求这些历史 trial 的 valid_eeg 为 1。

        balance_seed 控制所有可复现样本选择，与优化器 seed 解耦。Choice 条件
        默认是 (record_id, current pitch)，可按 choice_cell_history_fields 追加
        trial0-1 的历史字段。注意这些 nuisance 字段描述当前 trial 的前一试次，
        并不是随预测 lag 改变的 target 来源。
        """
        self._pb_data_config = {key: value for key, value in locals().items()
                                if key not in {"self", "records", "__class__"}}
        # 先规范化所有配置，后续分支只处理确定类型和合法值。
        if task not in TASK_SPECS:
            raise ValueError(f"Unknown task: {task}")
        configured_lags = normalize_lags(lags)
        # 当前 trial 与历史 trial 的条件平衡定义不同，禁止混在同一数据集中。
        if 0 in configured_lags and len(configured_lags) > 1:
            raise ValueError("lag=0 current-trial decoding must be trained separately from positive lags")
        normalized_pitch_class_indices = normalize_pitch_class_indices(
            pitch_class_indices
        )
        if pitch_class_indices is not None and task != "pitch":
            raise ValueError("pitch_class_indices is only supported for the pitch task")
        if (
            task == "pitch"
            and normalized_pitch_class_indices != DEFAULT_PITCH_CLASS_INDICES
            and configured_lags != [0]
        ):
            raise ValueError(
                "pitch_class_indices subsets are only supported for pitch lag=0"
            )
        normalized_cell_weighting = normalize_cell_weighting(cell_weighting)
        normalized_cell_weight_max = normalize_cell_weight_max(cell_weight_max)
        (
            normalized_dynamic_cell_sampling,
            normalized_min_cell_class_count,
            normalized_majority_ratio_cap,
        ) = normalize_dynamic_choice_sampling(
            dynamic_cell_sampling,
            min_cell_class_count,
            majority_ratio_cap,
        )
        normalized_choice_cell_history_fields = (
            normalize_choice_cell_history_fields(choice_cell_history_fields)
        )
        (
            normalized_label_shuffle,
            normalized_label_shuffle_seed,
        ) = normalize_choice_label_shuffle(
            label_shuffle_within_cells,
            label_shuffle_seed,
        )
        if normalized_choice_cell_history_fields and task != "choice":
            raise ValueError(
                "choice_cell_history_fields is only supported for the choice task"
            )
        if normalized_label_shuffle and (task not in ("choice", "pitch") or split != "train"):
            raise ValueError(
                "within-cell label shuffle is only supported for the choice/pitch train split"
            )
        if normalized_label_shuffle and task == "pitch" and configured_lags == [0]:
            raise ValueError(
                "within-cell label shuffle for pitch is degenerate at lag=0: the condition "
                "cell (record, current_pitch) contains only one class (target == "
                "current_pitch), so shuffling is a no-op. Use lag>0."
            )
        (
            normalized_time_shuffle,
            normalized_time_shuffle_seed,
        ) = normalize_time_shuffle(
            time_shuffle_within_train,
            time_shuffle_seed,
        )
        if normalized_time_shuffle and split != "train":
            raise ValueError(
                "within-train time shuffle is only supported for the train split"
            )
        if normalized_cell_weighting == "none" and normalized_cell_weight_max is not None:
            raise ValueError("cell_weight_max requires inverse_frequency cell weighting")
        if normalized_cell_weighting != "none" and task not in ("choice", "pitch"):
            raise ValueError("inverse_frequency cell weighting is only supported for choice/pitch")
        if normalized_cell_weighting != "none" and balance_cells:
            raise ValueError(
                "inverse_frequency cell weighting requires balance_cells=false so all trials are retained"
            )
        if drop_incomplete_cells and task != "choice":
            raise ValueError("drop_incomplete_cells is only supported for choice")
        if drop_incomplete_cells and normalized_cell_weighting != "inverse_frequency":
            raise ValueError(
                "drop_incomplete_cells requires inverse_frequency cell weighting"
            )
        if normalized_dynamic_cell_sampling:
            # 动态采样自己负责下采样，并依赖精确逆频率权重平衡保留后的类别。
            if task != "choice" or split != "train":
                raise ValueError(
                    "dynamic Choice sampling is only supported for the choice train split"
                )
            if len(configured_lags) != 1:
                raise ValueError(
                    "dynamic Choice sampling requires exactly one configured lag"
                )
            if balance_cells:
                raise ValueError(
                    "dynamic Choice sampling requires balance_cells=false"
                )
            if normalized_cell_weighting != "inverse_frequency":
                raise ValueError(
                    "dynamic Choice sampling requires train_cell_weighting=inverse_frequency"
                )
            if normalized_cell_weight_max is not None:
                raise ValueError(
                    "dynamic Choice sampling uses exact normalized weights and does not allow "
                    "train_cell_weight_max"
                )
        self.records = list(records)
        # Reuse the same lazy loading, normalization, and dynamic-window implementation.
        self.source_files = SourceFileBinding([split_csv, *(record.meta_path for record in records)])
        self.store = SubjectArrayStore(
            records,
            normalization_json,
            time_samples=time_samples,
            time_window_start_s=time_window_start_s,
            time_window_end_s=time_window_end_s,
            cache_size=cache_size,
            offline_cache_dir=offline_cache_dir,
            dynamic_time_window_samples=dynamic_time_window_samples,
            dynamic_time_window_sample_options=dynamic_time_window_sample_options,
            dynamic_time_window_random=dynamic_time_window_random,
            dynamic_time_window_seed=(
                int(seed)
                if dynamic_time_window_seed is None
                else int(dynamic_time_window_seed)
            ),
            dynamic_time_window_deterministic=dynamic_time_window_deterministic,
            graph_laplacian_json=graph_laplacian_json,
            graph_laplacian_sha256=graph_laplacian_sha256,
            graph_laplacian_mode=graph_laplacian_mode,
        )
        self.time_s = self.store.time_s
        self.input_channels = self.store.input_channels
        # Exp1_SubXX 与 Exp2_SubXX 复用 store 中同一个参与者编号。
        self.participant_indices = self.store.participant_indices
        self.task = task
        self.task_id = int(TASK_SPECS[task]["task_id"])
        self.pitch_class_indices = normalized_pitch_class_indices
        self.class_indices = (
            normalized_pitch_class_indices
            if task == "pitch"
            else tuple(range(len(TASK_SPECS[task]["classes"])))
        )
        self.class_values = tuple(
            int(TASK_SPECS[task]["classes"][index])
            for index in self.class_indices
        )
        self.class_count = len(self.class_indices)
        self.split = str(split)
        self.balance_cells = bool(balance_cells)
        self.balance_seed = int(seed if balance_seed is None else balance_seed)
        if self.balance_seed < 0:
            raise ValueError("balance_seed must be non-negative")
        self.cell_weighting = normalized_cell_weighting
        self.cell_weight_max = normalized_cell_weight_max
        self.drop_incomplete_cells = bool(drop_incomplete_cells)
        self.dynamic_choice_sampling = normalized_dynamic_cell_sampling
        self.min_cell_class_count = normalized_min_cell_class_count
        self.majority_ratio_cap = normalized_majority_ratio_cap
        self.dynamic_choice_epoch = 0
        self.dynamic_choice_pool: Tuple[LagExample, ...] = ()
        self.dynamic_choice_dropped_conditions: Tuple[Tuple[int, ...], ...] = ()
        self.choice_cell_history_fields = normalized_choice_cell_history_fields
        self.label_shuffle_within_cells = normalized_label_shuffle
        self.label_shuffle_seed = normalized_label_shuffle_seed
        self.label_shuffle_target_sha256: str | None = None
        self.time_shuffle_within_train = normalized_time_shuffle
        self.time_shuffle_seed = normalized_time_shuffle_seed
        self.time_shuffle_assignment_sha256: str | None = None
        # lags 是真正监督的历史距离；required_lags 还覆盖中间 trial 以验证连续性。
        self.lags = configured_lags
        self.lag_to_index = {lag: index for index, lag in enumerate(self.lags)}
        # 即使只预测部分 lag，也要求直到最大 lag 的历史连续存在，避免跨 block。
        required_lags = list(range(1, max(self.lags) + 1)) if max(self.lags) > 0 else []
        if self.choice_cell_history_fields and 1 not in required_lags:
            # 即使解码 lag=0，扩展 Choice 条件仍需读取当前 trial 的前一试次。
            required_lags.append(1)
            required_lags.sort()
        # class_to_index 将元数据原始标签值映射为模型使用的连续 0-based target。
        classes = list(self.class_values)
        class_to_index = {value: index for index, value in enumerate(classes)}
        split_map = load_split_map(split_csv)
        candidates: List[LagExample] = []
        self.pitch_filter_dropped_count = 0

        for subject_index, record in enumerate(self.records):
            metadata = metadata_by_trial0(Path(record.meta_path))
            with np.load(record.npz_path) as data:
                clean_trial0 = np.asarray(data["trial0"], dtype=int)
            for row_index, trial in enumerate(clean_trial0):
                current = metadata[int(trial)]  # trial=trial0 原始编号
                block = int(current["block"])
                if split_map[(record.subject_uid, block)] != split:
                    continue  # 划分以「当前 trial 所在 block」为准
                current_stimulus = int(current["pitch"])
                if task == "pitch" and current_stimulus not in self.class_values:
                    self.pitch_filter_dropped_count += 1
                    continue  # pitch lag=0：在平衡和连续重映射前移除未启用类别
                # 历史必须存在且同 block，防止跨 block 泄漏
                required_histories = [metadata.get(int(trial) - lag) for lag in required_lags]
                if any(
                    history is None or int(history["block"]) != block
                    for history in required_histories
                ):
                    continue
                if require_clean_history and any(
                    int(history["valid_eeg"]) != 1 for history in required_histories
                ):
                    continue  # 可选：历史标签也必须来自非 outlier EEG trial
                history_by_lag = dict(zip(required_lags, required_histories))
                previous = history_by_lag.get(1)  # Choice nuisance 固定读 lag=1
                history_values: dict[str, int] = {}
                for field in CHOICE_CELL_HISTORY_FIELDS:
                    source_field = field.removeprefix("prev_")
                    if previous is None:
                        history_values[field] = -1
                    elif source_field in previous and previous[source_field] != "":
                        history_values[field] = int(previous[source_field])
                    elif field in self.choice_cell_history_fields:
                        raise ValueError(
                            f"Configured history field {field} requires metadata "
                            f"column {source_field} in {record.meta_path}"
                        )
                    else:
                        history_values[field] = -1
                # 同一行当前 EEG 可展开成多个 lag 样本（标签来源不同）
                histories = [
                    current if lag == 0 else metadata[int(trial) - lag]
                    for lag in self.lags
                ]
                choice_condition_values = (
                    current_stimulus,
                    *(
                        history_values[field]
                        for field in self.choice_cell_history_fields
                    ),
                )
                for lag, history in zip(self.lags, histories):
                    label_column = TASK_SPECS[task]["column"]
                    target_label = int(history[label_column])  # 原始取值，如 pitch 1..5
                    try:
                        target_index = class_to_index[target_label]  # → 0-based
                    except KeyError as exc:
                        raise ValueError(
                            f"Invalid {label_column} label {target_label!r} for "
                            f"{record.subject_uid} trial0={int(trial)} lag={lag}; "
                            f"expected one of {classes}"
                        ) from exc
                    candidates.append(
                        LagExample(
                            subject_index=subject_index,
                            row_index=row_index,  # EEG 始终当前行
                            trial0=int(trial),
                            exp=record.exp,
                            lag=lag,
                            current_stimulus=current_stimulus,
                            target_index=target_index,  # 监督来自 history（可能是 lag 前）
                            prev_pitch=history_values["prev_pitch"],
                            prev_motor=history_values["prev_motor"],
                            prev_mapping=history_values["prev_mapping"],
                            prev_choice=history_values["prev_choice"],
                            choice_condition_values=choice_condition_values,
                        )
                    )

        if self.label_shuffle_within_cells:
            assert self.label_shuffle_seed is not None
            candidates = shuffle_choice_labels_within_conditions(
                candidates,
                seed=self.label_shuffle_seed,
            )
            # 对最终标签分配单独取摘要，便于确认两次控制实验使用同一置乱。
            assignment = "\n".join(
                (
                    f"{example.subject_index},{example.row_index},{example.trial0},"
                    f"{example.lag},{example.source_target_index},{example.target_index}"
                )
                for example in candidates
            )
            self.label_shuffle_target_sha256 = hashlib.sha256(
                assignment.encode("ascii")
            ).hexdigest()

        # balance_cells 会在每个 record × 当前刺激内部平衡 lag × target 类别。
        # 训练时这能减少类别偏差；验证/测试默认关闭，保留完整 held-out 分布。
        selected_examples = (
            _balance_lag_cells(
                candidates,
                n_classes=self.class_count,
                seed=self.balance_seed,
                task=task,
            )
            if balance_cells
            else candidates
        )
        pre_drop_count = len(selected_examples)
        if self.drop_incomplete_cells:
            selected_examples, self.incomplete_cell_dropped_groups = drop_incomplete_lag_cells(
                selected_examples,
                task=task,
            )
        else:
            self.incomplete_cell_dropped_groups = ()
        self.incomplete_cell_dropped_example_count = pre_drop_count - len(selected_examples)
        if self.dynamic_choice_sampling:
            assert self.min_cell_class_count is not None
            assert self.majority_ratio_cap is not None
            dynamic_pool, self.dynamic_choice_dropped_conditions = (
                filter_dynamic_choice_conditions(
                    selected_examples,
                    min_cell_class_count=self.min_cell_class_count,
                )
            )
            if not dynamic_pool:
                raise ValueError(
                    "No Choice train conditions remain after "
                    "train_min_cell_class_count filtering"
                )
            # 候选池固定不变；每个 epoch 只从中重选多数类的具体 trial。
            self.dynamic_choice_pool = tuple(dynamic_pool)
            self.examples = sample_dynamic_choice_epoch(
                self.dynamic_choice_pool,
                majority_ratio_cap=self.majority_ratio_cap,
                seed=self.balance_seed,
                epoch=0,
            )
        else:
            self.examples = apply_lag_sample_weights(
                selected_examples,
                task=task,
                mode=self.cell_weighting,
                max_weight=self.cell_weight_max,
            )
        # 报告同时覆盖理论条件、历史过滤后的候选条件和初始化时的保留条件。
        history_combinations = product(
            *(
                CHOICE_CELL_FIELD_VALUES[field]
                for field in self.choice_cell_history_fields
            )
        )
        history_combinations = tuple(history_combinations) or ((),)
        expected_conditions = {
            (subject_index, int(current_stimulus), *history_values)
            for subject_index in range(len(self.records))
            for current_stimulus in (
                self.class_values
                if task == "pitch"
                else TASK_SPECS["pitch"]["classes"]
            )
            for history_values in history_combinations
        }
        candidate_conditions = {
            _choice_condition(example) for example in candidates
        }
        selected_conditions = {
            _choice_condition(example) for example in self.examples
        }
        self.balance_report = LagBalanceReport(
            candidate_count=len(candidates),
            selected_count=len(self.examples),
            expected_condition_count=len(expected_conditions),
            candidate_condition_count=len(candidate_conditions),
            selected_condition_count=len(selected_conditions),
            missing_candidate_conditions=tuple(sorted(expected_conditions - candidate_conditions)),
            dropped_conditions=tuple(sorted(candidate_conditions - selected_conditions)),
        )
        # 对时间置乱控制记录置乱分配的唯一指纹；seed 是整个分配的决定因素，
        # 因此 SHA 结合 seed 与最终样本列表，可独立确认两次运行的置乱完全一致。
        if self.time_shuffle_within_train:
            assert self.time_shuffle_seed is not None
            assignment = "\n".join(
                f"{example.subject_index},{example.row_index},{example.trial0}"
                for example in self.examples
            )
            self.time_shuffle_assignment_sha256 = hashlib.sha256(
                f"seed={self.time_shuffle_seed}\n{assignment}".encode("ascii")
            ).hexdigest()

        self.source_files.check_stamps()
        self.store.source_files.check_stamps()

    def __len__(self) -> int:
        """返回当前 epoch 可见的 lag 样本数。"""
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """返回一条当前 EEG 与指定 lag 监督的样本字典。

        eeg 是 [channel, time]。lag 是真实历史距离；lag_index 是该距离在排序后
        self.lags 中的位置，供多 lag 头索引。record_id 区分 Exp1/Exp2 文件，
        subject_id 合并同一参与者。choice_cell_id 只编码 current pitch 与配置
        历史字段，不含 record_id；需要全局唯一条件时应与 record_id 联合使用。
        target 是 0-based 类别，sample_weight 是 float32 标量损失权重。
        """
        example = self.examples[index]
        history_fields = tuple(
            getattr(self, "choice_cell_history_fields", ())
        )
        condition_values = _choice_condition_values(example)
        eeg = self.store.get_normalized(example.subject_index, example.row_index)  # float [C,T] 已归一化
        if getattr(self, "time_shuffle_within_train", False):
            # 每试次用确定性独立种子置换时间轴（训练集专用的负控制实验）。
            # 种子键包含 trial0 保证跨 trial 置换独立；包含 subject_index/row_index
            # 保证同 trial0 在不同 record 不发生碰撞。get_normalized 已返回新 tensor，
            # 无需额外 clone，直接索引时间维即可。
            assert self.time_shuffle_seed is not None
            trial_rng = np.random.default_rng(
                (self.time_shuffle_seed, example.subject_index, example.row_index, example.trial0)
            )
            perm = trial_rng.permutation(eeg.shape[-1])
            eeg = eeg[..., torch.from_numpy(perm)]
        return {
            # debug 时重点看：eeg/target/lag/subject_id/sample_weight
            "eeg": eeg,
            "record_id": torch.tensor(example.subject_index, dtype=torch.long),  # 文件级 0..n_records-1
            "subject_id": torch.tensor(
                self.participant_indices[example.subject_index], dtype=torch.long
            ),  # 人级 embedding 下标（Exp1/2 同 Sub 可相同）
            "exp": torch.tensor(example.exp, dtype=torch.long),  # 1 或 2
            "trial0": torch.tensor(example.trial0, dtype=torch.long),  # 原始 trial 编号
            "lag": torch.tensor(example.lag, dtype=torch.long),  # 历史距离（秒不是，是 trial 个数）
            "lag_index": torch.tensor(
                self.lag_to_index[example.lag], dtype=torch.long
            ),  # lag 在配置 self.lags 排序中的位置（多 lag 头用）
            "task_id": torch.tensor(self.task_id, dtype=torch.long),  # pitch=0 choice=1 motor=2
            "current_stimulus": torch.tensor(example.current_stimulus, dtype=torch.long),  # pitch 1..5
            "prev_pitch": torch.tensor(example.prev_pitch, dtype=torch.long),
            "prev_motor": torch.tensor(example.prev_motor, dtype=torch.long),
            "prev_mapping": torch.tensor(example.prev_mapping, dtype=torch.long),
            "prev_choice": torch.tensor(example.prev_choice, dtype=torch.long),
            "choice_cell_id": torch.tensor(
                choice_condition_id(condition_values, history_fields),
                dtype=torch.long,
            ),  # 条件编码 int；指标分层用；不含 record_id
            "target": torch.tensor(example.target_index, dtype=torch.long),  # y，0-based
            "target_original": torch.tensor(
                self.class_indices[example.target_index], dtype=torch.long
            ),  # 原始任务类别下标；Pitch 子集训练时不同于连续 target
            "sample_weight": torch.tensor(example.sample_weight, dtype=torch.float32),  # 默认 1.0
        }

    def set_epoch(self, epoch: int) -> None:
        """Advance dynamic time crops and, when enabled, the Choice subset.

        The underlying store always receives the epoch so order-independent random crops
        can vary across epochs. Dynamic Choice sampling then replaces examples before the
        next sampler iteration; static datasets otherwise keep their example list unchanged.
        """
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.store.set_epoch(epoch)
        self.time_s = self.store.time_s
        if not self.dynamic_choice_sampling:
            return
        assert self.majority_ratio_cap is not None
        self.dynamic_choice_epoch = epoch
        self.examples = sample_dynamic_choice_epoch(
            self.dynamic_choice_pool,
            majority_ratio_cap=self.majority_ratio_cap,
            seed=self.balance_seed,
            epoch=epoch,
        )


def _balance_lag_cells(
    candidates: Sequence[LagExample],
    n_classes: int,
    seed: int,
    task: str,
) -> List[LagExample]:
    """对完整条件内的 lag × target 单元做固定、无放回等量下采样。

    正 lag 的分组键是完整 Choice 条件，即 record_id、当前 pitch 和可选历史
    nuisance 字段；组内必须拥有所有实际出现 lag 与全部 target 的笛卡尔积，
    否则整组跳过。每个单元取共同最小数量，因而平衡会丢弃样本。

    lag=0 单独处理。对 pitch 任务，target 本身就是 current pitch，若仍按当前
    pitch 分组，每组天然只有一个类别，所以只按 record_id 分组；其他任务仍按
    完整 Choice 条件分组。seed 只控制具体保留哪些样本。
    """
    rng = np.random.default_rng(seed)
    balanced: List[LagExample] = []

    # 当前 trial 解码不和正 lag 混合计算共同最小值。
    current_groups: Dict[Tuple[int, ...], Dict[int, List[LagExample]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for example in candidates:
        if example.lag != 0:
            continue
        # lag=0/pitch 的 target 就是当前 pitch，不能再按 current_stimulus 分层，
        # 否则每个 current pitch 条件中只会有一个 target 类别。
        key = (
            _choice_condition(example)
            if task != "pitch"
            else (example.subject_index,)
        )
        current_groups[key][example.target_index].append(example)
    for key in sorted(current_groups):
        by_target = current_groups[key]
        required_cells = list(range(n_classes))
        if any(cell not in by_target for cell in required_cells):
            continue
        n_take = min(len(by_target[cell]) for cell in required_cells)
        for cell in required_cells:
            values = by_target[cell]
            chosen = rng.choice(len(values), size=n_take, replace=False)
            balanced.extend(values[int(index)] for index in chosen)

    # 正 lag 条件内的二级键显式包含 lag 和 target。
    groups: Dict[Tuple[int, ...], Dict[Tuple[int, int], List[LagExample]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for example in candidates:
        if example.lag == 0:
            continue
        key = _choice_condition(example)
        groups[key][(example.lag, example.target_index)].append(example)

    for key in sorted(groups):
        by_lag_target = groups[key]
        lags = sorted({lag for lag, _target in by_lag_target})
        required_cells = [(lag, target) for lag in lags for target in range(n_classes)]
        if any(cell not in by_lag_target for cell in required_cells):
            continue
        # 所有单元格取共同最小值，防止类别不均衡被误认为重激活差异。
        n_take = min(len(by_lag_target[cell]) for cell in required_cells)
        for cell in required_cells:
            values = by_lag_target[cell]
            chosen = rng.choice(len(values), size=n_take, replace=False)
            balanced.extend(values[int(index)] for index in chosen)
    return balanced


class GlobalBatchSampler(Sampler[List[int]]):
    """在当前 Dataset 的全部样本上构造全局 batch。

    shuffle=True 时使用 seed + epoch 重排 Dataset 下标，因此一个 batch 可以
    混合多个实验-被试记录。这样随机性强，但可能频繁切换 SubjectArrayStore
    的压缩 NPZ；是否值得取决于缓存大小和 I/O 成本。
    """

    def __init__(
        self,
        dataset: EEGCleanDataset | LagDecodingDataset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool = False,
        seed: int = 20260530,
    ) -> None:
        """保存采样配置；实际索引顺序在每次 __iter__ 时按当前 epoch 生成。"""
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """更新随机种子偏移，使每个 epoch 使用不同的全局样本顺序。"""
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        """在全部当前样本上重排下标，再连续切成 batch_size 大小的列表。"""
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = np.arange(len(self.dataset), dtype=int)
        if self.shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size].tolist()
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        """返回按当前 Dataset 长度可生成的 batch 数。"""
        n_full, remainder = divmod(len(self.dataset), self.batch_size)
        return n_full + int(bool(remainder) and not self.drop_last)


class SubjectBatchSampler(Sampler[List[int]]):
    """按实验-被试记录连续生成 batch，减少压缩 NPZ 重复解压。

    分组使用 example.subject_index，也就是 record_id，而不是跨实验 subject_id；
    这是因为 LRU 的缓存单元正是一条记录对应的 NPZ。不同记录之间绝不拼接一个
    batch，代价是末尾小 batch 更多、跨被试混合随机性较弱。
    """

    def __init__(
        self,
        dataset: EEGCleanDataset | LagDecodingDataset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool = False,
        seed: int = 20260530,
    ) -> None:
        """保存按记录分组的 batch 配置；分组在迭代时从最新 examples 重建。"""
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def _by_subject(self) -> Dict[int, List[int]]:
        """构造 record_id 到当前 Dataset 下标列表的映射。

        每次重新构造是为了兼容动态 Choice 采样在 set_epoch 中替换 examples。
        """
        by_subject: Dict[int, List[int]] = defaultdict(list)
        for index, example in enumerate(self.dataset.examples):
            by_subject[int(example.subject_index)].append(index)
        return dict(by_subject)

    def set_epoch(self, epoch: int) -> None:
        """更新随机种子偏移，使每个 epoch 获得新的样本顺序。"""
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        """重排记录顺序及记录内样本，再逐记录切 batch。"""
        rng = np.random.default_rng(self.seed + self.epoch)
        by_subject = self._by_subject()
        subjects = np.asarray(sorted(by_subject), dtype=int)
        if self.shuffle:
            rng.shuffle(subjects)
        for subject in subjects:
            indices = np.asarray(by_subject[int(subject)], dtype=int)
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size].tolist()
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        """逐记录计算 batch 数；drop_last 也分别作用于每条记录的尾部。"""
        total = 0
        for indices in self._by_subject().values():
            n_full, remainder = divmod(len(indices), self.batch_size)
            total += n_full + int(bool(remainder) and not self.drop_last)
        return total


def make_batch_sampler(
    dataset: EEGCleanDataset | LagDecodingDataset,
    batch_size: int,
    shuffle: bool,
    strategy: str = "global",
    seed: int = 20260530,
) -> Sampler[List[int]]:
    """按策略名创建 batch sampler。

    global 在整个 Dataset 上切 batch；subject_grouped 以 record_id 分组切 batch。
    两者均返回 Dataset 下标列表，且默认保留最后一个不足 batch_size 的 batch。
    动态数据集每轮先调用 dataset.set_epoch，再调用 sampler.set_epoch。
    """
    if strategy == "global":
        return GlobalBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle, seed=seed)
    if strategy == "subject_grouped":
        return SubjectBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle, seed=seed)
    raise ValueError(f"Unknown batch sampling strategy: {strategy}")
