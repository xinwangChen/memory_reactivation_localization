"""SwanLab 的可选初始化、指标命名映射与生命周期封装。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


def _plain_value(value: Any) -> Any:
    """递归把 Path、tuple 等配置值转换为 SwanLab 可序列化的普通对象。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


def _env_bool(name: str) -> bool | None:
    """读取常见形式的布尔环境变量；未设置时返回 None。"""
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, or on/off")


def _swanlab_enabled_from_config(logger_config: Mapping[str, Any]) -> bool:
    """合并 YAML 与环境变量，得到最终的 SwanLab 开关。

    PB2023_SWANLAB_ENABLED 可显式覆盖 YAML；全局 SWANLAB_DISABLED=true
    的优先级最高，便于 debug/smoke run 临时禁止联网记录。
    """
    enabled = bool(logger_config.get("enabled", False))
    env_enabled = _env_bool("PB2023_SWANLAB_ENABLED")
    if env_enabled is not None:
        enabled = env_enabled
    env_disabled = _env_bool("SWANLAB_DISABLED")
    if env_disabled is True:
        enabled = False
    return enabled


# 左侧是训练 CSV/返回字典中的稳定字段，右侧是 SwanLab 面板中的层级路径。
COMMON_SWANLAB_KEYS = {
    "learning_rate": "optimization/learning_rate",
    "train_learning_rate": "optimization/learning_rate",
    "muon_learning_rate": "optimization/muon_learning_rate",
    "adamw_learning_rate": "optimization/adamw_learning_rate",
    "grad_norm": "optimization/gradient_norm",
    "train_grad_norm": "optimization/gradient_norm",
    "nonfinite_grad_batches": "optimization/nonfinite_gradient_batches",
    "train_nonfinite_grad_batches": "optimization/nonfinite_gradient_batches",
    "amp_scale": "optimization/amp_scale",
    "train_amp_scale": "optimization/amp_scale",
    "train_val_gap": "diagnostics/train_val_gap",
    "iteration": "progress/iteration",
    "epoch": "progress/data_epoch",
}

TARGET_SWANLAB_KEYS = {
    "train_loss": "train/loss/total",
    # 训练时可包含按概率抽取的非 Top-q 位置；验证项才是严格 Top-q。
    "train_time_loss": "train/loss/time_supervision",
    "train_time_loss_position_count_per_sample": "train/supervision/time_loss_position_count",
    "train_non_topq_time_loss_count_per_sample": "train/supervision/non_topq_count",
    "train_pooled_loss": "train/loss/attention_pooled",
    "train_fusion_loss": "train/loss/fusion",
    "train_reconstruction_loss": "train/loss/reconstruction",
    "train_topk_covered_time_loss": "train/loss/topk_covered_time",
    "train_attention_mask_loss": "train/loss/attention_mask",
    "train_attention_mass_loss": "train/loss/attention_mass",
    "train_clean_attention_loss": "train/loss/clean_attention",
    "train_clean_attention_candidate_count_per_sample": "train/attention/clean_candidate_count",
    "train_clean_attention_selected_count_per_sample": "train/attention/clean_selected_count",
    "train_clean_attention_selected_fraction": "train/attention/clean_selected_fraction",
    "train_attention_smoothness_loss": "train/loss/attention_smoothness",
    "train_covered_time_fraction": "train/augmentation/covered_time_fraction",
    "train_time_accuracy": "train/accuracy/time_topk",
    "train_time_no_class_prediction_fraction": "train/prediction/time_no_class_fraction",
    "train_pooled_accuracy": "train/accuracy/attention_pooled",
    "train_fusion_accuracy": "train/accuracy/fusion",
    "train_time_balanced_accuracy": "train/accuracy/time_balanced",
    "train_time_cell_balanced_accuracy": "train/accuracy/time_cell_balanced",
    "train_time_cell_count": "train/cells/time_complete_count",
    "train_time_subject_cell_balanced_accuracy": "train/accuracy/time_subject_cell_balanced",
    "train_time_subject_cell_count": "train/cells/time_subject_count",
    "train_pooled_balanced_accuracy": "train/accuracy/pooled_balanced",
    "train_pooled_cell_balanced_accuracy": "train/accuracy/pooled_cell_balanced",
    "train_pooled_cell_count": "train/cells/pooled_complete_count",
    "train_fusion_balanced_accuracy": "train/accuracy/fusion_balanced",
    "train_fusion_cell_balanced_accuracy": "train/accuracy/fusion_cell_balanced",
    "train_fusion_cell_count": "train/cells/fusion_complete_count",
    "val_loss": "val/loss/total",
    "val_time_loss": "val/loss/time_topk",
    "val_time_loss_position_count_per_sample": "val/supervision/time_loss_position_count",
    "val_non_topq_time_loss_count_per_sample": "val/supervision/non_topq_count",
    "val_pooled_loss": "val/loss/attention_pooled",
    "val_fusion_loss": "val/loss/fusion",
    "val_reconstruction_loss": "val/loss/reconstruction",
    "val_topk_covered_time_loss": "val/loss/topk_covered_time",
    "val_attention_mask_loss": "val/loss/attention_mask",
    "val_attention_mass_loss": "val/loss/attention_mass",
    "val_clean_attention_loss": "val/loss/clean_attention",
    "val_attention_smoothness_loss": "val/loss/attention_smoothness",
    "val_covered_time_fraction": "val/augmentation/covered_time_fraction",
    "val_time_accuracy": "val/accuracy/time_topk",
    "val_time_no_class_prediction_fraction": "val/prediction/time_no_class_fraction",
    "val_pooled_accuracy": "val/accuracy/attention_pooled",
    "val_fusion_accuracy": "val/accuracy/fusion",
    "val_time_balanced_accuracy": "val/accuracy/time_balanced",
    "val_time_cell_balanced_accuracy": "val/accuracy/time_cell_balanced",
    "val_time_cell_count": "val/cells/time_complete_count",
    "val_time_subject_cell_balanced_accuracy": "val/accuracy/time_subject_cell_balanced",
    "val_time_subject_cell_count": "val/cells/time_subject_count",
    "val_pooled_balanced_accuracy": "val/accuracy/pooled_balanced",
    "val_pooled_cell_balanced_accuracy": "val/accuracy/pooled_cell_balanced",
    "val_pooled_cell_count": "val/cells/pooled_complete_count",
    "val_fusion_balanced_accuracy": "val/accuracy/fusion_balanced",
    "val_fusion_cell_balanced_accuracy": "val/accuracy/fusion_cell_balanced",
    "val_fusion_cell_count": "val/cells/fusion_complete_count",
}

TARGET_AUGMENTATION_DIAGNOSTIC_PATHS = {
    "augmented_trial_fraction": "augmentation/augmented_trial_fraction",
    "roll_sample_count": "augmentation/roll_sample_count",
    "roll_fraction": "augmentation/roll_fraction",
    "roll_fraction_among_augmented": "augmentation/roll_fraction_among_augmented",
    "roll_left_fraction": "augmentation/roll_left_fraction",
    "roll_right_fraction": "augmentation/roll_right_fraction",
    "roll_abs_shift_mean_samples": "augmentation/roll_abs_shift_mean_samples",
    "roll_abs_shift_max_samples": "augmentation/roll_abs_shift_max_samples",
    "replacement_trial_fraction": "augmentation/replacement_trial_fraction",
    "local_noise_fraction": "augmentation/local_noise_fraction",
    "near_full_noise_fraction_among_replacements": "augmentation/near_full_noise_fraction_among_replacements",
    "all_covered_invalid_bce": "augmentation/all_covered_invalid_bce",
    "topk_covered_count_per_sample": "augmentation/topk_covered_count_per_sample",
    "topk_covered_position_fraction": "augmentation/topk_covered_position_fraction",
    "topk_covered_recall": "augmentation/topk_covered_recall",
    "topk_covered_trial_fraction": "augmentation/topk_covered_trial_fraction",
    "topk_covered_probe_bce": "augmentation/topk_covered_probe_bce",
    "effective_attention_mass": "attention/effective_mass",
    "covered_effective_attention_mean": "attention/covered_effective_mean",
    "uncovered_effective_attention_mean": "attention/uncovered_effective_mean",
    "covered_effective_attention_mass": "attention/covered_effective_mass",
    "uncovered_effective_attention_mass": "attention/uncovered_effective_mass",
    "covered_effective_attention_mass_fraction": "attention/covered_effective_mass_fraction",
    "covered_attention_gate_mean": "attention/covered_gate_mean",
    "uncovered_attention_gate_mean": "attention/uncovered_gate_mean",
}
for _ratio_label in ("q10", "q20", "q30"):
    # 探针 q 不参与主损失，仅用于观察不同 Top-q 下噪声覆盖和注意力关系。
    TARGET_AUGMENTATION_DIAGNOSTIC_PATHS.update(
        {
            f"probe_{_ratio_label}_covered_count_per_sample": (
                f"augmentation/probes/{_ratio_label}/covered_count_per_sample"
            ),
            f"probe_{_ratio_label}_covered_position_fraction": (
                f"augmentation/probes/{_ratio_label}/covered_position_fraction"
            ),
            f"probe_{_ratio_label}_covered_recall": (
                f"augmentation/probes/{_ratio_label}/covered_recall"
            ),
            f"probe_{_ratio_label}_covered_trial_fraction": (
                f"augmentation/probes/{_ratio_label}/covered_trial_fraction"
            ),
            f"probe_{_ratio_label}_invalid_bce": (
                f"augmentation/probes/{_ratio_label}/invalid_bce"
            ),
        }
    )
for _phase in ("train", "val"):
    TARGET_SWANLAB_KEYS.update(
        {
            f"{_phase}_{metric_name}": f"{_phase}/{metric_path}"
            for metric_name, metric_path in TARGET_AUGMENTATION_DIAGNOSTIC_PATHS.items()
        }
    )

TIME_ONLY_SWANLAB_KEYS = {
    "train_loss": "train/loss/total",
    "train_time_loss": "train/loss/random_time",
    "train_fusion_loss": "train/loss/fusion",
    "train_sampled_covered_time_loss": "train/loss/sampled_covered_time",
    "train_reconstruction_loss": "train/loss/reconstruction",
    "train_covered_time_fraction": "train/augmentation/covered_time_fraction",
    "train_time_sample_fraction": "train/sampling/time_fraction",
    "train_time_accuracy": "train/accuracy/random_time",
    "train_fusion_accuracy": "train/accuracy/fusion",
    "train_time_balanced_accuracy": "train/accuracy/time_balanced",
    "train_time_cell_balanced_accuracy": "train/accuracy/time_cell_balanced",
    "train_time_cell_count": "train/cells/time_complete_count",
    "train_time_subject_cell_balanced_accuracy": "train/accuracy/time_subject_cell_balanced",
    "train_time_subject_cell_count": "train/cells/time_subject_count",
    "train_fusion_balanced_accuracy": "train/accuracy/fusion_balanced",
    "train_fusion_cell_balanced_accuracy": "train/accuracy/fusion_cell_balanced",
    "train_fusion_cell_count": "train/cells/fusion_complete_count",
    "val_loss": "val/loss/total",
    "val_time_loss": "val/loss/all_time",
    "val_fusion_loss": "val/loss/fusion",
    "val_sampled_covered_time_loss": "val/loss/sampled_covered_time",
    "val_reconstruction_loss": "val/loss/reconstruction",
    "val_time_sample_fraction": "val/sampling/time_fraction",
    "val_time_accuracy": "val/accuracy/all_time",
    "val_fusion_accuracy": "val/accuracy/fusion",
    "val_time_balanced_accuracy": "val/accuracy/time_balanced",
    "val_time_cell_balanced_accuracy": "val/accuracy/time_cell_balanced",
    "val_time_cell_count": "val/cells/time_complete_count",
    "val_time_subject_cell_balanced_accuracy": "val/accuracy/time_subject_cell_balanced",
    "val_time_subject_cell_count": "val/cells/time_subject_count",
    "val_fusion_balanced_accuracy": "val/accuracy/fusion_balanced",
    "val_fusion_cell_balanced_accuracy": "val/accuracy/fusion_cell_balanced",
    "val_fusion_cell_count": "val/cells/fusion_complete_count",
}



def _mapped_metrics(metrics: Mapping[str, object], metric_keys: Mapping[str, str]) -> dict[str, object]:
    """只提取当前指标字典实际存在的键，并改名为 SwanLab 层级路径。"""
    combined_keys = {**metric_keys, **COMMON_SWANLAB_KEYS}
    return {
        swanlab_key: metrics[csv_key]
        for csv_key, swanlab_key in combined_keys.items()
        if csv_key in metrics
    }


def target_swanlab_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    """映射带注意力池化的 target decoder 指标。"""
    return _mapped_metrics(metrics, TARGET_SWANLAB_KEYS)


def time_only_swanlab_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    """映射随机时间点监督的 time-only decoder 指标。"""
    return _mapped_metrics(metrics, TIME_ONLY_SWANLAB_KEYS)


class SwanLabLogger:
    """允许关闭的 SwanLab 轻量适配器。

    关闭时 run=None，log 和 finish 都是无操作，从而训练代码无需在每个
    调用点重复判断；开启时才延迟导入 swanlab，未安装依赖会给出明确错误。
    """

    def __init__(self, run: Any | None = None) -> None:
        """保存底层 SwanLab run；None 表示日志已关闭。"""
        self.run = run

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        output_dir: Path,
        job_type: str,
        experiment_suffix: str | None = None,
        project_suffix: str | None = None,
    ) -> "SwanLabLogger":
        """从完整实验配置创建 logger，并把运行目录作为本地日志目录。"""
        logger_config = config.get("swanlab", {})
        if not isinstance(logger_config, Mapping):
            raise ValueError("swanlab config must be a mapping")
        if not _swanlab_enabled_from_config(logger_config):
            return cls()
        try:
            import swanlab
        except ImportError as exc:
            raise RuntimeError("SwanLab logging is enabled but swanlab is not installed") from exc
        experiment_name = logger_config.get("experiment_name")
        if experiment_name is not None and experiment_suffix:
            experiment_name = f"{experiment_name}-{experiment_suffix}"
        project = logger_config.get("project", "pb2023-eeg-deep")
        if project is not None and project_suffix:
            project = f"{project}-{project_suffix}"
        init_kwargs = {
            "project": project,
            "workspace": logger_config.get("workspace"),
            "experiment_name": experiment_name,
            "description": logger_config.get("description"),
            "job_type": job_type,
            "group": logger_config.get("group"),
            "tags": logger_config.get("tags"),
            "mode": logger_config.get("mode"),
            "logdir": str(output_dir / "swanlab"),
            "config": _plain_value(config),
        }
        init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}
        run = swanlab.init(**init_kwargs)
        return cls(run)

    @property
    def enabled(self) -> bool:
        """SwanLab 是否已成功初始化。"""
        return self.run is not None

    def log(self, metrics: Mapping[str, object], step: int | None = None) -> None:
        """记录一组指标；关闭状态下直接返回。"""
        if self.run is None:
            return
        self.run.log(dict(metrics), step=step)

    def finish(self) -> None:
        """结束远端 run；关闭状态下直接返回。"""
        if self.run is None:
            return
        self.run.finish()
