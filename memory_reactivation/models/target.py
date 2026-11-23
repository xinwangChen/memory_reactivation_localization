"""Target decoder assembled from shared encoders, conditioning, attention and heads."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import torch
from torch import nn
from .encoders import EEGEncoder
from .heads import SharedTaskClassifier, IndependentLinearTimeClassifier, IndependentMLPTimeClassifier, FeatureFusion
from .conditioning import SharedResidualSubjectEmbedding, build_subject_embedding, normalize_subject_conditioning_mode, SubjectAdaLNZero
from .attention import ATTENTION_MODES, ATTENTION_QUERY_SOURCES, EFFECTIVE_NORMALIZATIONS, GATE_ACTIVATIONS, GATE_COMBINATIONS, GatedQueryTimeAttentionPooling, GatedTimeAttentionPooling, POOLED_HEAD_MODES, QueryTimeAttentionPooling, TimeAttentionPooling, _combine_gate_weights, _masked_attention_softmax, _validate_effective_normalization, _validate_gate_options

TASK_CLASS_COUNTS = {"pitch": 5, "choice": 2, "motor": 2}

def normalize_task_class_indices(
    value: Mapping[str, Sequence[int]] | None,
) -> dict[str, tuple[int, ...]]:
    """返回每个任务“内部类别 -> 原始类别下标”的稳定映射。"""
    resolved = {
        task: tuple(range(class_count))
        for task, class_count in TASK_CLASS_COUNTS.items()
    }
    if value is None:
        return resolved
    if not isinstance(value, Mapping):
        raise ValueError("task_class_indices must be a mapping")
    unknown = sorted(set(str(task) for task in value) - set(TASK_CLASS_COUNTS))
    if unknown:
        raise ValueError(f"Unknown task_class_indices tasks: {unknown}")
    for task, raw_indices in value.items():
        task = str(task)
        if isinstance(raw_indices, (str, bytes)) or not isinstance(
            raw_indices, Sequence
        ):
            raise ValueError(f"task_class_indices.{task} must be an integer sequence")
        indices: list[int] = []
        for raw_index in raw_indices:
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise ValueError(
                    f"task_class_indices.{task} must contain only integers"
                )
            indices.append(int(raw_index))
        if len(indices) < 2:
            raise ValueError(
                f"task_class_indices.{task} must retain at least two classes"
            )
        if len(set(indices)) != len(indices):
            raise ValueError(f"task_class_indices.{task} contains duplicates")
        if any(not 0 <= index < TASK_CLASS_COUNTS[task] for index in indices):
            raise ValueError(
                f"task_class_indices.{task} contains an out-of-range class"
            )
        resolved[task] = tuple(sorted(indices))
    return resolved

class TargetAttentionDecoder(nn.Module):
    """同时执行 time-level 与 trial-level 分类的单阶段 EEG decoder。

    数据流为 ``[B,C,T] -> encoder -> [B,T,D]``。完整时间轴先进入时间分类
    头，得到 ``[B,T,Kt]``；attention 再在完整轴或 ``attention_indices`` 指定
    的候选区间上汇聚，得到 ``[B,D]`` trial 表征和 ``[B,K]`` pooled logits。
    ``Kt=K``，或在显式启用 no-class 时为 ``K+1``。可选 fusion 只对 pooled
    表征做 MLP 变换，其输出仍为原始 K 类。

    ``time_head_mode=shared`` 在所有 t 复用一个分类器；两个 independent 模式
    则为每个时间点保存独立参数，并严格要求运行时 T 与构造时一致。Top-q 不在
    本类中裁剪 time logits，而由训练代码按返回的 effective attention 排序后
    选择参与 time loss 的位置。
    """

    def __init__(
        self,
        encoder: EEGEncoder,
        attention_dim: int | None = None,
        attention_dropout: float = 0.0,
        attention_mode: str = "scorer",
        attention_query_source: str = "subject",
        attention_gate_init_bias: float = 2.0,
        n_subjects: int | None = None,
        subject_conditioning_enabled: bool = False,
        subject_conditioning_mode: str = "individual",
        subject_training_ids: Sequence[int] | None = None,
        subject_embedding_dim: int | None = None,
        subject_condition_dropout: float = 0.0,
        subject_adaln_dropout: float = 0.0,
        time_source_block: int | None = None,
        time_head_mode: str = "shared",
        time_point_count: int | None = None,
        time_no_class_enabled: bool = False,
        reconstruction_enabled: bool = False,
        pooled_head_mode: str = "auto",
        pool_valid_context_only: bool = False,
        task_class_indices: Mapping[str, Sequence[int]] | None = None,
        classifier_num_layers: int = 2,
        classifier_hidden_dim: int | None = None,
        classifier_dropout: float = 0.0,
        fusion_enabled: bool = False,
        fusion_hidden_dim: int | None = None,
        fusion_dropout: float = 0.0,
        attention_score_centering: bool = False,
        attention_gate_centering: bool = False,
        attention_gate_activation: str = "sigmoid",
        attention_gate_combination: str = "multiply",
        attention_effective_normalization: str = "none",
    ) -> None:
        """构建编码、被试条件、时间头、attention pooling 与可选 fusion。

        Args:
            encoder: 保持时间长度的 :class:`EEGEncoder`。
            attention_dim: scorer 隐维度或 query/key 投影维度。
            attention_dropout: gated 模式的 pooling 权重 dropout，或 scorer
                模式打分器内部 dropout。
            attention_mode: ``scorer``、``gated_scorer``、``query`` 或
                ``gated_query``。
            attention_query_source: query 模式的条件来源，目前仅支持 subject。
            attention_gate_init_bias: sigmoid gate 的初始 bias。
            n_subjects: subject embedding 表大小。
            subject_conditioning_enabled: 是否在各 encoder block 后启用 AdaLN。
            subject_conditioning_mode: individual 保持历史行为；shared_plus_residual
                使用可学习共享条件加个体偏移，未见被试只使用共享条件。
            subject_training_ids: 新模式实际参与训练的被试ID，随 state_dict 保存。
            subject_embedding_dim: 被试条件维度 ``S``，省略时等于 D。
            subject_condition_dropout: 按 trial 丢弃整条个体向量的概率；新模式在
                相加前只丢弃个体偏移，保留共享条件，不对保留项放大。
            subject_adaln_dropout: AdaLN 条件网络内 dropout。
            time_source_block: 时间头读取的 block（从 1 计数）；省略则读取
                encoder 最终 output norm 后的 latent。
            time_head_mode: 共享头、逐点独立线性头或逐点独立 MLP。
            time_point_count: independent 模式所需的实际时间点数 T。
            time_no_class_enabled: 是否仅为 time-level 增加第 K 个 no-class 输出。
            pooled_head_mode: auto 沿用历史规则；shared_real_class 让 pooled
                复用共享 time head 的前 K 类参数。
            pool_valid_context_only: attention softmax 是否排除 encoder 无法获得
                完整左右上下文的首尾 token；time logits 仍覆盖完整时间轴。
            task_class_indices: 每个内部类别对应的原始类别下标；缺省使用完整任务。
            reconstruction_enabled: Enable the shared linear EEG reconstruction head.
            classifier_num_layers: 共享/MLP 分类器层数。
            classifier_hidden_dim: 两层分类器的中间维度。
            classifier_dropout: 两层分类器的中间 dropout。
            fusion_enabled: 是否启用额外 trial-level fusion 分支。
            fusion_hidden_dim: fusion MLP 中间维度。
            fusion_dropout: fusion MLP dropout。
            attention_score_centering: 仅 scorer 输入减去每条 trial 完整时间轴的
                latent 均值；time head 和实际汇聚值保持原始局部特征。
                仅支持 scorer/gated_scorer，默认关闭，且不增加参数。
            attention_gate_centering: gate 输入独立选择是否减去同一个时间均值；
                默认关闭，仅支持 gated_scorer，不改变 gate bias、激活、
                time head 或实际汇聚值，也不增加参数。
            attention_gate_activation: sigmoid（历史默认）、tanh 或 tanh_plus_one。
                非默认值仅支持 gated_scorer；激活直接读取 gate 线性输出。
            attention_gate_combination: multiply（历史默认）或 add，均在 scorer
                softmax 之后执行。非默认值仅支持 gated_scorer。
            attention_effective_normalization: none 保留历史行为；one_plus_mass
                在聚合/dropout 前对非负 effective weights 执行 w/(1+sum(w))。
                softmax 在相同位置对合成后的w沿有效时间轴执行softmax（温度1）。
                loss、Top-q 和导出均使用处理后的实际权重，仅支持 gated_scorer。
        """
        super().__init__()
        self.encoder = encoder
        self._encoder_frozen = False
        self._subject_conditioning_frozen = False
        if attention_mode not in ATTENTION_MODES:
            valid = ", ".join(sorted(ATTENTION_MODES))
            raise ValueError(f"Unknown attention_mode: {attention_mode}; expected one of {valid}")
        if not isinstance(attention_score_centering, bool):
            raise ValueError("attention_score_centering must be a boolean")
        if attention_score_centering and attention_mode not in {"scorer", "gated_scorer"}:
            raise ValueError("attention_score_centering requires scorer or gated_scorer attention")
        if not isinstance(attention_gate_centering, bool):
            raise ValueError("attention_gate_centering must be a boolean")
        if attention_gate_centering and attention_mode != "gated_scorer":
            raise ValueError("attention_gate_centering requires gated_scorer attention")
        _validate_gate_options(attention_gate_activation, attention_gate_combination)
        _validate_effective_normalization(attention_effective_normalization, attention_gate_activation)
        if attention_effective_normalization != "none" and attention_mode != "gated_scorer":
            raise ValueError("attention_effective_normalization requires gated_scorer attention")
        if (attention_gate_activation != "sigmoid" or attention_gate_combination != "multiply") and attention_mode != "gated_scorer":
            raise ValueError("Non-default attention gate options require gated_scorer attention")
        if not 0.0 <= float(subject_condition_dropout) <= 1.0:
            raise ValueError("subject_condition_dropout must be between 0 and 1")
        self.attention_mode = attention_mode
        self.attention_score_centering = attention_score_centering
        self.attention_gate_centering = attention_gate_centering
        self.attention_gate_activation = attention_gate_activation
        self.attention_gate_combination = attention_gate_combination
        self.attention_effective_normalization = attention_effective_normalization
        self.subject_conditioning_enabled = bool(subject_conditioning_enabled)
        self.subject_conditioning_mode = normalize_subject_conditioning_mode(subject_conditioning_mode)
        self.subject_condition_dropout = float(subject_condition_dropout)
        self.subject_dim = int(subject_embedding_dim) if subject_embedding_dim is not None else encoder.latent_dim
        needs_subject_embedding = self.subject_conditioning_enabled or attention_mode in {"query", "gated_query"}
        if needs_subject_embedding:
            if self.subject_dim < 1:
                raise ValueError("subject_embedding_dim must be positive")
            self.subject_embedding = build_subject_embedding(n_subjects, self.subject_dim,
                mode=self.subject_conditioning_mode, training_subject_ids=subject_training_ids)
        else:
            self.subject_embedding = None
        self.subject_adapters = (
            nn.ModuleList(
                [
                    SubjectAdaLNZero(
                        latent_dim=encoder.latent_dim,
                        condition_dim=self.subject_dim,
                        dropout=subject_adaln_dropout,
                    )
                    for _ in range(encoder.num_blocks)
                ]
            )
            if self.subject_conditioning_enabled
            else None
        )
        self.time_source_block = None if time_source_block is None else int(time_source_block)
        if self.time_source_block is not None and (
            self.time_source_block < 1 or self.time_source_block > encoder.num_blocks
        ):
            raise ValueError(
                "time_source_block must be null or between 1 and encoder.num_blocks, "
                f"got {self.time_source_block} for {encoder.num_blocks} blocks"
            )
        self.time_head_mode = str(time_head_mode).lower().replace("-", "_")
        valid_time_head_modes = {
            "shared",
            "independent_linear",
            "independent_mlp",
        }
        if self.time_head_mode not in valid_time_head_modes:
            valid = ", ".join(sorted(valid_time_head_modes))
            raise ValueError(f"time_head_mode must be one of: {valid}")
        if self.time_head_mode.startswith("independent_"):
            if time_point_count is None or int(time_point_count) < 1:
                raise ValueError(
                    "time_point_count must be positive for an independent time head"
                )
            # T 来自数据加载阶段解析出的真实时间轴，不写死为常见的 275 点。
            self.time_point_count = int(time_point_count)
        else:
            self.time_point_count = None
        if not isinstance(time_no_class_enabled, bool):
            raise ValueError("time_no_class_enabled must be a boolean")
        self.time_no_class_enabled = time_no_class_enabled
        self.pooled_head_mode = str(pooled_head_mode).strip().lower().replace("-", "_")
        if self.pooled_head_mode not in POOLED_HEAD_MODES:
            valid_modes = ", ".join(sorted(POOLED_HEAD_MODES))
            raise ValueError(f"pooled_head_mode must be one of: {valid_modes}")
        if not isinstance(pool_valid_context_only, bool):
            raise ValueError("pool_valid_context_only must be a boolean")
        self.pool_valid_context_only = pool_valid_context_only
        if self.pooled_head_mode == "shared_real_class":
            if self.attention_mode.startswith("gated_"):
                raise ValueError(
                    "shared_real_class pooled head requires attention weights "
                    "that sum to one; gated attention is unsupported"
                )
            if self.time_head_mode != "shared":
                raise ValueError("shared_real_class pooled head requires time_head_mode=shared")
            if not self.time_no_class_enabled:
                raise ValueError(
                    "shared_real_class pooled head is only meaningful with time_no_class_enabled=true"
                )
            if int(classifier_num_layers) != 1:
                raise ValueError(
                    "shared_real_class pooled head requires classifier_num_layers=1 "
                    "for exact time-logit pooling equivalence"
                )
        self.task_class_indices = normalize_task_class_indices(task_class_indices)
        self.task_class_counts = {
            task: len(indices) for task, indices in self.task_class_indices.items()
        }
        # time-level 可为 K+1；全局常量仍只描述未筛选任务的默认类别数。
        self.time_class_counts = {
            task: class_count + int(self.time_no_class_enabled)
            for task, class_count in self.task_class_counts.items()
        }
        if attention_mode == "scorer":
            self.attention = TimeAttentionPooling(
                latent_dim=encoder.latent_dim,
                attention_dim=attention_dim,
                dropout=attention_dropout,
            )
        elif attention_mode == "gated_scorer":
            self.attention = GatedTimeAttentionPooling(
                latent_dim=encoder.latent_dim,
                attention_dim=attention_dim,
                dropout=attention_dropout,
                gate_init_bias=attention_gate_init_bias,
                gate_activation=attention_gate_activation,
                gate_combination=attention_gate_combination,
                effective_normalization=attention_effective_normalization,
            )
        elif attention_mode == "query":
            self.attention = QueryTimeAttentionPooling(
                latent_dim=encoder.latent_dim,
                subject_dim=self.subject_dim,
                attention_dim=attention_dim,
                query_source=attention_query_source,
            )
        else:
            self.attention = GatedQueryTimeAttentionPooling(
                latent_dim=encoder.latent_dim,
                subject_dim=self.subject_dim,
                attention_dim=attention_dim,
                dropout=attention_dropout,
                gate_init_bias=attention_gate_init_bias,
                query_source=attention_query_source,
            )
        # class_heads 始终是 time-level 头，输出维度使用 time_class_counts。
        if self.time_head_mode == "independent_linear":
            assert self.time_point_count is not None
            self.class_heads = nn.ModuleDict(
                {
                    task: IndependentLinearTimeClassifier(
                        time_point_count=self.time_point_count,
                        latent_dim=encoder.latent_dim,
                        n_classes=n_classes,
                    )
                    for task, n_classes in self.time_class_counts.items()
                }
            )
        elif self.time_head_mode == "independent_mlp":
            assert self.time_point_count is not None
            self.class_heads = nn.ModuleDict(
                {
                    task: IndependentMLPTimeClassifier(
                        time_point_count=self.time_point_count,
                        latent_dim=encoder.latent_dim,
                        n_classes=n_classes,
                        num_layers=classifier_num_layers,
                        hidden_dim=classifier_hidden_dim,
                        dropout=classifier_dropout,
                    )
                    for task, n_classes in self.time_class_counts.items()
                }
            )
        else:
            self.class_heads = nn.ModuleDict(
                {
                    task: SharedTaskClassifier(
                        latent_dim=encoder.latent_dim,
                        n_classes=n_classes,
                        num_layers=classifier_num_layers,
                        hidden_dim=classifier_hidden_dim,
                        dropout=classifier_dropout,
                    )
                    for task, n_classes in self.time_class_counts.items()
                }
            )
        # trial-level 始终只预测真实 K 类。独立时间头无法接收 [B,D]，而 K+1
        # 时间头的输出维度也不适合 pooled，因此这两种情况必须建立独立参数。
        # 反之，共享且未启用 no-class 时继续复用历史 class_heads：既保持原有
        # 参数共享语义，也避免凭空增加 pooled_heads.* state_dict 键，兼容旧
        # checkpoint 的严格加载布局。
        self.pooled_heads = (
            nn.ModuleDict(
                {
                    task: SharedTaskClassifier(
                        latent_dim=encoder.latent_dim,
                        n_classes=n_classes,
                        num_layers=classifier_num_layers,
                        hidden_dim=classifier_hidden_dim,
                        dropout=classifier_dropout,
                    )
                    for task, n_classes in self.task_class_counts.items()
                }
            )
            if self.pooled_head_mode == "auto"
            and (
                self.time_head_mode.startswith("independent_")
                or self.time_no_class_enabled
            )
            else None
        )
        self.fusion = (
            FeatureFusion(
                latent_dim=encoder.latent_dim,
                hidden_dim=fusion_hidden_dim,
                dropout=fusion_dropout,
            )
            if fusion_enabled
            else None
        )
        self.fusion_heads = (
            nn.ModuleDict(
                {
                    task: nn.Linear(encoder.latent_dim, n_classes)
                    for task, n_classes in self.task_class_counts.items()
                }
            )
            if fusion_enabled
            else None
        )
        if not isinstance(reconstruction_enabled, bool):
            raise ValueError("reconstruction_enabled must be a boolean")
        # Instantiate after existing heads so enabling this auxiliary branch does
        # not change their seeded initialization.
        self.reconstruction_head = (
            nn.Linear(encoder.latent_dim, encoder.in_channels)
            if reconstruction_enabled
            else None
        )

    def configure_frozen_components(
        self,
        *,
        encoder: bool = False,
        subject_conditioning: bool = False,
    ) -> None:
        """冻结指定组件，并让其中的随机层持续处于 eval 模式。

        ``requires_grad_(False)`` 负责阻止参数更新；额外调用 ``eval`` 可防止
        encoder/subject 模块中的 dropout 在冻结后仍随机变化。该状态由本类的
        :meth:`train` 覆盖逻辑维护，不影响 attention 与分类头的训练模式。
        """
        freeze_encoder = bool(encoder)
        freeze_subject = bool(subject_conditioning)
        subject_modules = [
            module
            for module in (self.subject_embedding, self.subject_adapters)
            if module is not None
        ]
        if freeze_subject and not subject_modules:
            raise ValueError(
                "Cannot freeze subject conditioning because the model has no "
                "subject embedding/adapters"
            )
        self.encoder.requires_grad_(not freeze_encoder)
        for module in subject_modules:
            module.requires_grad_(not freeze_subject)
        self._encoder_frozen = freeze_encoder
        self._subject_conditioning_frozen = freeze_subject
        if self._encoder_frozen:
            self.encoder.eval()
        if self._subject_conditioning_frozen:
            for module in subject_modules:
                module.eval()

    def train(self, mode: bool = True) -> "TargetAttentionDecoder":
        """切换训练模式，同时强制已冻结的子模块保持 eval。"""
        super().train(mode)
        if self._encoder_frozen:
            self.encoder.eval()
        if self._subject_conditioning_frozen:
            if self.subject_embedding is not None:
                self.subject_embedding.eval()
            if self.subject_adapters is not None:
                self.subject_adapters.eval()
        return self

    @property
    def fusion_enabled(self) -> bool:
        """仅当 fusion 表征网络和分类头都存在时返回 True。"""
        return self.fusion is not None and self.fusion_heads is not None

    @property
    def reconstruction_enabled(self) -> bool:
        """Return whether the shared EEG reconstruction head is present."""
        return self.reconstruction_head is not None

    def time_class_count(self, task: str) -> int:
        """返回指定任务 time-level 输出数 ``K`` 或 ``K+1``。"""
        if task not in self.task_class_counts:
            raise ValueError(f"Unknown task: {task}")
        return self.time_class_counts[task]

    def class_count(self, task: str) -> int:
        """返回指定任务当前模型实际使用的真实类别数 K。"""
        if task not in self.task_class_counts:
            raise ValueError(f"Unknown task: {task}")
        return self.task_class_counts[task]

    def class_indices(self, task: str) -> tuple[int, ...]:
        """返回内部连续标签到原始任务类别下标的映射。"""
        if task not in self.task_class_indices:
            raise ValueError(f"Unknown task: {task}")
        return self.task_class_indices[task]

    def time_no_class_index(self, task: str) -> int | None:
        """返回额外 no-class 的索引 K；未启用时返回 ``None``。"""
        if task not in self.task_class_counts:
            raise ValueError(f"Unknown task: {task}")
        if not self.time_no_class_enabled:
            return None
        return self.task_class_counts[task]

    def _subject_condition(self, subject_id: torch.Tensor | None, batch_size: int | None = None) -> torch.Tensor | None:
        """把 ``[B]`` 被试编号映射为 ``[B,S]`` 条件向量。

        dropout 按 trial 采样 ``[B,1]`` mask；individual 模式丢弃整个被试条件，
        shared_plus_residual 只在相加前丢弃个体偏移。冻结条件模块时关闭该随机失活。
        """
        if self.subject_embedding is None:
            return None
        if isinstance(self.subject_embedding, SharedResidualSubjectEmbedding):
            if subject_id is None:
                if batch_size is None:
                    raise ValueError("batch_size is required when subject_id is omitted")
                return self.subject_embedding.shared_condition[None, :].expand(batch_size, -1)
            if batch_size is not None and subject_id.shape != (batch_size,):
                raise ValueError("subject_id must have shape [batch]")
            return self.subject_embedding(subject_id, condition_dropout=(
                0. if self._subject_conditioning_frozen else self.subject_condition_dropout))
        if subject_id is None:
            raise ValueError("subject_id is required when subject embedding is enabled")
        condition = self.subject_embedding(subject_id.to(dtype=torch.long, device=self.subject_embedding.weight.device))
        if (
            self.training
            and not self._subject_conditioning_frozen
            and self.subject_condition_dropout > 0.0
        ):
            keep = torch.rand(condition.shape[0], 1, device=condition.device) >= self.subject_condition_dropout
            condition = condition * keep.to(dtype=condition.dtype)
        return condition

    def forward(
        self,
        eeg: torch.Tensor,
        task: str,
        attention_indices: torch.Tensor | None = None,
        subject_id: torch.Tensor | None = None,
        *,
        boundary_noise_eeg: torch.Tensor | None = None,
        boundary_trial_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """执行完整时间分类、attention trial 汇聚与可选 fusion。

        Args:
            eeg: ``[B,C,T]`` EEG；其中噪声增强位置仍是有效输入位置。
            task: ``pitch``、``choice`` 或 ``motor``。
            attention_indices: 可选 ``[Ta]`` 时间索引，只限定 attention/Top-q
                的候选区间；不会裁剪完整 ``time_logits``，也不会重排独立头编号。
            subject_id: ``[B]`` 被试ID。新模式中训练名单以外的ID只使用共享条件，
                ``None`` 表示整批使用共享条件；旧 individual 模式仍必须提供。
            boundary_noise_eeg: 可选的噪声 pad ``[B,C,2r]``，仅传给 encoder。
            boundary_trial_ids: eval 自动采样噪声 pad 所需的稳定 trial 身份。

        Returns:
            ``time_logits`` 为 ``[B,T,Kt]``，``pooled_time_latent`` 为 ``[B,D]``，
            ``pooled_logits`` 为 ``[B,K]``；``attention_weights`` 与
            ``attention_scores`` 为 ``[B,Ta]``（未给索引时 ``Ta=T``）。gated
            模式额外返回同形状的 ``attention_gate`` 和
            ``effective_attention_weights``。启用 fusion 时再返回
            ``fusion_latent [B,D]`` 与 ``fusion_logits [B,K]``。
        """
        if task not in self.task_class_counts:
            raise ValueError(f"Unknown task: {task}")
        # eeg: [B,C,T]；subject_id: [B] long（启用 subject 条件时）
        # attention_indices: [Ta] 全局时间下标，只给 pooling/Top-q，不改 time 轴长度
        subject_condition = self._subject_condition(subject_id, batch_size=eeg.shape[0])  # [B,S] 或 None
        if self.encoder.boundary_mode == "noise_pad" and boundary_noise_eeg is None:
            # 冻结 encoder 会令其保持 eval；补边的随机策略仍跟随整个训练任务。
            boundary_noise_eeg = self.encoder.sample_boundary_noise(
                eeg, trial_ids=boundary_trial_ids, training=self.training,
            )
        features = self.encoder.forward_features(
            eeg,
            subject_adapters=self.subject_adapters,
            subject_condition=subject_condition,
            boundary_noise_eeg=boundary_noise_eeg,
            boundary_trial_ids=boundary_trial_ids,
        )
        if self.time_source_block is None:
            latent = features["latent"]  # [B,T,D] 全时间轴隐表征
        else:
            block_latents = features["block_latents"]  # list 长度=num_blocks，每项 [B,T,D]
            if not block_latents:
                raise ValueError("TargetAttentionDecoder requires at least one encoder block")
            latent = block_latents[self.time_source_block - 1]  # 配置 1..L → 下标 0..L-1
        if not isinstance(latent, torch.Tensor):
            raise TypeError("encoder.forward_features returned an invalid latent tensor")

        # Kt = K 或 K+1(no-class)；t 与独立头一一对应，故必须用完整 T
        time_logits = self.class_heads[task](latent)  # [B,T,Kt]
        attention_latent = (
            latent.index_select(dim=1, index=attention_indices)  # [B,Ta,D] Ta≤T
            if attention_indices is not None
            else latent
        )
        centered_attention_latent = None
        if self.attention_score_centering or self.attention_gate_centering:
            # 每条 trial、每个特征维各自沿完整 T 求均值，不跨 batch。
            # attention_indices / pool-valid mask 只限制汇聚支持，不缩短背景参照。
            # 两个分支共享同一个背景参照且均保留梯度；低精度用 float32 去均值。
            reference = (
                latent.float()
                if latent.dtype in (torch.float16, torch.bfloat16)
                else latent
            )
            centered_attention_latent = (
                attention_latent - reference.mean(dim=1, keepdim=True)
            ).to(dtype=attention_latent.dtype)
        attention_score_latent = centered_attention_latent if self.attention_score_centering else None
        attention_gate_latent = centered_attention_latent if self.attention_gate_centering else None
        attention_valid_mask = None
        if self.pool_valid_context_only:
            full_valid_mask = torch.ones(
                latent.shape[1], dtype=torch.bool, device=latent.device
            )
            left = int(self.encoder.left_context_samples)
            right = int(self.encoder.right_context_samples)
            # noise_pad 已补全每个真实 query 的指定局部窗口；不能再次按旧
            # 截短边界规则移除真实首尾点。pad 本身从未进入 latent。
            if left and self.encoder.boundary_mode != "noise_pad":
                full_valid_mask[:left] = False
            if right and self.encoder.boundary_mode != "noise_pad":
                full_valid_mask[-right:] = False
            attention_valid_mask = (
                full_valid_mask.index_select(dim=0, index=attention_indices)
                if attention_indices is not None
                else full_valid_mask
            )
        attention_gate = None  # gated 时 [B,Ta]
        effective_attention_weights = None  # gated 时 [B,Ta]；训练 Top-q 优先用它
        if self.attention_mode == "query":
            assert isinstance(self.attention, QueryTimeAttentionPooling)
            pooled_latent, attention_weights, attention_scores = self.attention(
                attention_latent,
                subject_condition=subject_condition,
                valid_mask=attention_valid_mask,
            )  # pooled:[B,D] weights/scores:[B,Ta]
        elif self.attention_mode == "gated_query":
            assert isinstance(self.attention, GatedQueryTimeAttentionPooling)
            (
                pooled_latent,  # [B,D]
                attention_weights,  # [B,Ta] softmax
                attention_scores,  # [B,Ta] 未归一
                attention_gate,  # [B,Ta] sigmoid
                effective_attention_weights,  # [B,Ta] = softmax*gate
            ) = self.attention(
                attention_latent,
                subject_condition=subject_condition,
                valid_mask=attention_valid_mask,
            )
        elif self.attention_mode == "gated_scorer":
            assert isinstance(self.attention, GatedTimeAttentionPooling)
            (
                pooled_latent,
                attention_weights,
                attention_scores,
                attention_gate,
                effective_attention_weights,
            ) = self.attention(
                attention_latent,
                valid_mask=attention_valid_mask,
                score_latent=attention_score_latent,
                gate_latent=attention_gate_latent,
            )
        else:
            pooled_latent, attention_weights, attention_scores = self.attention(
                attention_latent,
                valid_mask=attention_valid_mask,
                score_latent=attention_score_latent,
            )
        # independent/no-class 时 pooled 头参数独立；否则 shared 头复用 class_heads
        if self.pooled_head_mode == "shared_real_class":
            pooled_logits = self.class_heads[task](pooled_latent)[
                ..., : self.task_class_counts[task]
            ]
        else:
            pooled_head = (
                self.pooled_heads[task]
                if self.pooled_heads is not None
                else self.class_heads[task]
            )
            pooled_logits = pooled_head(pooled_latent)  # [B,K] 真实类数，不含 no-class
        outputs = {
            "time_logits": time_logits,  # [B,T,Kt]
            "pooled_time_latent": pooled_latent,  # [B,D]
            "pooled_logits": pooled_logits,  # [B,K]
            "attention_weights": attention_weights,  # [B,Ta]
            "attention_scores": attention_scores,  # [B,Ta]
        }
        if self.reconstruction_enabled:
            assert self.reconstruction_head is not None
            outputs["reconstructed_eeg"] = self.reconstruction_head(latent).transpose(1, 2)
        if attention_gate is not None:
            outputs["attention_gate"] = attention_gate  # [B,Ta]
        if effective_attention_weights is not None:
            outputs["effective_attention_weights"] = effective_attention_weights  # [B,Ta]
            if self.attention_effective_normalization != "none":
                outputs["pre_normalization_effective_attention_weights"] = _combine_gate_weights(
                    attention_weights, attention_gate, self.attention_gate_combination, attention_valid_mask
                )
            if self.attention_gate_activation == "tanh" and self.attention_effective_normalization != "softmax":
                # 保留实际 pooling 的正负号；质量约束和 Top-q 看贡献强度，
                # 避免负权重逃避噪声惩罚，或因排序符号而丢失强负贡献。
                magnitude_weights = effective_attention_weights.abs()
                outputs["attention_magnitude_weights"] = magnitude_weights
                outputs["time_selection_attention_weights"] = magnitude_weights
        if attention_valid_mask is not None:
            outputs["attention_valid_mask"] = attention_valid_mask  # [Ta] pool-only support
            # Top-q 排序权重与 pooling 权重完全对齐：在 pool-valid 子集上重归一化，
            # invalid 位置严格为 0。否则 Top-q 会选中 pooling 明确排除的边界位置，
            # 使 time 分支的监督/指标与 trial 分支的实际汇聚对象互相矛盾。
            time_selection_weights = attention_weights
            if effective_attention_weights is not None:
                time_selection_weights = outputs.get(
                    "attention_magnitude_weights", effective_attention_weights
                )
            outputs["time_selection_attention_weights"] = time_selection_weights
        if self.fusion_enabled:
            assert self.fusion is not None and self.fusion_heads is not None
            fusion_latent = self.fusion(pooled_latent)  # [B,D]
            outputs["fusion_latent"] = fusion_latent
            outputs["fusion_logits"] = self.fusion_heads[task](fusion_latent)  # [B,K]
        return outputs

