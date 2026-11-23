"""分类头组件。

本模块只负责把编码器产生的隐表示映射为分类 logits，不在这里计算 CE/BCE
损失或执行 Top-q 选择。全文统一使用以下张量维度记号： ``B`` 为 batch
大小，``T`` 为时间点数，``D`` 为隐空间维度，``H`` 为 MLP 中间维度，
``K`` 为当前分类头的输出类别数。对于启用 no-class 的 time-level 任务，
``K`` 已经包含额外的 no-class 输出；trial-level 的类别数由 decoder 单独决定。
"""

from __future__ import annotations

import torch
from torch import nn


class SharedTaskClassifier(nn.Module):
    """在任意前导维上复用同一组参数的任务分类头。

    输入最后一维必须是 ``D``，其余前导维原样保留。因此它既可接收
    trial-level 的 ``[B, D]``，也可接收 time-level 的 ``[B, T, D]``；
    对应输出分别为 ``[B, K]`` 与 ``[B, T, K]``。当用于时间轴时，所有
    时间点共享参数，这一点与两个 ``Independent*TimeClassifier`` 不同。
    """

    def __init__(
        self,
        latent_dim: int,
        n_classes: int,
        num_layers: int = 2,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        """构建一层线性头或两层 MLP 头。

        Args:
            latent_dim: 输入隐表示维度 ``D``。
            n_classes: 输出类别数 ``K``。
            num_layers: ``1`` 表示单个线性层，``2`` 表示 MLP。
            hidden_dim: 两层 MLP 的中间维度 ``H``；省略时等于 ``D``。
            dropout: 两层 MLP 中间的 dropout 概率。
        """
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_classes = int(n_classes)
        self.num_layers = int(num_layers)
        if self.num_layers not in {1, 2}:
            raise ValueError("classifier_num_layers must be 1 or 2")
        hidden = int(hidden_dim) if hidden_dim is not None else self.latent_dim
        if hidden < 1:
            raise ValueError("classifier_hidden_dim must be positive")
        if self.num_layers == 1:
            self.net = nn.Linear(self.latent_dim, self.n_classes)  # 单层：可兼容旧 ckpt 键名
        else:
            self.net = nn.Sequential(
                nn.Linear(self.latent_dim, hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, self.n_classes),
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """显式初始化全部线性层：权重采用 fan-in 均匀分布，bias 置零。"""
        for module in self.net.modules():
            if not isinstance(module, nn.Linear):
                continue
            bound = module.in_features**-0.5
            nn.init.uniform_(module.weight, -bound, bound)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """将 ``[..., D]`` 映射为 ``[..., K]`` logits。"""
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected last dim {self.latent_dim}, got {latent.shape}")
        return self.net(latent)  # 可吃 [B,D] 或 [B,T,D]；T 上共享同一组参数

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """兼容加载历史单层分类头的参数键。

        旧 checkpoint 把单层头保存为 ``<task>.weight/bias``，当前结构则因
        线性层位于 ``net`` 中而保存为 ``<task>.net.weight/bias``。这里仅在
        ``num_layers == 1`` 时原位迁移字典键，使新模型可严格加载旧权重；
        新模型自身的 ``state_dict`` 键仍保持当前格式。
        """
        # 如果新旧键同时存在，无法判断应以哪份参数为准，交由 PyTorch 以
        # 明确错误终止，而不是静默覆盖 checkpoint 内容。
        if self.num_layers == 1:
            for name in ("weight", "bias"):
                legacy_key = f"{prefix}{name}"
                current_key = f"{prefix}net.{name}"
                if legacy_key in state_dict:
                    if current_key in state_dict:
                        error_msgs.append(
                            "Checkpoint contains both historical and current keys "
                            f"for {prefix.rstrip('.')}: {legacy_key}, {current_key}"
                        )
                    else:
                        state_dict[current_key] = state_dict.pop(legacy_key)  # 旧: .weight → 新: .net.weight
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class IndependentLinearTimeClassifier(nn.Module):
    """为每个时间点分配一套互不共享的线性分类参数。

    参数以张量形式向量化保存，并不意味着时间点之间共享权重。第 ``t`` 个
    输入 ``latent[:, t, :]`` 只与 ``weight[t]`` 和 ``bias[t]`` 运算，得到
    ``logits[:, t, :]``。头的数量等于运行时数据解析出的 ``T``，不假定固定
    为 275；因此输入时间长度必须与构造时的 ``time_point_count`` 完全一致。
    """

    def __init__(self, time_point_count: int, latent_dim: int, n_classes: int) -> None:
        """初始化 ``T`` 个独立的 ``D -> K`` 线性映射。"""
        super().__init__()
        self.time_point_count = int(time_point_count)
        self.latent_dim = int(latent_dim)
        self.n_classes = int(n_classes)
        if self.time_point_count < 1:
            raise ValueError("time_point_count must be positive")
        # ``weight`` 为 [T,D,K]，``bias`` 为 [T,K]；使用单个 Parameter 张量
        # 只是为了高效并行计算，并不会让不同 t 的切片共享可训练元素。
        # 形状 [T,D,K]/[T,K]：一整个 Parameter，但 t 切片之间不共享元素
        self.weight = nn.Parameter(
            torch.empty(self.time_point_count, self.latent_dim, self.n_classes)
        )
        self.bias = nn.Parameter(torch.empty(self.time_point_count, self.n_classes))
        bound = self.latent_dim**-0.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.zeros_(self.bias)  # 无随机类别先验

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """用各时间点自己的参数将 ``[B,T,D]`` 映射为 ``[B,T,K]``。"""
        self._validate_input(latent)  # T 必须等于构造时的 time_point_count
        # t 维不共享：latent[:,t] 只乘 weight[t]
        return torch.einsum("btd,tdc->btc", latent, self.weight) + self.bias

    def _validate_input(self, latent: torch.Tensor) -> None:
        """检查 rank、时间点数 ``T`` 与隐维度 ``D``。"""
        if latent.ndim != 3:
            raise ValueError(f"Expected [batch, time, latent] input, got {latent.shape}")
        if latent.shape[1:] != (self.time_point_count, self.latent_dim):
            raise ValueError(
                "Independent time head expected "
                f"[batch, {self.time_point_count}, {self.latent_dim}], got {latent.shape}"
            )


class IndependentMLPTimeClassifier(nn.Module):
    """为每个时间点分配独立参数的向量化 MLP 分类头。

    一层模式等价于 :class:`IndependentLinearTimeClassifier`；两层模式中，
    输入投影 ``[D,H]`` 和输出投影 ``[H,K]`` 都带独立的 ``T`` 维，因此
    相邻时间点之间既不交换特征，也不共享参数。dropout 只作用于各时间点
    自己的中间特征，不改变上述独立性。
    """

    def __init__(
        self,
        time_point_count: int,
        latent_dim: int,
        n_classes: int,
        num_layers: int = 2,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        """初始化 ``T`` 套独立的一层线性头或两层 MLP 头。"""
        super().__init__()
        self.time_point_count = int(time_point_count)
        self.latent_dim = int(latent_dim)
        self.n_classes = int(n_classes)
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else self.latent_dim
        if self.time_point_count < 1:
            raise ValueError("time_point_count must be positive")
        if self.num_layers not in {1, 2}:
            raise ValueError("classifier_num_layers must be 1 or 2")
        if self.hidden_dim < 1:
            raise ValueError("classifier_hidden_dim must be positive")
        self.dropout = nn.Dropout(float(dropout))
        if self.num_layers == 1:
            # 一层模式参数形状为 [T,D,K] 与 [T,K]。
            self.weight = nn.Parameter(
                torch.empty(self.time_point_count, self.latent_dim, self.n_classes)
            )
            self.bias = nn.Parameter(
                torch.empty(self.time_point_count, self.n_classes)
            )
            bound = self.latent_dim**-0.5
            nn.init.uniform_(self.weight, -bound, bound)
            # 与独立线性头及共享头保持一致：初始类别先验由输入特征决定，
            # 不让每个时间点携带随机的固定偏置。
            nn.init.zeros_(self.bias)
        else:
            # 两个投影均显式携带 T 维：[T,D,H] 和 [T,H,K]；中间特征不会
            # 使用相邻时间点的参数。
            self.hidden_weight = nn.Parameter(
                torch.empty(
                    self.time_point_count,
                    self.latent_dim,
                    self.hidden_dim,
                )
            )
            self.hidden_bias = nn.Parameter(
                torch.empty(self.time_point_count, self.hidden_dim)
            )
            self.output_weight = nn.Parameter(
                torch.empty(
                    self.time_point_count,
                    self.hidden_dim,
                    self.n_classes,
                )
            )
            self.output_bias = nn.Parameter(
                torch.empty(self.time_point_count, self.n_classes)
            )
            hidden_bound = self.latent_dim**-0.5
            output_bound = self.hidden_dim**-0.5
            nn.init.uniform_(self.hidden_weight, -hidden_bound, hidden_bound)
            nn.init.zeros_(self.hidden_bias)
            nn.init.uniform_(self.output_weight, -output_bound, output_bound)
            nn.init.zeros_(self.output_bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """用 ``T`` 套独立参数将 ``[B,T,D]`` 映射为 ``[B,T,K]``。"""
        self._validate_input(latent)
        if self.num_layers == 1:
            return torch.einsum("btd,tdc->btc", latent, self.weight) + self.bias
        hidden = (
            torch.einsum("btd,tdh->bth", latent, self.hidden_weight)  # 仍按 t 独立
            + self.hidden_bias
        )
        hidden = self.dropout(torch.nn.functional.gelu(hidden))  # dropout 不跨时间点混
        return (
            torch.einsum("bth,thc->btc", hidden, self.output_weight)
            + self.output_bias
        )

    def _validate_input(self, latent: torch.Tensor) -> None:
        """确保输入恰好为构造时约定的 ``[B,T,D]``。"""
        if latent.ndim != 3:
            raise ValueError(f"Expected [batch, time, latent] input, got {latent.shape}")
        if latent.shape[1:] != (self.time_point_count, self.latent_dim):
            raise ValueError(
                "Independent time head expected "
                f"[batch, {self.time_point_count}, {self.latent_dim}], got {latent.shape}"
            )


class FeatureFusion(nn.Module):
    """对 attention 已汇聚的 trial 表征做可选特征融合。

    该模块只处理 ``[B,D]``，不会重新访问时间轴或混合不同 trial。它产生新的
    ``[B,D]`` fusion latent，随后由独立 fusion 分类头输出原始 ``K`` 类。
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        """构建保持输入输出维度均为 D 的两层 MLP。"""
        super().__init__()
        self.latent_dim = int(latent_dim)
        hidden = int(hidden_dim) if hidden_dim is not None else self.latent_dim
        if hidden < 1:
            raise ValueError("fusion hidden_dim must be positive")
        self.net = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.latent_dim),
            nn.GELU(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """将 pooled ``[B,D]`` 映射为 fusion latent ``[B,D]``。"""
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected [batch, {self.latent_dim}] pooled latent, got {latent.shape}")
        return self.net(latent)
