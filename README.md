# Memory Reactivation Localization

用于 EEG 记忆再激活分析的解码工具。Python 包名为 `memory_reactivation`，安装包名为 `memory-reactivation`。

通过统一的数据、模型和训练接口完成 Choice、Pitch、Motor 解码，并支持逐时间点分类、trial 级分类、注意力分析和可视化。

| 功能 | 实现 |
| --- | --- |
| 数据处理 | manifest、block-CV、subject-CV、训练集归一化、Graph Laplacian、磁盘缓存 |
| 编码器 | CNN、Patch MLP、局部 Transformer |
| 模型组件 | 被试条件化、注意力与门控、时间分类头、pooled 分类头、可选 fusion 与重建头 |
| 训练 | Top-q、no-class、CE/BCE/Focal、噪声增强、正则项、AMP、早停和断点恢复 |
| 优化器 | AdamW、SGD、Muon、Signum，学习率与 weight decay 调度 |
| 评估 | 原始 EEG、Gaussian、时间打乱、时间平移及 trial/被试/时间点/条件单元统计 |
| 模型载入 | 新旧 Target checkpoint 与原生 spatial baseline 的严格加载和预测 |
| 可视化 | 注意力、门控、时间准确率、激活、分类 margin 和分组曲线 |

```text
memory_reactivation/
  models/           编码器、被试条件化、注意力、分类头和模型工厂
  data/             样本组织、交叉验证、归一化、空间处理与缓存
  training/         数据增强、目标函数、batch 计算与训练调度
  evaluation/       模型预测、指标计算与 CSV 导出
  analysis/         数值汇总与曲线分析
  visualization/    图形绘制
  artifacts/        模型读写、兼容加载和导出身份校验
  workflows/        数据准备、训练、推理、重导出与报告入口
scripts/            命令脚本与历史模型对照工具
tests/              单元测试和合成数据回归测试
```

1. 安装环境。

   使用 Python 3.10 及以上版本。

   ```bash
   git clone git@github.com:xinwangChen/memory_reactivation_localization.git
   cd memory_reactivation_localization

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e '.[test]'

   python -m memory_reactivation --help
   ```

   SwanLab 日志依赖通过以下命令安装。

   ```bash
   python -m pip install -e '.[logging]'
   ```

   | 命令 | 用途 |
   | --- | --- |
   | `python -m memory_reactivation prepare` | 构建 manifest、数据划分和归一化统计 |
   | `python -m memory_reactivation train` | 根据 YAML 配置训练指定任务和一折 |
   | `python -m memory_reactivation inspect` | 查看 checkpoint、模型构造信息和参数规模 |
   | `python -m memory_reactivation predict` | 对预处理后的 EEG NPZ 执行批量预测 |
   | `python -m memory_reactivation reexport` | 重新推理、导出评估 CSV 并绘图 |
   | `python -m memory_reactivation report` | 从已有 CSV 生成图表 |

   各子命令均支持 `--help`。

2. 准备数据。

   预处理后的文件按实验和被试组织。

   ```text
   data/preprocessed/
     Exp1/
       Sub01_eeg_clean.npz
       Sub01_meta.csv
       Sub02_eeg_clean.npz
       Sub02_meta.csv
     electrodes/
       Exp1_electrodes.npz
   ```

   EEG NPZ 使用 `X: [N,C,T]` 保存 trial、电极和时间三个维度，`trial0: [N]` 保存原始 trial 编号，`time_s: [T]` 保存以秒为单位的共享时间轴。元数据 CSV 保存 trial 编号、block、有效性及各任务的完整行为标签。

   `data.lags: [0]` 对应当前 trial 标签，`data.lags: [1]` 对应同一 block 内前一 trial 的标签。每次 Target 训练使用一个非负整数 lag。

   block-CV 在各被试内部划分 block，归一化使用训练数据估计。

   ```bash
   python -m memory_reactivation prepare --preprocessed-root data/preprocessed \
     --artifact-dir artifacts --cv-folds 5 --cv-strategy block \
     --cv-output-dir artifacts/cv5 --seed 42
   ```

   subject-CV 按被试划分，使用训练被试估计全局归一化统计。

   ```bash
   python -m memory_reactivation prepare --preprocessed-root data/preprocessed \
     --artifact-dir artifacts --cv-folds 5 --cv-strategy subject \
     --normalization-mode train_global \
     --cv-output-dir artifacts/cv5-subject-seed42 --seed 42
   ```

   每个 CV 根目录生成 `manifest.csv`，各折目录保存 `block_splits.csv` 和 `normalization.json`。Graph Laplacian 通过 `preprocessing.graph_laplacian` 配置，电极文件的通道顺序需要与 EEG 一致。

3. 提供训练配置。

   将配置保存为自己的 YAML 文件，并通过 `--config` 指定。相对路径以该 YAML 所在目录为基准。以下示例以配置文件位于仓库根目录为例，使用已生成的 block-CV 资料训练 CNN。

   ```yaml
   seed: 42
   cv_fold: 0
   cv_folds: 5
   cv_split_dir: ./artifacts/cv5

   data:
     manifest_csv: ./artifacts/cv5/manifest.csv
     split_csv: ./artifacts/cv5/fold_0/block_splits.csv
     normalization_json: ./artifacts/cv5/fold_0/normalization.json
     experiments: [1]
     lags: [1]
     cv_strategy: block
     normalization_mode: per_subject
     balance_train_cells: false
     balance_eval_cells: false
     num_workers: 0

   model:
     encoder:
       type: cnn
       in_channels: auto
       latent_dim: 32
       num_blocks: 2
       kernel_size: 5
       dropout: 0.1
     attention_mode: gated_scorer
     attention_gate_activation: tanh_plus_one
     time_no_class_enabled: true
     classifier_num_layers: 1

   train:
     iterations: 10000
     validation_interval_iterations: 100
     log_interval_iterations: 100
     batch_size: 64
     optimizer: adamw
     learning_rate: 0.0001
     lr_schedule: constant
     weight_decay: 0.01
     loss_type: ce
     time_loss_weight: 0.5
     pooled_loss_weight: 1.0
     attention_topk_ratio: 0.2
     checkpoint_metric: val_pooled_accuracy
     checkpoint_mode: max
     amp: true

   plots:
     enabled: true
     splits: [val]
     gaussian_baseline:
       enabled: false

   run:
     base_output_dir: ./runs
     base_experiment_name: choice-cnn

   swanlab:
     enabled: false
   ```

   `data` 定义样本选择、划分和归一化，`model` 定义模型结构，`train` 定义优化与监督，`plots` 定义图表，`run` 定义运行目录。

   使用 subject-CV 时，将 CV 路径指向相应资料目录，设置 `data.cv_strategy: subject` 和 `data.normalization_mode: train_global`。启用被试条件化时，设置 `model.subject_conditioning.enabled: true`，并使用 `mode: shared_plus_residual`。

   局部 Transformer 使用 `model.encoder.type: window_transformer`，通过 `window_size`、`num_heads`、`num_layers` 和 `attention_pattern` 设置窗口、注意力头数、层数和方向。`centered_local` 配合 `boundary_mode: noise_pad` 启用高斯 K/V 补边，`boundary_noise_seed` 控制验证时的稳定采样。

4. 训练和恢复。

   ```bash
   python -m memory_reactivation train \
     --config /path/to/train.yaml --task choice --cv-fold 0
   ```

   `--task` 可选择 `choice`、`pitch` 或 `motor`，`--cv-fold` 从 0 开始指定一折，`--lag` 可覆盖配置中的 lag。`swanlab.enabled` 控制训练日志。

   恢复训练使用该运行的配置与 checkpoint。

   ```bash
   python -m memory_reactivation train \
     --config /path/to/train.yaml --task choice --cv-fold 0 \
     --resume-from /path/to/run/last.ckpt
   ```

   checkpoint 保存模型、优化器、AMP scaler、随机数状态、数据游标、最佳分数和早停状态。恢复时核验代码实现、配置和数据身份。延长 `iterations` 时沿用已保存的学习率日程。

   | 运行文件 | 内容 |
   | --- | --- |
   | `config_snapshot.yaml` | 解析后的训练配置 |
   | `code_snapshot/` | 完整源码快照 |
   | `artifacts/` | 数据划分、归一化和预处理身份 |
   | `train_examples.csv` | 训练样本清单 |
   | `metrics.csv`、`train_metrics.csv` | 验证与训练指标 |
   | `best_val.ckpt` | 按指定验证指标选出的最佳权重 |
   | `last.ckpt` | 最近保存的训练状态 |

5. 载入模型和预测。

   ```bash
   python -m memory_reactivation inspect /path/to/run/best_val.ckpt
   ```

   ```python
   from memory_reactivation.artifacts.checkpoints import load_model, save_model

   loaded = load_model("/path/to/run/best_val.ckpt", device="cpu")
   model = loaded.model
   print(loaded.task, loaded.spec)
   save_model("/path/to/model.ckpt", model, config=loaded.config)
   ```

   模型加载会核验参数和构造配置，恢复训练时的类别映射、时间长度与被试条件化信息。原生 spatial baseline 使用同目录中的 `identity.json` 和 `resolved_model.json`。

   预测输入 NPZ 按 checkpoint 对应的空间处理与归一化协议准备。

   | 字段 | 形状与用途 |
   | --- | --- |
   | `eeg` | float32 `[B,C,T]`，预处理后的 EEG |
   | `time_s` | `[T]`，Dataset 提供的共享时间轴，供已保存的注意力时间窗使用 |
   | `subject_id` | `[B]`，被试条件模型使用的全局整数编号 |
   | `exp`、`trial0` | `[B]`，实验编号与原始 trial 编号 |
   | `record_uid`、`record_id`、`lag`、`target` | 可选的逐样本身份与标签 |

   `noise_pad` 模型需要 `exp`、`subject_id`、`trial0` 三个非负整数数组，以便按物理 trial 身份生成稳定补边。

   ```bash
   python -m memory_reactivation predict /path/to/run/best_val.ckpt \
     --input /path/to/prepared_eeg.npz \
     --output /path/to/predictions.npz \
     --batch-size 64 --device cpu
   ```

   输出包含 logits、注意力和预测元数据。Python 流式预测接口为 `memory_reactivation.evaluation.predict.iter_predictions`。

6. 评估、导出和可视化。

   重新计算验证集并生成图表。

   ```bash
   python -m memory_reactivation reexport --run-dir /path/to/run --splits val
   ```

   一次导出原始输入、Gaussian、时间打乱和时间平移四种条件。

   ```bash
   python -m memory_reactivation reexport --run-dir /path/to/run --splits val \
     --input-kinds real gaussian time_shuffle time_roll \
     --time-shuffle-seed 42 --time-roll-shift-s 0.5 \
     --controls-output-dir /path/to/control-results
   ```

   `--reuse-existing-kinds real` 可复用经过权重、模型行为、实现、数据和完整性核验的原始输入导出。

   已有 CSV 通过 `report` 生成图表。

   ```bash
   python -m memory_reactivation report --run-dir /path/to/run --splits val \
     --output-dir /path/to/figures
   ```

   CSV 包括 trial 与被试指标、时间准确率、注意力权重、时间激活、混淆矩阵和条件单元统计。图表支持注意力与门控曲线、有效注意力总质量、时间准确率、激活、分类 margin，以及按被试和当前刺激分组的结果。

7. 运行测试。

   ```bash
   CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider
   ```

   普通测试使用合成 EEG、电极数据和临时配置。GPU 测试按 CUDA 可用性运行。历史模型与训练轨迹对照工具的参数见 [USAGE.md](USAGE.md)。

详细操作见 [USAGE.md](USAGE.md)。许可采用 [Apache-2.0](LICENSE)。
