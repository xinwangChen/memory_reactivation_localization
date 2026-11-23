# Memory Reactivation Localization

PB2023 EEG 解码工具，Python 包名为 `pb2023_eeg`，命令入口为 `pb-eeg`。

Target 主线覆盖数据准备、CNN / Patch MLP / 局部 Transformer 建模、被试条件化、注意力与门控、时间分类头与 pooled 分类头、训练、评估、导出和可视化。任务包括 Choice、Pitch 和 Motor。现有主线配方覆盖 Choice / Pitch 的 block-CV 与 subject-CV，并提供对应的 `noise_pad` 配方。

历史模型兼容层支持 Target 与原生 spatial baseline 的严格加载和预测。baseline / downstream 训练、定位标注、遮罩与跨实验调度列入后续迁移范围。

需要 Python 3.10 及以上版本，依赖由 `pyproject.toml` 声明。

```bash
git clone git@github.com:xinwangChen/memory_reactivation_localization.git
cd memory_reactivation_localization

python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'

pb-eeg --help
```

使用 SwanLab 时可安装日志依赖。

```bash
python -m pip install -e '.[logging]'
```

目录组织如下。

```text
pb2023_eeg/
  models/           编码器、条件化、注意力、分类头和模型工厂
  data/             样本、CV、归一化、空间处理与缓存
  training/         增强、目标函数、单 batch 计算和 trainer
  evaluation/       预测、指标与导出
  analysis/         数值汇总和曲线分析
  visualization/    图形绘制
  artifacts/        模型保存、加载和导出身份校验源码
  workflows/        数据准备、训练、评估、重导出与报告
configs/            实验配方
scripts/            兼容命令入口和历史对照工具
tests/              单元测试与合成数据回归测试
```

数据按实验组织到 `data/preprocessed/`，例如 `Exp1/Sub01_eeg_clean.npz` 与 `Exp1/Sub01_meta.csv`。Graph Laplacian 使用与 EEG 通道顺序匹配的电极文件 `data/preprocessed/electrodes/Exp1_electrodes.npz`。

配方中的相对路径以 YAML 所在目录为基准。默认使用仓库根目录下的 `artifacts/`、`cache/` 和 `runs/` 保存预处理产物、缓存与训练结果。这些目录由运行过程生成，Git 忽略规则覆盖相应文件。

先生成五折划分和训练集归一化统计。

```bash
pb-eeg prepare --preprocessed-root data/preprocessed \
  --artifact-dir artifacts --cv-folds 5 --cv-strategy block \
  --cv-output-dir artifacts/cv5 --seed 42

pb-eeg prepare --preprocessed-root data/preprocessed \
  --artifact-dir artifacts --cv-folds 5 --cv-strategy subject \
  --normalization-mode train_global \
  --cv-output-dir artifacts/cv5-subject-seed42 --seed 42
```

随后选择任务和一折训练。完整命令与数据约定见 [USAGE.md](USAGE.md)。

```bash
PB2023_SWANLAB_ENABLED=0 pb-eeg train \
  --config configs/noise-pad/pitch-subject.yaml --task pitch --cv-fold 0

pb-eeg inspect /path/to/run/best_val.ckpt
pb-eeg reexport --run-dir /path/to/run --splits val
pb-eeg report --run-dir /path/to/run
```

训练默认完整复制源码到运行目录的 `code_snapshot/`，并保存配置、数据身份、训练样本清单、指标、最佳与最近 checkpoint。精确续训核验实现、数据和配置身份。历史权重通过 `load_model` 加载，旧训练过程使用相应运行的冻结源码恢复。

普通回归使用合成数据，可以在新克隆的仓库中执行。GPU 用例根据 CUDA 可用性运行。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider
```

历史权重和训练轨迹的对照工具通过参数读取外部 checkpoint、冻结源码与参考资料。具体参数见 [USAGE.md](USAGE.md)。许可采用 [Apache-2.0](LICENSE)。
