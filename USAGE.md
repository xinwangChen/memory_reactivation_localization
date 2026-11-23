# PB2023 EEG 使用说明

在仓库根目录创建 Python 3.10 及以上版本的环境，安装包与测试依赖。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

命令支持 `prepare`、`train`、`inspect`、`predict`、`reexport` 和 `report`。各子命令提供 `--help`。

1. 准备数据与交叉验证资料。

   已清洗数据使用 `data/preprocessed/Exp1/Sub01_eeg_clean.npz` 和同目录的 `Sub01_meta.csv` 组织。NPZ 保存 EEG、trial 身份、标签和时间轴，元数据 CSV 保存完整行为历史与 block 信息。

   `prepare` 可通过 `--run-preprocessor --preprocess-script /path/to/preprocess.py` 调用外部原始数据转换脚本。下面的命令直接使用预处理数据。

   ```bash
   pb-eeg prepare --preprocessed-root data/preprocessed \
     --artifact-dir artifacts --cv-folds 5 --cv-strategy block \
     --cv-output-dir artifacts/cv5 --seed 42

   pb-eeg prepare --preprocessed-root data/preprocessed \
     --artifact-dir artifacts --cv-folds 5 --cv-strategy subject \
     --normalization-mode train_global \
     --cv-output-dir artifacts/cv5-subject-seed42 --seed 42
   ```

   每个 CV 目录包含 manifest 和各折的 split、normalization 资料。subject-CV 使用仅由训练被试估计的全局归一化。启用 Graph Laplacian 时，在配置中指定 `preprocessing.graph_laplacian.electrodes_npz`。配方默认读取 `data/preprocessed/electrodes/Exp1_electrodes.npz`。

2. 选择配方训练一折。

   `configs/{choice,pitch}-{block,subject}.yaml` 提供四份基础主线配方。`configs/noise-pad/` 提供对应的噪声补边配方。`target_pitch_cnn_launch.yaml` 与 `target_pitch_transformer_launch.yaml` 提供编码器对照配方。

   YAML 中的相对路径以配置文件所在目录为基准。`cv_split_dir` 指向 CV 根目录，`data` 指定 manifest、split、normalization 和缓存，`run.base_output_dir` 指定输出位置。根据实际数据目录编辑这些路径。模型和训练参数在各配方中显式记录。

   ```bash
   PB2023_SWANLAB_ENABLED=0 pb-eeg train \
     --config configs/choice-block.yaml --task choice --cv-fold 0

   PB2023_SWANLAB_ENABLED=0 pb-eeg train \
     --config configs/noise-pad/pitch-subject.yaml --task pitch --cv-fold 0
   ```

   `--lag` 可以覆盖配方的 lag，lag 必须为非负整数。Target 使用固定时间轴，动态窗口候选支持一个有效长度。训练输出保存完整源码快照、解析后的配置、数据产物身份、样本清单、指标 CSV、`best_val.ckpt` 和 `last.ckpt`。

   同一实现下可恢复优化器、AMP scaler、RNG、数据游标、最佳分数和早停状态。

   ```bash
   PB2023_SWANLAB_ENABLED=0 pb-eeg train \
     --config configs/choice-block.yaml --task choice --cv-fold 0 \
     --resume-from /path/to/run/last.ckpt
   ```

   恢复时核验实现、模型、训练设置和数据身份。延长 `iterations` 会沿用已保存的学习率日程。从旧断点恢复时，最佳权重必须符合该断点的步数和记录分数。缺失相应历史最佳权重时，程序在截断指标日志前报错。

3. 检查、加载与保存模型。

   ```bash
   pb-eeg inspect /path/to/run/best_val.ckpt
   ```

   ```python
   from pb2023_eeg.artifacts.checkpoints import load_model, save_model

   loaded = load_model("/path/to/run/best_val.ckpt", device="cpu")
   model = loaded.model
   print(loaded.task, loaded.spec)
   save_model("/path/to/model.ckpt", model, config=loaded.config)
   ```

   Target 加载会核验 ModelSpec、配置与参数，恢复已保存的 embedding 容量和训练被试 ID。历史 checkpoint 使用内部配置或相邻配置快照恢复构造信息。包含旧序列化对象的可信 checkpoint 可通过 `trusted_legacy=True` 读取。

   原生 spatial baseline 使用相邻的 `identity.json` 和 `resolved_model.json`，预测任务与其训练任务绑定。

4. 使用准备好的 EEG 预测。

   输入 NPZ 的 `eeg` 为 float32 数组，形状为 `[B,C,T]`，按目标 checkpoint 对应的空间处理和归一化协议准备。条件模型需要训练时的全局 `subject_id: [B]`。可以保存 `record_uid`、`exp`、`trial0`、`lag` 和 `target` 等逐样本身份。

   checkpoint 配置了注意力时间窗时，输入提供来自 Dataset 的共享 `time_s: [T]`。时间轴必须有限、严格递增，并与已保存的时间轴指纹匹配。

   ```bash
   pb-eeg predict /path/to/run/best_val.ckpt \
     --input /path/to/prepared_eeg.npz \
     --output /path/to/predictions.npz --batch-size 64 --device cpu
   ```

   输出包括原生 logits、注意力和元数据。Python 流式接口为 `evaluation.predict.iter_predictions`，分数类型由 PredictionSpec 指定。

   `noise_pad` 用于 centered-local Transformer，在标准化 EEG 空间为边界补充高斯 K/V。训练按 RNG 采样，真实/null 对照复用补边。验证按 `(exp, subject_id, trial0)` 固定采样，同一 trial 的原始、Gaussian、shuffle 和 roll 条件使用一致补边。输入 NPZ 因此需要这三个非负整数数组。直接调用 eval 模型时传入 `boundary_trial_ids` 或显式的 `boundary_noise_eeg`。

5. 重导出评估与绘图。

   ```bash
   pb-eeg reexport --run-dir /path/to/run --splits val

   pb-eeg reexport --run-dir /path/to/run --splits val \
     --input-kinds real gaussian time_shuffle time_roll \
     --controls-output-dir /path/to/control-results

   pb-eeg report --run-dir /path/to/run --splits val
   ```

   `reexport` 载入权重、核验配置和数据后重新推理，导出 CSV 并生成图表。`report` 从已有 CSV 绘图。使用 `--reuse-existing-kinds real` 可复用通过权重、模型行为、实现、数据、参数和完整性核验的训练导出。

   重导出默认把 CSV 发布到运行目录，`--output-dir` 指定图目录。`--controls-output-dir` 与显式输入条件组合使用，把对照 CSV 和图表写入指定目录。

   CSV 包括 trial 与被试指标、注意力、时间准确率、时间激活、混淆矩阵和条件单元统计。绘图代码位于 `visualization/layouts.py`，由 `workflows/report.py` 组织。

6. 运行测试与历史对照。

   普通测试使用临时合成 EEG、电极数据和仓库内配方。

   ```bash
   CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider
   ```

   历史对照使用外部模型、冻结源码和参考输入。参考目录包含 `reference_validation.json` 与相应的 `*_forward.npz`。workspace 中按原资料布局存放 `main-config-models/` 和 baseline 资料。

   ```bash
   python scripts/validate_legacy_models.py \
     --workspace /path/to/historical-workspace \
     --reference-dir /path/to/reference-inputs \
     --out /path/to/check-results

   python scripts/validate_training_parity.py \
     --source /path/to/frozen/code_snapshot \
     --out /path/to/parity-results
   ```

   模型对照检查严格加载与逐张量 CPU 前向一致性。训练对照检查 CNN、局部 Transformer 的参数、优化器、AMP、RNG、指标和最佳权重选择。运行产物通过 Git 忽略规则管理。
