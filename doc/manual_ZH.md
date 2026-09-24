这份文档说明如何使用 FFN 仓库中的代码，构建一条与 [FFN 论文](https://doi.org/10.1038/s41592-018-0049-4) 所述相似的神经元重建流水线。推荐设置已更新，以反映当前的最佳实践。

> 说明：本文是 `doc/manual.md` 的中文译本。代码块、参数名、路径与 URL 均保持原文不变。

# 模型训练

为获得最佳的模型质量，推荐使用朴素的异步 SGD，`learning_rate = 0.001`。批大小通常由可用内存决定，往往相当小（2–4）。使用多达 32 块 GPU 训练通常能带来加速。超过这个规模后，尽管按优化步数/秒衡量的处理速度看似更快，模型收敛反而会变差。

## 数据预处理

务必确保训练集与推理集使用完全相同的预处理。我们推荐对所有图像应用 CLAHE，以减小切片之间与区域之间的对比度差异。我们也强调精确数据对齐的重要性。通常需要弹性对齐才能达到足够精度，某些用 FIB-SEM 采集的数据集可能是例外。当沿 'z' 维滚动时视觉上平滑、没有抖动、跳变或漂移，就可以认为数据对齐良好。

## 标注要求

为获得最佳效果，推荐使用至少 150 Mvx 的数据集专属真值标注。如果拿不到、或不易采集，通常可以采用自举（bootstrapping）方法，它能减少获取必要标注所需的总工作量。做法是：先用现有标注训练一个初始 FFN 模型，用它生成一版草稿分割；然后对分割结果中的神经元进行人工校对，并加入真值集；收集到足够数据后重新训练 FFN 模型。视需要可反复迭代，并且借助过分割共识（oversegmentation consensus），校对通常可以限制为对神经元碎片进行人工聚集（agglomeration）。如果初始人工标注不够像素级精确，也推荐采用自举。为获得最佳效果，分割掩码应覆盖整个突起，即细胞质、细胞器和细胞膜。

## 收敛与检查点选择

一个经验法则是：训练到模型见过的 FOV 数量与真值集中标注体素的数量相当。见过的 FOV 数可按 `<batch_size> * <step_number>` 计算。例如，`batch_size=4` 且有 150 Mvx 训练数据时，至少应训练 37.5M 步。

训练过程中，模型快照（检查点）会按预设频率保存。训练结束后，应使用一个独立的验证数据集进行**检查点选择**。为此，我们推荐用尽可能多的已保存检查点对该数据集运行推理，从最新的开始；然后用你关心的指标评估得到的分割结果，并把最佳检查点用于大规模推理。分割评估代码目前不属于 FFN 仓库。我们推荐使用骨架（skeleton）指标进行评估，以确保所选检查点是为拓扑正确性优化的。

# 分割推理

分割推理通过 `InferenceRequest` 协议缓冲区消息配置。以下选项是目前针对 FFN 模型推荐的：

```
 image_stddev: 33.0
 image_mean: 128.0
 seed_policy: "PolicyPeaks"
 inference_options {
   init_activation: 0.95
   pad_value: 0.5
   move_threshold: 0.6
   min_boundary_dist {x:2 y:2 z:1}
   segment_threshold: 0.6
   min_segment_size: 1000
 }
```

`pad_value` 与 `move_threshold` 应与训练时使用的 `--seed_pad` 和 `--threshold` 一致。`image_stddev` 和 `image_mean` 在实践中影响不大；只要 EM 图像是用 CLAHE 以及匹配的设置（训练时的 `--image_mean`、`--image_stddev`）归一化的，就可以保持推荐的默认值。对于各向同性体素的数据集，`min_boundary_dist` 可以调整为 `{x:2 y:2 z:2}`；也可以把各值降到 1 以提高填充率（fill rate），代价是计算开销增加。

## 性能考量

在使用 GPU 等加速器进行推理时，可能需要在单个 worker 上处理多个子体积才能充分利用设备。并发执行的推理调用数量由 `InferenceRequest` 的 `batch_size` 字段控制：启动单个 `inference.Runner`，并从多个线程调用 `runner.run()`（典型配置是线程与子体积 1:1）。为进一步隐藏主机端开销，同一个 worker 可以同时处理多于 `batch_size` 个子体积。

想进一步提升推理速度，可以使用在 `tf.ConfigProto` 中设置了 `graph_options.rewrite_options.auto_mixed_precision=1` 的 `tf.Session`。这会把图中选定的算子自动转换为以 float16 运算。该转换可带来 2 倍以上加速，代价是合并错误率略有上升。精度下降的幅度可能与数据集相关，因此建议在大规模部署前先在验证数据集上实验。

## 分布式处理与组装

本仓库包含以逐子体积为单位生成分割所需的函数（"子体积"指数据集中的一个区域，通常是边长几百体素），结果以 npz 文件保存。这些操作是完全可并行的（embarrassingly parallel），同一步骤产生的各子体积之间没有依赖。本仓库不包含工作负载分发的支持，因为我们预期这高度依赖使用者所用的计算环境。我们推荐采用带分布式 worker 的简单任务队列系统作为处理模型。

子体积结果就绪后，仍需从中组装出全局分割，把各子体积各自的局部 ID 空间协调为单一的全局 ID 空间，并以 HDF5 等格式保存。该协调过程需要维护一个并查集（union-find）数据结构，在分布式系统中可能也需要如此。此功能目前在**本仓库中未实现**。FFN 论文使用了 [Rastogi 等人](https://arxiv.org/abs/1203.5387) 所述算法的自定义实现来建立全局 ID 空间，并用 [TensorStore](https://github.com/google/tensorstore) 的前身来存储协调后的数据。

## 共识

过分割共识可用于在超体素（supervoxel）层面降低假合并率，通过 `ffn/inference/consensus.py:compute_consensus()` 运行，并由 `ConsensusRequest` 协议缓冲区消息配置。我们推荐把 `split_min_size` 设为 1000。

原则上可以在任意两个 FFN 分割之间计算共识（例如用不同模型检查点生成的两份），但最典型的应用是正反种子顺序共识：第一份分割使用 `PolicyPeaks`，第二份使用 `PolicyInverseOrigins`，并把 `segmentation_dir` 指向第一份分割的位置。注意作为共识输入的分割**不需要**经过组装（即共识代码直接处理逐子体积的 .npz 文件）。

# 通过再分割进行聚集

FFN 模型可通过一种称为再分割（resegmentation）的过程来聚集分割片段：从两个不同的种子出发，把分割的某个子集从头重新计算两次。分割被限制在以决策点为中心的小子体积内，决策点通常选为两个原始分割片段最接近的点。然后把再分割结果与原始分割比较，计算出兼容性分数。

与分割推理类似，本仓库提供选择决策点、以及运行并对再分割打分所需的库函数，但不提供分发和管理这些工作的方式。对于后者，我们同样推荐基于任务的队列系统。

## 决策点

决策点是分割中两个物体最接近的位置，可用 `decision_point.find_decision_points()` 函数计算。通常的做法是在已组装并协调好的基础分割的重叠子体积上运行该函数。

## 再分割

再分割可通过调用 `resegmentation.py:process_point()` 执行，并由 `ResegmentationRequest` 协议缓冲区消息配置。在请求中，`inference` 指定 FFN 推理配置，通常应设为与基础分割推理相同的设置。重要的是，`inference.init_segmentation` 应指向计算决策点所用的那个分割体数据。

单个再分割请求可用于许多决策点，这些点的位置及对应的分割片段 ID 对各存于重复的 `points` 字段中。

针对 Z 方向 2 倍各向异性的体数据，典型的再分割专属设置如下：

```
radius { x: 50 y: 50 z: 25 }
analysis_radius { x: 34 y: 34: z: 17 }
max_retry_iters: 8
exclusion_radius: { x: 8 y: 8 z: 4 }
segment_recovery_fraction: 0.6
```

增大 `max_retry_iters` 或 `segment_recovery_fraction`、或减小 `exclusion_radius`，会让再分割代码花费更多精力为给定的片段对寻找潜在连接，代价是计算开销增加。

## 打分

再分割结果可用 `resegmentation_analysis.py:evaluate_pair_resegmentation()` 后处理，它会产生一个带分数的 `PairResegmentationResult` proto。

分数向量到片段对合并概率的映射原则上需要按数据集分别校准。我们观察到以下规则是一个良好的保守（即尽量减少假合并）起点：

 * `eval.iou > 0.8`
 * `eval.from_(a|b)_segmment_(a|b)_consistency > 0.6`
 * `eval.from_a.deleted_voxels / eval.from_a.num_voxels < 0.02` 或
   `eval.from_b.deleted_voxels / eval.from_b.num_voxels < 0.02`

若上述条件全部满足，对应的片段对就可以接受为合并。

直观上，`iou` 衡量两个不同种子所得分割之间的一致性，`from_X.segment_Y_consistency` 衡量从 `X` 出发做再分割时对片段 `Y` 的复现程度，而"删除体素比例" `deleted_voxels / num_voxels` 则是推理时模型混乱程度（即不同时间步体素标注不一致）的度量。
