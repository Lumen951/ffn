# FFN 代码仓库阅读指南

本文给出阅读本仓库的推荐顺序。这个顺序不是按目录字母序，也不是按文件大小，而是**按"先建立心智模型、再沿一条调用链打通"的方式**组织的。

---

## 一、先建立心智模型

这个仓库是 Google 的 **Flood-Filling Networks**（FFN），用于体积电镜（volume EM）脑组织的**神经元实例分割**。相关论文：

- <https://arxiv.org/abs/1611.00421>
- <https://doi.org/10.1101/200675>
- <https://doi.org/10.1038/s41592-018-0049-4>（`doc/manual.md` 复现的目标流水线）

### 核心反直觉点（决定了后面所有代码怎么读）

**这个网络不是"图像 → 分割图"。** 它预测的是：

```
输入：图像的一个 FoV 块 + 一个种子体素（单点）
输出：仅这一个物体的 mask（object mask prediction）
```

实现见 [ffn/training/models/convstack_3d.py:29](../../ffn/training/models/convstack_3d.py#L29) 的 `_predict_object_mask()`。

推理时把 mask 中的高置信区域作为新种子，让 FoV **一步步向前推进**（flood-fill），从一个点"长"出整个神经元。这个"推着走"的过程就是 inference 的全部内容。这也解释了为什么 `deltas`（每步移动多少体素）和 `fov_size` 是模型的两个核心参数——参见 [configs/inference_training_sample2.pbtxt:8](../../configs/inference_training_sample2.pbtxt#L8)。

### 第二个关键点：训练数据是坐标，不是图像块

训练集不是"图像块集合"，而是**坐标文件（TFRecord of coordinates）**：

1. [compute_partitions.py](../../compute_partitions.py) 把标签体数据按"邻域内同标号占比"量化成 partition；
2. [build_coordinates.py](../../build_coordinates.py) 再按 partition 均衡采样出坐标，保证每个 partition 出现频率大致相等。

这是 FFN 训练能否收敛的关键。读训练部分之前必须先看懂这两步，否则无法理解 `train.py` 为什么要接一个坐标文件。

---

## 二、阅读顺序

### 第 0 层：文档（不看代码）

| 文件 | 作用 |
| --- | --- |
| [README.md](../../README.md) | 项目定位、安装、训练/推理的示例命令 |
| [doc/manual.md](../manual.md) | **最有价值的文件**：按"训练 → 分割推理 → 共识 → 再分割 agglomeration"组织，基本与模块一一对应，相当于一份带解释的目录 |
| [doc/sample_workflow.md](../sample_workflow.md) | 端到端的工作流视角 |

### 第 1 层：数据面

- [ffn/input/volume.py](../../ffn/input/volume.py) —— HDF5/npz 体数据怎么读、怎么缓存
- [ffn/utils/bounding_box.py](../../ffn/utils/bounding_box.py) —— 全仓库通用的坐标抽象

> **容易卡住的坑**
>
> 你会到处看到 `from connectomics.common import bounding_box`。`connectomics` **不在这个仓库里**，它是 [pyproject.toml](../../pyproject.toml) 声明的外部 pip 包（Google 的 connectomics 库）。`bounding_box`、`segmentation.labels`、`jax.training` 都来自那里。不要在本地仓库里 grep 半天找不到。

### 第 2 层：模型本体（最短，但最重要）

1. [ffn/training/model.py](../../ffn/training/model.py) —— 看 `ModelInfo`，理解 fov / deltas 的几何含义
2. [ffn/training/models/convstack_3d.py](../../ffn/training/models/convstack_3d.py) —— **只有 102 行，是整个网络的实现**

**先读它，不要先碰 inference。** 先把"种子 → mask"这一步搞清楚。

> `model.py` 使用的是 `tensorflow.compat.v1` + graph mode，与现代 TF2 的写法不同。不要被 `tf.Session` 之类的写法绕进去。

### 第 3 层：训练侧

1. [ffn/training/examples.py](../../ffn/training/examples.py) —— 怎么从"坐标 + 图像 + 标签"造出一个训练样本
2. [ffn/training/inputs.py](../../ffn/training/inputs.py) —— 输入流水线
3. [train.py](../../train.py) —— 主流程
4. [compute_partitions.py](../../compute_partitions.py) / [build_coordinates.py](../../build_coordinates.py) —— 回到数据准备，理解坐标文件的来历

### 第 4 层：推理主干（仓库的重心）

这一层**不要按文件名顺序读，按调用链读**。入口在 [run_inference.py:49-52](../../run_inference.py#L49-L52)：

```
run_inference.py  →  Runner.start() / Runner.run()   (runner.py，编排层)
                  →  Canvas                            (inference.py，状态机，真正的核心)
                  →  ExecutorClient.predict()          (executor.py，模型调用)
```

推荐顺序：

1. **[ffn/inference/inference_flags.py](../../ffn/inference/inference_flags.py)** + [configs/inference_training_sample2.pbtxt](../../configs/inference_training_sample2.pbtxt)（`inference_pb2.py` 为生成代码）
   先搞清一个 `InferenceRequest` 有哪些字段。后面所有参数都有出处。

2. **[ffn/inference/runner.py](../../ffn/inference/runner.py)**
   `start()`(L165) / `make_canvas()`(L307) / `run()`(L484)。看编排层如何把配置、模型、种子策略组装起来。

3. **[ffn/inference/executor.py](../../ffn/inference/executor.py)**
   文件头的 docstring 写得很清楚：client-server 结构，server 独占模型与加速器资源做 batching，client 各自维护分割状态。关注 `ExecutorClient`(L85) / `ThreadingBatchExecutor`(L207) / `JAXExecutor`(L343)。

4. **[ffn/inference/inference.py](../../ffn/inference/inference.py)** —— **`Canvas` 类(L129) 是全部核心**
   关键方法：`reset_state`(L291)、`predict`(L356)、`update_at`(L386)、`segment_at`(L460)、`segment_all`(L538)。
   flood-fill 的"推进"逻辑就在这里。

5. **[ffn/inference/movement.py](../../ffn/inference/movement.py)**
   FoV 往哪走。`FaceMaxMovementPolicy`(L97) 是默认策略，`MovementRestrictor`(L178) 管边界约束。

6. **[ffn/inference/seed.py](../../ffn/inference/seed.py)**
   种子从哪来。`PolicyPeaks`(L142) 是推荐默认，`PolicyInvertOrigins`(L455) 专门用于正反共识。

7. **[ffn/inference/align.py](../../ffn/inference/align.py)**
   只有 172 行，是子体积的 ad-hoc 局部对齐（基类默认是 identity / no-op）。
   **放在 Canvas 之后读**：Canvas 持有 alignment 对象，先了解调用方，才知道它对齐的是"哪个坐标系到哪个坐标系"、在补偿什么。

8. **[ffn/inference/storage.py](../../ffn/inference/storage.py)**
   npz / prob 的落盘与加载。

### 第 5 层：下游后处理

1. [ffn/inference/consensus.py](../../ffn/inference/consensus.py) —— 正反种子顺序共识
2. [ffn/utils/decision_point.py](../../ffn/utils/decision_point.py) —— 找决策点
3. [ffn/inference/resegmentation.py](../../ffn/inference/resegmentation.py) —— 再分割
4. [ffn/inference/resegmentation_analysis.py](../../ffn/inference/resegmentation_analysis.py) —— 打分；阈值建议见 [doc/manual.md:194-216](../manual.md#L194-L216)
5. [ffn/utils/proofreading.py](../../ffn/utils/proofreading.py) —— 可最后看

### 第 6 层：JAX 并行实现（与 TF 版是两套）

- [ffn/jax/](../../ffn/jax/) —— Flax/JAX 重写的**训练**（`train.py` 764 行 + `input_pipeline.py` 489 行），不含推理
- [ffn/secgan/](../../ffn/secgan/) —— SECGAN，与 FFN 无关的另一个模型，`train.py` 1629 行是全仓库最大的文件

这两块的版权年份是 2024/2026，提交历史里也是最近加入的（`2564541 Open source the JAX SECGAN code`），属于新增部分而非 FFN 主线。**建议放到最后**，否则会被大量样板代码淹没。

---

## 三、两条实操建议

1. **不要顺序通读。**
   先在第 2 层建立"种子 → mask → 推进"的图像，然后只沿着 `run_inference.py` 那一条调用链（第 4 层）打通一遍。打通之后，`seed.py` / `movement.py` / `align.py` 都只是这条链上的**可替换策略**，读起来会快很多。

2. **这个仓库比公开的 google/ffn 新。**
   提交历史里有 2024–2026 的 commit（`JAXExecutor`、`checkpoint_every_minutes`、`threshold_abs` 等）。如果对照 GitHub 上的公开版本读，可能会发现对不上——**以本地为准**。

---

## 附录：模块速查

| 模块 | 职责 |
| --- | --- |
| `ffn/input/` | 体数据读取（HDF5 / npz） |
| `ffn/training/` | 模型定义、训练输入流水线、数据增强、优化器 |
| `ffn/inference/` | 推理主干：Canvas、执行器、种子/移动策略、存储、共识、再分割 |
| `ffn/utils/` | 几何工具、边界框、决策点、proofreading |
| `ffn/jax/` | FFN 训练的 JAX/Flax 实现 |
| `ffn/secgan/` | SECGAN 模型（与 FFN 主线无关） |
| `build_coordinates.py` / `compute_partitions.py` | 训练坐标文件的生成 |
| `train.py` / `run_inference.py` | 训练与推理的命令行入口 |
