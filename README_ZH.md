# Flood-Filling Networks

Flood-Filling Networks（FFN，泛洪填充网络）是一类神经网络，专为复杂、大型形状的实例分割而设计，尤其适用于脑组织的体积电镜（volume EM）数据集。

更多细节请参见相关论文：

 * https://arxiv.org/abs/1611.00421
 * https://doi.org/10.1101/200675

本项目并非 Google 的官方产品。

> 说明：本文是 `README.md` 的中文译本。所有代码块、命令行参数、路径与 URL 均保持原文不变，以便直接复制使用。

# 安装

无需安装。依赖声明在 `pyproject.toml` 中，版本锁定在 `uv.lock` 中。安装依赖请运行：

```shell
  uv sync
```

可选依赖组以 extra 形式提供：`--extra jax`（JAX/Flax 训练代码）、`--extra proofreading`、`--extra interactive`、`--extra dev`。

代码已在配备 Tesla P100 GPU 的 Ubuntu 16.04.3 LTS 系统上测试通过。

# 训练

FFN 网络可通过 `train.py` 脚本训练，该脚本需要一个 TFRecord 文件，其中存放了从输入体数据中采样数据所用的坐标。

## 准备训练数据

有两个脚本可为以 HDF5 文件形式存储的带标注数据集生成训练坐标文件：`compute_partitions.py` 和 `build_coordinates.py`。

`compute_partitions.py` 把标签体数据转换为一个中间体数据：其中每个体素 `A` 的值，等于在以 `A` 为中心、半径为 `lom_radius` 的子体积内，与 `A` 标注相同的体素所占比例的量化值。`lom_radius` 通常应设为 `(fov_size // 2) + deltas`（其中 `fov_size` 和 `deltas` 是 FFN 模型的设置）。每一个这样的量化比例称为一个 *partition*。调用示例：

```shell
  python compute_partitions.py \
    --input_volume third_party/neuroproof_examples/validation_sample/groundtruth.h5:stack \
    --output_volume third_party/neuroproof_examples/validation_sample/af.h5:af \
    --thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
    --lom_radius 24,24,24 \
    --min_size 10000
```

`build_coordinates.py` 使用上一步得到的 partition 体数据，生成一个坐标 TFRecord 文件，其中每个 partition 被采样的频率大致相等。调用示例：

```shell
  python build_coordinates.py \
     --partition_volumes validation1:third_party/neuroproof_examples/validation_sample/af.h5:af \
     --coordinate_output third_party/neuroproof_examples/validation_sample/tf_record_file \
     --margin 24,24,24
```

## 示例数据

我们在 `third_party` 中提供了 FIB-25 `validation1` 体积的示例坐标文件。由于该文件较大，它托管在 Google Cloud Storage 上。如果你此前未使用过，需要先安装 Google Cloud SDK，并执行以下配置：

```shell
  gcloud auth application-default login
```

你还需要在本地创建标签与图像的副本：

```shell
  gcloud storage rsync --recursive --exclude ".*.gz" gs://ffn-flyem-fib25/ third_party/neuroproof_examples
```

## 运行训练

坐标文件准备好之后，就可以用以下命令开始训练 FFN：

```shell
  python train.py \
    --train_coords gs://ffn-flyem-fib25/validation_sample/fib_flyem_validation1_label_lom24_24_24_part14_wbbox_coords-*-of-00025.gz \
    --data_volumes validation1:third_party/neuroproof_examples/validation_sample/grayscale_maps.h5:raw \
    --label_volumes validation1:third_party/neuroproof_examples/validation_sample/groundtruth.h5:stack \
    --model_name convstack_3d.ConvStack3DFFNModel \
    --model_args "{\"depth\": 12, \"fov_size\": [33, 33, 33], \"deltas\": [8, 8, 8]}" \
    --image_mean 128 \
    --image_stddev 33
```

注意，使用所提供的模型进行训练和推理都是计算开销很大的过程。我们建议使用配备 GPU 的机器以获得最佳效果，尤其是在 Jupyter notebook 中交互式地使用 FFN 时。按上述配置训练 FFN 需要 12 GB 显存的 GPU。你可以通过减小 batch size、模型深度、`fov_size` 或卷积层特征数来降低内存占用。

训练脚本未针对多 GPU 或分布式训练进行配置。相关设置方法请参见 [Distributed TensorFlow](https://www.tensorflow.org/deploy/distributed#replicated_training) 文档。

# 推理

我们提供两个使用已训练 FFN 模型运行推理的示例。对于非交互式场景，可以使用 `run_inference.py` 脚本：

```shell
  python run_inference.py \
    --inference_request="$(cat configs/inference_training_sample2.pbtxt)" \
    --bounding_box 'start { x:0 y:0 z:0 } size { x:250 y:250 z:250 }'
```

该命令会分割 `training_sample2` 体积，并把结果保存到 `results/fib25/training2` 目录。会生成两个文件：`seg-0_0_0.npz` 和 `seg-0_0_0.prob`。两者均为 `npz` 格式，分别包含分割图和量化后的概率图。在 Python 中可以这样加载分割结果：

```python
  from ffn.inference import storage
  seg, _ = storage.load_segmentation('results/fib25/training2', (0, 0, 0))
```

我们在 `results/fib25/sample-training2.npz` 中提供了示例分割结果。对于 training2 体积，使用 P100 GPU 进行分割耗时约 7 分钟。

对于交互式场景，请查看
[`ffn_inference_colab_demo.ipynb`](https://colab.sandbox.google.com/github/google/ffn/blob/master/notebooks/ffn_inference_colab_demo.ipynb)。
这个 Colab notebook 演示了如何以显式指定的种子分割单个物体，并在推理运行过程中可视化结果。

两个示例都配置为使用一个 3d convstack FFN 模型，该模型在 FlyEM 项目（Janelia）FIB-25 数据集的 `validation1` 体积上训练得到。

# 更多信息

请参见 `doc/manual.md`。
