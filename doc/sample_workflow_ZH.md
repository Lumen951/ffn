本文档给出一个使用 FFN 的示例工作流。

> 说明：本文是 `doc/sample_workflow.md` 的中文译本。代码块、参数名、路径与 URL 均保持原文不变。

# 数据
* 实验室拥有某个生物体的电镜数据，表示为图像栈（一组 png 或 tif）或稠密数据格式（例如 hdf5 立方体）。数据可能是各向同性的，也可能不是，并且对整个数据集的跨度有一套约定好的（全局）坐标。为执行各种操作，可以从较大的体数据中取出较小的盒子。我们把这份数据集称为"原始"（raw）数据。注意：即使已经生成了用于人类可视化的数据替代视图，本流水线通常仍应使用未经修改的原始值。你会使用原始数据的一个子集进行训练（为取得好结果，目标是一个边长 600 的立方体）。
* 实验室已产出分割结果——这些稠密数据集意在对应较大数据集中特定的盒子，但其中每个体素的值不是图像数据，而是该体素被认为所属的分割片段。一个分割片段理解为分割数据集中共享同一取值的坐标集合（与某个片段对应的具体数值本身没有特别含义，但在工作流的各部分之间保持一致会让讨论特定片段更容易）。

# 使用 FFN 的目标
实验室希望借助机器学习和推理产出新的分割数据集。实验室拥有适合此用途的 GPU 加速硬件。

# 软件前提
* 并非必需，但你可以用 "pip install ." 安装 FFN，以便其他软件把它作为库使用
* Conda 是为 python 环境获取所需组件的合理方式

# 环境搭建与流程
* 分割数据集所覆盖的那部分全局坐标系区域，应准备成两个 hdf5 体数据：一个装原始数据，一个装分割数据。分割数据在体数据中应为 int64 数据集。原始数据通常是 uint8 数据集。
* 按 readme 所述运行 compute_partitions.py 和 build_coordinates；它们把真值转换成 tensorflow record 文件（用于启动网络进行训练）
* 运行 train.py，把该 record 文件传给 train_coords，同时传入原始图像数据和标签体数据。其他参数应无需调整。这段代码会持续输出带编号的新网络快照；若被中断可以重启。最好让它训练非常长的时间（即使使用高端 GPU，视期望质量而定，让它跑上数周甚至更久也可能是合理的）。多 GPU 训练没有内建支持，但可以用分布式 Tensorflow API 实现：https://www.tensorflow.org/deploy/distributed ，采用异步 SGD 和自定义训练脚本。
* train.py 结束后，你会得到一个装满 model.ckpt-NUMBER.EXTENSION 文件的目录。通常你只关心编号最大的那一组；这就是你训练好的网络。
* 从你的图像栈中取出一个新的、不重叠的数据集（边长 300 的立方体是合理的；边长 600 或更大的立方体要跑一阵子，但有耐心的生物学家可以用），并按与训练所用原始数据集相同的格式保存为 hdf5 文件。这就是你的推理目标。
* 写一个 pbtxt 作为推理配置。具体细节见本文档末尾
* 运行 run_inference.py，传入你刚生成的 protobuf-text 配置（同 readme 中那样），以及一个边界框规格（把 start 部分都留为零，size 设为推理目标在每个维度上的尺寸）。结果会输出到你指定的目录中。这一步可能要跑一阵子
* 结果完成后，你会想把它转换成便于用其他软件修改的格式（hdf5 或图像栈）。下面提供了一段代码示例。

# 导出 FFN 结果
``` python
sys.path.append('/path/to/ffn')
from ffn.inference import storage
import h5py

seg, _ = storage.load_segmentation('/path/to/ffn/resultsdir', (0, 0, 0))
dest = h5py.File('/target/h5pyfile.h5', 'w')
dest.create_dataset('seg', data=seg, dtype='int64')
```

# 编写推理配置
从 configs/inference_training_sample2.pbtxt 开始，复制一份再按需修改。

（源码中的）这个 protobuf 文件记录了该文件中大部分参数：

ffn/inference/inference.proto

至少需要修改这些字段：
* image —— 指定你希望在其上运行推理的图像体数据
* segmentation_output_dir —— 为本次运行的快照和结果指定一个（最好唯一的）目录

你很可能会想调整推理选项中的 pad_value 和 move_threshold 来微调结果。
