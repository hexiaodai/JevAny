# CPU 分布式训练

JevAny 使用同一套 DDP 训练循环支持 CPU 和 CUDA：CPU 使用 Gloo，CUDA 使用 NCCL。
CPU 路径可用于开发环境中的分布式功能验证，GPU 路径用于后续的 GPU 训练。
各 worker 必须使用相同的模型、训练数据、训练参数和设备类型。

## 单机多进程

安装训练依赖并准备入门数据：

```bash
python -m pip install -e '.[train]'
jevany data init --out data/starter
```

在仓库根目录启动两个 CPU worker：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PROCESSES_PER_HOST=2 \
  bash infra/train.sh --config recipes/sft.toml \
  --device cpu --dtype fp32 --weights-dtype fp32 \
  --max-steps 4 --checkpoint-every-steps 2 --out runs/cpu-ddp
```

也可以直接使用标准入口：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  -m jevany train --config recipes/sft.toml \
  --device cpu --dtype fp32 --weights-dtype fp32 \
  --max-steps 4 --checkpoint-every-steps 2 --out runs/cpu-torchrun
```

`infra/train.sh` 在没有可见 GPU 时默认启动一个进程；有 GPU 时默认按可见 GPU 数量启动。
CPU 训练应显式设置 `PROCESSES_PER_HOST`，进程数不受可见 GPU 数量限制。
设备和精度仍由训练参数决定，使用 CUDA 入门配置时必须覆盖以上三个 CPU 参数。

## 多机训练

各主机使用相同路径挂载训练数据、基础模型和输出目录。主节点地址必须能被所有 worker 访问，
并允许 Gloo worker 之间通信。在 Kubernetes 中，输出目录应使用支持跨节点读写的共享卷。

以下示例使用两台主机，每台一个训练进程。在第一台主机执行：

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.10 MASTER_PORT=29500 \
  PROCESSES_PER_HOST=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  bash infra/train.sh --config recipes/sft.toml \
  --device cpu --dtype fp32 --weights-dtype fp32 \
  --data /mnt/data/train.jsonl --base-load-path /mnt/base \
  --max-steps 4 --checkpoint-every-steps 2 --out /mnt/output/cpu-ddp
```

第二台执行相同命令，将 `NODE_RANK` 改为 `1`，`MASTER_ADDR` 仍指向第一台主机。
每台主机的进程数保持一致。Gloo 自动选择的网卡不正确时，通过 `GLOO_SOCKET_IFNAME` 指定网卡。

Kubeflow Trainer 提供的 `PET_*` 环境变量由 `torchrun` 读取。
Runtime 应使用 `torchrun -m jevany train ...` 启动，并在 TrainJob 中显式设置每节点进程数。
仅增加 Pod 数量并直接执行 `jevany train`，不能组成分布式训练任务。

## 训练与恢复语义

- 每个进程保存完整基础模型、LoRA、决策头和本地训练状态，增加进程数不会降低单进程模型内存需求。
- `--batch` 是每进程的微批大小，名义全局有效批大小为 `batch × accum × WORLD_SIZE`；末尾不足一个累积窗口时按实际记录数归一化。
- 数据按进程切分；不能均分时补齐少量记录，补齐数量记录在训练配置和指标中。
- 训练损失和计数跨进程汇总，评测结果合并；只有全局 rank 0 保存模型、汇总报告并上报 Kubeflow 进度。
- `--resume` 要求保持原来的设备类型、进程总数、训练参数和数据。恢复时所有 worker 使用相同检查点，`--out` 指向新的共享目录。
- 改变设备类型或进程总数时，可使用 `--init-from` 加载兼容权重并开始新训练计划。
- 任一 worker 失败后当前 DDP 任务应整体结束；重新启动并加载检查点的策略由调用方负责。

CPU 首轮验证使用 FP32。`--dtype bf16` 的自动混合精度路径仍只支持 CUDA。
CPU 与 GPU 镜像分别安装对应的 PyTorch 构建，切换 GPU 时按每张可见 GPU 一个进程分配资源。

## 验证范围

单机上的 k3d 多节点可以验证跨 Pod 组网、梯度同步、数据切分、评测、模型保存和续训。
这些节点共享宿主机的 CPU 和内存，不能将各节点上报的容量相加作为物理容量。
CPU 验证不覆盖 CUDA/NCCL、GPU 显存和真实跨物理机的网络性能，投产前需要在目标 GPU 集群验证。
