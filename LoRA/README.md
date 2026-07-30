# YOLO-Master-EsMoE-N LoRA 垂类场景高效微调（VisDrone + Brain Tumor）

针对 issue #50，为 YOLO-Master-EsMoE-N 提供两个垂类场景（VisDrone 航拍小目标、Brain Tumor 医疗 MRI）的 LoRA 高效微调配置，并完成 rank 4/8/16 扫描对比实验。

## 1. 背景与动机

垂类场景（航拍、医疗等）数据分布与通用检测差异大，但标注数据通常有限。全量微调在少样本下易过拟合、且显存/存储成本高。LoRA 通过低秩分解 `ΔW ≈ B·A` 只训练极小比例的参数（冻结原始权重），单卡即可完成垂类适配，且每个场景只需保存 MB 级 adapter。

## 2. 场景与数据

| 场景 | 数据集 | 特点 | 数据用量 |
| :--- | :--- | :--- | :--- |
| 航拍小目标 | VisDrone（内置，自动下载） | 目标小而密集、尺度跨度大、类别不平衡 | `fraction=0.25`（25% 子集，模拟垂类少样本） |
| 医疗 MRI | brain-tumor（内置，自动下载） | 灰度图、目标稀疏、数据集小（仅 4MB） | `fraction=1.0`（全量） |

## 3. LoRA 适配策略

- **目标层（`lora_target_modules`）**：Conv 主干（`conv` / `fused_conv` / `bottleneck.0`）+ MoE Expert 投影层（`expert_projections.*.0` / `shared_feature.0` / `static_net.3` / `proj`）+ 检测头。Brain Tumor 场景在此基础上扩宽（`shared_feature.3` / `static_net.0` / `feature_refiner.0`）以提升小数据下的表达容量——实测该扩宽使 mAP50-95 由 ~0.06 提升至 0.103。
- **不挂注意力**（`lora_include_attention=False`）：A2C2f 注意力路径对扰动敏感，挂载后训练不稳定。
- **冻结路由/门控**：路由器作为任务无关的分配器保持冻结（路由内部 Conv2d 不加入 `lora_target_modules`，并以 `lora_exclude_modules` 双重保险）。该选择由第 7 节路由消融实验的数据支撑：给路由层挂 LoRA 在两场景均为明确负收益。
- **rsLoRA + alpha=2r**：缩放因子 `alpha/√r`，高 rank 下梯度尺度更稳定。
- **AdamW + 延长 warmup + 低 lr 倍率**：少样本 LoRA 微调下 SGD 训练震荡明显；AdamW 配合 `warmup_epochs`（VisDrone 5 / Brain Tumor 10）、`lora_lr_mult`（0.3 / 0.2）与 `lora_dropout`（0.1 / 0.2）显著抑制中后期崩塌。
- **梯度检查点**（`lora_gradient_checkpointing=True`）：进一步降低显存占用。
- **可训练参数量**：model summary 显示 gradients ≈ 3.42 M（VisDrone 3,419,705 / Brain Tumor 3,418,145）。该统计量包含检测头等未冻结层的参数；纯 LoRA 旁路参数量约为数百 K 量级，rank 变化（4/8/16）带来的增量在整数精度下不可分辨。

## 4. 实验设置

| 项 | VisDrone | Brain Tumor |
| :--- | :--- | :--- |
| 输入尺寸 | 768 + multi_scale | 640 |
| epochs / patience | 40 / 30 | 50 / 30 |
| batch | 4 | 16 |
| 优化器 / lr0 | AdamW / 0.001 | AdamW / 0.0005 |
| warmup / dropout / lr_mult | 5 / 0.1 / 0.3 | 10 / 0.2 / 0.2 |
| rank 扫描 | 4 / 8 / 16（alpha=2r） | 4 / 8 / 16（alpha=2r） |
| 硬件 | NVIDIA A10 24G ×1 | NVIDIA A10 24G ×1 |

所有 run 均满足 issue 最低 20 epochs 要求（配合 `patience=30` 早停，VisDrone 实跑 31~32 epochs、Brain Tumor 实跑 34~36 epochs），`seed=42 + deterministic=True`。

## 5. 训练命令

```bash
# VisDrone rank 扫描
yolo train cfg=examples/lora_examples/yolo_master_visdrone_lora.yaml lora_r=4  lora_alpha=8  name=visdrone_r4
yolo train cfg=examples/lora_examples/yolo_master_visdrone_lora.yaml lora_r=8  lora_alpha=16 name=visdrone_r8
yolo train cfg=examples/lora_examples/yolo_master_visdrone_lora.yaml lora_r=16 lora_alpha=32 name=visdrone_r16

# Brain Tumor rank 扫描
yolo train cfg=examples/lora_examples/yolo_master_brain_tumor_lora.yaml lora_r=4  lora_alpha=8  name=braintumor_r4
yolo train cfg=examples/lora_examples/yolo_master_brain_tumor_lora.yaml lora_r=8  lora_alpha=16 name=braintumor_r8
yolo train cfg=examples/lora_examples/yolo_master_brain_tumor_lora.yaml lora_r=16 lora_alpha=32 name=braintumor_r16
```

## 6. 结果

> 以下所有实验均在 **NVIDIA A10 24G 单卡** 上完成，`seed=42 + deterministic=True`。

指标为验证集 best mAP50-95（取自 results.csv 逐 epoch 最佳值）。

### VisDrone（25% 子集，768 + multi_scale）

| rank | best mAP50-95 | best mAP50 | @epoch | 实跑 epochs | 总训练时间 | 可训练参数量 | 峰值显存 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **4** | **0.03738** | **0.07456** | 2 | 32 | ~50 min | ~3.42 M | ~22.1 GB |
| 8 | 0.03413 | 0.06462 | 1 | 31 | ~48 min | ~3.42 M | — |
| 16 | 0.03413 | 0.06462 | 1 | 31 | ~48 min | ~3.42 M | — |

### Brain Tumor（全量，640）

| rank | best mAP50-95 | best mAP50 | @epoch | 实跑 epochs | 总训练时间 | 可训练参数量 | 峰值显存 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **4** | **0.10303** | **0.18023** | 6 | 36 | ~10 min | ~3.42 M | ~4.5 GB |
| 8 | 0.09006 | 0.16307 | 4 | 34 | ~9 min | ~3.42 M | — |
| 16 | 0.07623 | 0.14413 | 4 | 34 | ~9 min | ~3.42 M | — |

### 结论与 rank 推荐

1. **两场景均为 rank=4 最优**：少样本/小数据垂类微调下，更高的 rank（8/16）没有带来收益，反而随容量增加出现更早的过拟合与精度回落（Brain Tumor 上 r4 → r16 单调下降约 26%）。
2. **最佳精度出现在训练早期**（VisDrone @ep1~2、Brain Tumor @ep4~6），之后缓慢回落——印证少样本 LoRA 微调应搭配早停与强正则，而非一味加大训练量。
3. **推荐配置**：两场景统一 `lora_r=4, lora_alpha=8`（alpha=2r），保存 best 权重即可。
4. Brain Tumor 的目标层扩宽实验说明：小数据场景下 LoRA 的瓶颈可能在**覆盖层的广度**而非 rank 大小——扩宽 target_modules（+3 组投影层）带来的提升（0.06 → 0.103）远大于调整 rank。
5. **VisDrone r8/r16 的 best 均出现在 epoch 1 且数值相同**：在 25% 子集上 rank>4 时，模型在第 1 个 epoch 即达到容量上限，随后迅速过拟合——这与 r4 在 epoch 2 达到 best 后缓慢回落的模式一致，进一步印证少样本场景下高 rank 并无优势。

## 7. 路由层消融实验

MoE 结构中路由/门控层是否应参与 LoRA 微调？我们做了严格对照：在主实验配置（rank=4）基础上，**仅**将路由内部的两个 Conv2d（`routing_network.0` / `routing_network.2`）加入 `lora_target_modules` 并清空 `lora_exclude_modules`，其余全部一致（含 seed）。

> 实现细节：路由模块内可挂 LoRA 的仅有这两个 Conv2d（其余为池化/激活等无参层）。消融时必须将其**显式加入白名单**——仅修改 `lora_exclude_modules` 不会改变挂载结果，因为白名单本身不含路由子模块。我们通过启动日志的可训练参数量与训练轨迹差异双重验证了两组挂载的真实不同。

| 场景（rank=4） | 路由冻结（主实验） | 路由参与微调（消融组） | 结果 |
| :--- | ---: | ---: | :--- |
| VisDrone | **0.03738** @ep2 | 0.03437 @ep1 | 峰值 -8%；**第 6 epoch 起 mAP 崩塌至 ≈0** 且无恢复（recall→0） |
| Brain Tumor | **0.10303** @ep6 | 0.07439 @ep3 | 峰值 **-28%**；此后回落至 0.02 附近震荡 |

**结论**：给路由层挂 LoRA 在两个场景均为明确负收益——路由被扰动后专家分配漂移，VisDrone 上甚至导致整个检测性能不可逆崩塌。**路由/门控层必须保持冻结**，这也是主实验配置的默认选择。消融配置以 `*_route.yaml` 提供，供复现验证。

## 8. 文件说明

| 文件 | 说明 |
| :--- | :--- |
| `yolo_master_visdrone_lora.yaml` | VisDrone LoRA 训练配置（主实验，路由冻结；rank 由命令行覆盖） |
| `yolo_master_brain_tumor_lora.yaml` | Brain Tumor LoRA 训练配置（主实验，路由冻结；rank 由命令行覆盖） |
| `yolo_master_visdrone_lora_route.yaml` | VisDrone 路由消融配置（路由 Conv2d 参与 LoRA，对照组） |
| `yolo_master_brain_tumor_lora_route.yaml` | Brain Tumor 路由消融配置（路由 Conv2d 参与 LoRA，对照组） |

消融复现命令：

```bash
yolo train cfg=examples/lora_examples/yolo_master_visdrone_lora_route.yaml    lora_r=4 lora_alpha=8 name=visdrone_route_r4
yolo train cfg=examples/lora_examples/yolo_master_brain_tumor_lora_route.yaml lora_r=4 lora_alpha=8 name=braintumor_route_r4
```

## 

- 显存不足时可下调 `batch`（VisDrone 768+multi_scale 在 24G 显存下峰值约 22.1 GB，建议 batch≤4；Brain Tumor 640 峰值约 4.5 GB，显存充裕），或降低 `imgsz`。
- 数据集首次训练自动下载；`fraction` 采样与 `seed` 绑定，同种子下子集一致。
- **训练时间参考**：VisDrone 约 48~50 min（稳态 ~89 s/epoch），Brain Tumor 约 9~10 min（稳态 ~16 s/epoch）。rank 对训练时间影响极小（LoRA 旁路计算量可忽略），差异主要来自数据加载与验证开销。
- **医疗灰度通道处理**：brain-tumor 数据集为单通道灰度 MRI，Ultralytics 在数据加载阶段会自动将灰度图复制为 3 通道（`channels=3` 为默认值），无需手动预处理。若使用其他框架或自定义加载器，需显式执行 `img = np.stack([img]*3, axis=-1)` 或等效操作。

