# flow_policy — 双臂 Flow-Matching 策略（最小可用版本）

GOAI 2026 ARX 双臂的流匹配策略训练代码。**不动** patch_policy / X-VLA 原仓库，此文件夹自包含。

## 架构

```
FlowPolicy
├── vision      DINOv2 (dinov2_vits14) 冻结，三视图 224x224 → 每视图 256 patch tokens
├── transformer patch_policy 的 TransformerForDiffusion（causal attention + patch-aware memory mask）
└── action head X-VLA 风格 flow matching：x_t = t·noise + (1-t)·gt，预测干净 action
                loss = ARXEE6DActionSpace.compute_loss（分量加权 MSE，pre/post no-op）
```

- 视觉 token 流：`image_input [B,T_obs,V,3,H,W]`(0-1) → DINOv2 → `obs_cond [B, T_obs·768, 384]`
  （T_obs=n_obs_steps 观测窗口，窗口序最旧在前；3 视图 × 256 patches 全部独立进 transformer，
  head 图像关键由模型自学习）。precompute 模式下 `image_input` 直接是 embedding `[B,T_obs,V,P,E]`，
  跳过 DINOv2。
- proprio（当前 ee 位姿 20 维）拼进每个 action token（decoder `input_dim=40`）；
  时间 t 走 patch_policy 的 cond token（SinusoidalPosEmb）
- 无 domain / 无语言指令概念

## 目录结构

```
flow_policy/
├── train.py                   # 训练主循环（无 rollout）
├── configs/train.yaml         # 配置（键名参照 patch_policy configs）
├── models/
│   ├── model.py               # FlowPolicy（核心）
│   ├── vision.py              # DinoV2Encoder（copy patch_policy dino.py）
│   ├── transformer.py         # TransformerForDiffusion（copy patch_policy 骨干）
│   └── action_hub.py          # ARXEE6DActionSpace（copy X-VLA）
└── datasets/                  # copy X-VLA xvla_datasets/（registry 精简到 arx_x5_ee）
```

## 运行

```bash
# 1) （可选）预计算 embedding：把视频编码成 DINOv2 patch 特征，每相机一个 npy。
#    单 task ~5 万帧 fp32 ≈ 58GB。不跑则走 on-the-fly 视频解码（保底）。
python flow_policy/precompute.py --config flow_policy/configs/train.yaml \
    --out_dir data_precompute/goai_task1

# 2) 训练（bf16 由 accelerate 指定，代码不硬编码精度）
#    train.yaml: precomputed_embeddings: true  + precomputed_dir: data_precompute/goai_task1
#    （false + 不设 precomputed_dir = on-the-fly）
accelerate launch --mixed_precision=bf16 \
    flow_policy/train.py --config flow_policy/configs/train.yaml

# 单 task 训练：在处理数据时把 meta.json 的 episodes 填成该 task 的 episode_index 列表即可
```

### precompute 产物

每相机一个 npy（`{out_dir}/{cam}.npy`，`[N_total, 256, 384]` fp32，episode 按 datalist 序连续行）+
`precompute_meta.json`（`episode_offsets` 帧→行映射，训练时 `PrecomputedObs` 用 mmap 随机索引）。

- precompute 只做 Resize(224, BICUBIC) + ToTensor(0-1)，**不做增强**（每帧固定编码，无 per-epoch 增强多样性）；ImageNet 归一化由 DINOv2 内部完成。
- 若中途换 DINOv2 backbone / 图像尺寸，需重新 precompute。
- 数据集变大时可按 task 分别 precompute 到不同目录，训练时指到对应目录。

## meta.json 格式

**由你放在数据路径下**，`train.yaml` 的 `metas_path` 指向它。本仓库不生成。

```json
{
  "codebase_version": "v3.0",
  "dataset_name": "goai_arx",
  "root_path": "/path/to/lerobot_v30_ee",
  "robot_type": "arx_x5_ee",
  "camera_keys": [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist"
  ],
  "fps": 25,
  "query_duration": 1.0,
  "episodes": null
}
```

- `robot_type` **必须写 `arx_x5_ee`**（registry 注册名；数据里的 `unified_robot` 未注册）
- `root_path` 指向数据集根目录（不是 `meta/`），handler 内部自动做 16→20 维 ee6d 转换
- `episodes`：全量训练留 `null`；单 task 训练填该 task 的 episode_index 列表（12 task 各 100 episodes 连续编号）

## 关键注意点

1. **归一化**：DINOv2 期望 `[0,1]` 输入并内部做 ImageNet Normalize。
   train.py 传入的 `image_transform` 只做 Resize+ColorJitter+ToTensor，**绝不加 Normalize**，
   覆盖 X-VLA 默认带 `Normalize(..., inplace=True)` 的 `image_aug`。
2. **causal 语义**：保留 patch_policy 的 decoder 因果 mask 与 patch-aware memory_mask；
   `n_obs_steps=1` 时 memory_mask 全允许（decoder 每步都看全图）。
3. **bf16 对齐**：`obs_cond` 在 model 内 cast 到 transformer 参数 dtype。
4. **gripper 约定**：handler `invert_gripper=False`（1=开，连续），与 ARX 连续 MSE 匹配，不要额外 sigmoid。
5. **DINOv2 权重**：首次运行联网下载 `facebookresearch/dinov2:b48308a`（缓存到 `~/.cache/torch/hub`）。
6. **视频解码是瓶颈**：`num_workers=4~8`。
