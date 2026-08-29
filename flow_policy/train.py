"""
双臂 flow-matching 策略训练（最小可用版本，无 rollout 评测）。

用法：
    accelerate launch --mixed_precision=bf16 \
        flow_policy/train.py --config flow_policy/configs/train.yaml

bf16 由 `accelerate launch --mixed_precision=bf16` 指定，代码不硬编码精度。
"""
import argparse
import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import yaml
from accelerate import Accelerator
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from models.model import FlowPolicy
from datasets import create_dataloader


def get_image_transform(training: bool, use_aug: bool = True):
    """DINOv2 期望 [0,1] 输入并在内部做 ImageNet Normalize，
    因此这里只 Resize + (ColorJitter) + ToTensor，绝不加 Normalize。"""
    return transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0
            )
            if (training and use_aug)
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),  # 0-1
        ]
    )


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, betas):
    # 只训 transformer；vision 已冻结。排除 ModuleAttrMixin 的空 _dummy_variable。
    params = [
        p for p in model.parameters() if p.requires_grad and p.numel() > 0
    ]
    return torch.optim.AdamW(params, lr=lr, betas=tuple(betas), weight_decay=weight_decay)


def main(cfg: Dict):
    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=cfg.get("save_path", "outputs"),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
    )
    accelerator.init_trackers("flow-policy")

    # 打印完整训练配置：日志首屏即参数确认依据，方便与 train.yaml 核对
    accelerator.print("=" * 60)
    accelerator.print("flow_policy Training config:")
    accelerator.print(yaml.dump(cfg, sort_keys=False, default_flow_style=False))
    accelerator.print("=" * 60)

    # seed
    seed = cfg.get("seed", 42)
    random.seed(seed + accelerator.process_index)
    np.random.seed(seed + accelerator.process_index)
    torch.manual_seed(seed + accelerator.process_index)

    # ---- model ----
    model = FlowPolicy(**cfg["model"])
    accelerator.print(
        f"trainable params: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M "
        f"(vision frozen)"
    )
    optim = build_optimizer(
        model,
        lr=float(cfg["optim"]["lr"]),  # yaml 6.x 把 `1e-4` 解析成 str，强制 float
        weight_decay=float(cfg["optim"].get("weight_decay", 0.0)),
        betas=[float(b) for b in cfg["optim"].get("betas", [0.9, 0.95])],
    )
    # ---- resume：从 ckpt 继续（对齐 X-VLA/train.py：权重/optimizer/RNG 都在 prepare 之前灌回
    # 裸模型，global_step 从保存处续跑；lr 是 global_step 的纯函数，循环内公式自动取调度点）----
    global_step = 0
    resume_path = cfg.get("resume")
    if resume_path:
        accelerator.print(f"Resuming from {resume_path}")
        # torch>=2.6 默认 weights_only=True，会拒绝 ckpt 里的 numpy RNG 状态；
        # ckpt 为自产可信文件，显式 weights_only=False
        _ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        # 1) 模型权重：load 进裸 FlowPolicy（prepare 包装前恢复，键直接匹配）
        _missing, _unexpected = model.load_state_dict(_ckpt["model_state_dict"])
        accelerator.print(
            f"  model weights: missing={len(_missing)} unexpected={len(_unexpected)}"
        )
        # 2) optimizer 动量/步数（AcceleratedOptimizer 会委托到底层 AdamW，prepare 前灌回最稳）
        optim.load_state_dict(_ckpt["optimizer_state_dict"])
        # 3) RNG（torch/cuda/numpy/random，per-rank）：恢复 dropout 等随机序列。
        #    旧 ckpt（ckpt-1000~3000）无此字段则跳过，新保存的 ckpt 自动带上。
        rng = _ckpt.get(f"rng_state_rank{accelerator.process_index}")
        if rng is not None:
            torch.set_rng_state(rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"][: torch.cuda.device_count()])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["random"])
            accelerator.print("  RNG state restored")
        else:
            accelerator.print("  no RNG state in ckpt; skip RNG restore")
        # 4) global_step 续跑：3000 > warmup(400)，lr 公式自动取 base_lr，无需额外处理
        global_step = int(_ckpt["global_step"])
        accelerator.print(
            f"  resumed at global_step={global_step}; "
            f"lr schedule picks {float(cfg['optim']['lr'])} (past warmup)"
        )

    model, optim = accelerator.prepare(model, optim)

    # ---- data ----
    # IterableDataset 含字符串字段（language_instruction），必须 device_placement=[False]，
    # 否则走 DataLoaderDispatcher 无法拼接字符串而崩溃；batch 由下方显式搬到 device。
    # precompute 模式：precomputed_embeddings=true 时从 npy 索引 embedding，不碰视频
    precomputed = bool(cfg.get("precomputed_embeddings", False))
    train_dataloader = create_dataloader(
        batch_size=cfg["batch_size"],
        metas_path=cfg["metas_path"],
        num_actions=cfg["model"]["num_actions"],
        action_mode="arx_ee6d",
        training=True,
        num_workers=cfg.get("num_workers", 8),
        use_frame_weight=cfg.get("use_frame_weight", False),
        image_transform=get_image_transform(
            training=True, use_aug=cfg.get("image_aug", True)
        ),
        precomputed_dir=(cfg.get("precomputed_dir") if precomputed else None),
        n_obs_steps=cfg["model"].get("n_obs_steps", 1),
    )
    train_dataloader = accelerator.prepare(train_dataloader, device_placement=[False])
    train_iter = iter(train_dataloader)

    # ---- training loop ----
    iters = cfg.get("iters", 100_000)
    save_interval = cfg.get("save_interval", 5000)
    log_interval = cfg.get("log_interval", 20)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    warmup_steps = cfg.get("warmup_steps", 2000)
    base_lr = float(cfg["optim"]["lr"])
    save_path = cfg.get("save_path", "outputs")
    os.makedirs(save_path, exist_ok=True)

    model.train()
    # global_step 已在 resume 块初始化（非 resume 时为 0；resume 则从保存处续跑）
    # 累积批次日志（对齐 X-VLA train.py）：每个 micro-batch 按真实样本数加权累积 loss
    # 分量，在 optimizer step 边界跨 micro-batch/rank 归并，日志对应完整 effective batch
    # 而不是最后一个 micro-batch。
    effective_loss_sums: Dict[str, torch.Tensor] = {}
    effective_loss_total_sum = None
    effective_batch_samples_local = 0
    while global_step < iters:
        with accelerator.accumulate(model):
            batch = next(train_iter)
            # 模型无 domain 概念、不用语言、忽略 image_mask（3 相机恒在）：
            # 去掉这些字段，只留 image_input / proprio / action
            batch.pop("is_key_frame", None)
            batch.pop("language_instruction", None)
            batch.pop("domain_id", None)
            batch.pop("image_mask", None)
            inputs = {
                k: (
                    v.to(accelerator.device, non_blocking=True)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in batch.items()
            }

            loss_dict: Dict[str, torch.Tensor] = model(
                **inputs, precomputed=precomputed
            )
            loss = sum(loss_dict.values())

            # 日志聚合：loss_dict 是 micro-batch 的 batch-mean，按真实 micro-batch 样本数
            # 加权累积（detach 只留标量），在 sync_gradients 边界归并成 effective-batch mean。
            micro_batch_samples = int(inputs["action"].shape[0])
            for name, value in loss_dict.items():
                weighted = value.detach().float() * micro_batch_samples
                if name in effective_loss_sums:
                    effective_loss_sums[name] += weighted
                else:
                    effective_loss_sums[name] = weighted
            weighted_total = loss.detach().float() * micro_batch_samples
            if effective_loss_total_sum is None:
                effective_loss_total_sum = weighted_total
            else:
                effective_loss_total_sum += weighted_total
            effective_batch_samples_local += micro_batch_samples

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                # 线性 warmup -> constant
                lr = base_lr * min(1.0, (global_step + 1) / max(warmup_steps, 1))
                for g in optim.param_groups:
                    g["lr"] = lr
                # 剪裁前梯度 L2 范数（与 X-VLA 日志一致）
                grad_norm = float(
                    sum(
                        p.grad.norm().item() ** 2
                        for p in model.parameters()
                        if p.grad is not None
                    )
                    ** 0.5
                )
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1

            if global_step % log_interval == 0:
                if (
                    effective_loss_total_sum is None
                    or effective_batch_samples_local <= 0
                ):
                    raise RuntimeError(
                        "No micro-batch losses collected for effective-batch logging"
                    )
                # 一次 collective 同时归并各 loss 分量与 total；除以全局真实样本数
                # 得到本 optimizer update 对应的 effective-batch mean loss。
                loss_names = tuple(effective_loss_sums)
                local_stats = torch.stack(
                    [effective_loss_sums[name] for name in loss_names]
                    + [
                        effective_loss_total_sum,
                        effective_loss_total_sum.new_tensor(
                            float(effective_batch_samples_local)
                        ),
                    ]
                )
                global_stats = accelerator.reduce(local_stats, reduction="sum")
                effective_batch_samples_global = float(global_stats[-1].item())
                denominator = max(effective_batch_samples_global, 1.0)
                logs = {
                    name: float(global_stats[index].item() / denominator)
                    for index, name in enumerate(loss_names)
                }
                logs["loss_total"] = float(global_stats[-2].item() / denominator)
                logs["effective_batch_samples"] = effective_batch_samples_global
                logs["grad_norm"] = grad_norm
                logs["step"] = global_step
                logs["lr"] = lr
                accelerator.log(logs, step=global_step)
                # 打印各 loss 分量（去 _loss 后缀），便于观察各动作头收敛
                loss_parts = " ".join(
                    f"{k[:-len('_loss')]}={v:.4f}"
                    for k, v in logs.items()
                    if k.endswith("_loss") and k != "loss_total"
                )
                accelerator.print(
                    f"[{global_step}/{iters}] "
                    f"loss={logs['loss_total']:.4f} [{loss_parts}] "
                    f"batch={int(logs['effective_batch_samples'])} "
                    f"grad_norm={logs['grad_norm']:.4f} "
                    f"lr={logs['lr']:.2e}"
                )

            # 无论本 step 是否打印，都必须在 optimizer step 边界清空累积器，
            # 避免把多个 optimizer update 混入下一次日志。
            effective_loss_sums = {}
            effective_loss_total_sum = None
            effective_batch_samples_local = 0

            if global_step % save_interval == 0:
                ckpt_path = os.path.join(save_path, f"ckpt-{global_step}.pt")
                # per-rank RNG（对齐 X-VLA save_rng_state：torch/cuda/numpy/random），供 resume
                # 恢复 dropout 等随机序列。单卡只有 rank0，嵌入同一文件；accelerator.save 只在
                # 主进程写盘，多卡时每 rank 需另存 rng_state_rank{k}.pt。
                accelerator.save(
                    {
                        "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "global_step": global_step,
                        f"rng_state_rank{accelerator.process_index}": {
                            "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available()
                            else None,
                            "numpy": np.random.get_state(),
                            "random": random.getstate(),
                        },
                    },
                    ckpt_path,
                )
                accelerator.print(f"Saved ckpt -> {ckpt_path}")

    accelerator.end_training()
    accelerator.print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to train.yaml")
    parser.add_argument("--metas_path", type=str, default=None, help="override meta.json path")
    parser.add_argument("--iters", type=int, default=None, help="override iters")
    parser.add_argument("--batch_size", type=int, default=None, help="override batch_size")
    parser.add_argument("--num_workers", type=int, default=None, help="override num_workers")
    parser.add_argument("--resume", type=str, default=None,
                        help="从该 ckpt 继续训练（不 warmup，global_step 从保存处续跑）")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for k in ["metas_path", "iters", "batch_size", "num_workers"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.resume:
        cfg["resume"] = args.resume

    main(cfg)
