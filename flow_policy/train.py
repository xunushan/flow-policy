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
    global_step = 0
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

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                # 线性 warmup -> constant
                lr = base_lr * min(1.0, (global_step + 1) / max(warmup_steps, 1))
                for g in optim.param_groups:
                    g["lr"] = lr
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1

            if global_step % log_interval == 0:
                logs = {k: v.detach().float().item() for k, v in loss_dict.items()}
                logs["step"] = global_step
                logs["lr"] = lr
                accelerator.log(logs, step=global_step)
                accelerator.print(
                    f"[{global_step}/{iters}] "
                    + " | ".join(
                        f"{k}={v:.4f}" for k, v in logs.items() if k != "step"
                    )
                )

            if global_step % save_interval == 0:
                ckpt_path = os.path.join(save_path, f"ckpt-{global_step}.pt")
                accelerator.save(
                    {
                        "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "global_step": global_step,
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
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for k in ["metas_path", "iters", "batch_size", "num_workers"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    main(cfg)
