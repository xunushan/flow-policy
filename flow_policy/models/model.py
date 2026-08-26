"""
FlowPolicy — 双臂 flow-matching 策略（最小可用版本）。

组合：
- vision: DINOv2 (dinov2_vits14) 冻结，三视图 224x224 → 每视图 256 patch tokens
- transformer: patch_policy 的 TransformerForDiffusion（causal attention + patch-aware memory mask）
- action head: X-VLA 的 flow matching（x_t = t·noise + (1-t)·gt，预测干净 action）+ ARXEE6DActionSpace loss
"""
from typing import Dict

import torch
import torch.nn as nn

from .vision import DinoV2Encoder
from .transformer import TransformerForDiffusion
from .action_hub import build_action_space


class FlowPolicy(nn.Module):
    def __init__(
        self,
        dim_action: int = 20,
        dim_propio: int = 20,
        cond_dim: int = 384,
        n_patches: int = 256,
        views: int = 3,
        num_actions: int = 10,
        n_obs_steps: int = 1,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.1,
    ):
        super().__init__()
        self.dim_action = dim_action
        self.dim_propio = dim_propio
        self.num_actions = num_actions
        self.n_obs_steps = n_obs_steps

        # 视觉 encoder：冻结 DINOv2（patch tokens，输入必须 [0,1]）
        self.vision = DinoV2Encoder(
            name="dinov2_vits14", feature_key="x_norm_patchtokens"
        )
        for p in self.vision.parameters():
            p.requires_grad_(False)

        # patch_policy 核心骨干：causal attention decoder + patch-aware memory mask
        # input_dim = action(20) + proprio(20)，proprio 拼进每个 action token（X-VLA 语义）
        # n_obs_steps>1：obs_cond 按窗口顺序平铺 [T_obs*V*P, E]，memory_mask 块因果对应
        self.transformer = TransformerForDiffusion(
            input_dim=dim_action + dim_propio,
            output_dim=dim_action,
            horizon=num_actions,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_embd,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            n_cond_layers=0,
            n_patches=n_patches * views,  # 768 = 3 views × 256 patches
        )

        # ARX 双臂 ee6d action space：dim=20，pre/post no-op，分量加权 MSE
        self.action_space = build_action_space("arx_ee6d")

    # ------------------------------------------------------------------ #
    # 视觉编码 -> obs_cond [B, T_obs*V*P, E]（窗口序：最旧帧在前）
    #   precomputed=True : image_input 即 embedding [B, T_obs, V, P, E]
    #   precomputed=False: image_input 为像素 [B, T_obs, V, 3, H, W] (0-1)，DINOv2 编码
    # ------------------------------------------------------------------ #
    def encode_visual(
        self, image_input: torch.Tensor, precomputed: bool = False
    ) -> torch.Tensor:
        if precomputed:
            return image_input.reshape(image_input.shape[0], -1, image_input.shape[-1])
        with torch.no_grad():
            # vision 支持任意前置维度：collapse (B,T_obs,V) -> 批量编码
            feats = self.vision(image_input)  # [B, T_obs, V, P, E]
        return feats.reshape(image_input.shape[0], -1, feats.shape[-1])

    # ------------------------------------------------------------------ #
    # 训练 forward：flow matching 插值 + 预测干净 action
    # ------------------------------------------------------------------ #
    def forward(
        self,
        image_input: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        precomputed: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, T = action.shape[:2]
        assert T == self.num_actions, f"action seq {T} != num_actions {self.num_actions}"

        obs_cond = self.encode_visual(image_input, precomputed)
        obs_cond = obs_cond.to(self.transformer.dtype)
        proprio = proprio.to(self.transformer.dtype)
        action = action.to(self.transformer.dtype)

        # flow matching 插值（对齐 X-VLA modeling_xvla.py forward）
        t = (
            torch.rand(1, device=action.device)
            + torch.arange(B, device=action.device).float() / B
        ) % (1 - 1e-5)  # [B]，分层均匀采样
        noise = torch.randn_like(action)
        action_noisy = noise * t.view(-1, 1, 1) + action * (1 - t.view(-1, 1, 1))  # x_t

        # 对齐 X-VLA modeling_xvla.py:185 —— preprocess 必须在进 transformer 前对 proprio/x_t 应用。
        # ARXEE6DActionSpace.preprocess 当前是 no-op（连续 gripper），但保留调用以保持 action space 语义正确。
        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)

        # proprio 拼进每个 action token：decoder input [B, T, 40]
        proprio_tokens = proprio_m.unsqueeze(1).expand(B, T, self.dim_propio)
        sample = torch.cat([action_noisy_m, proprio_tokens], dim=-1)

        pred = self.transformer(sample, t, cond=obs_cond)  # 预测干净 action [B,T,20]
        return self.action_space.compute_loss(pred, action)

    # ------------------------------------------------------------------ #
    # 推理：迭代去噪（训练不使用，供后续单 task 推理/评测）
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate_actions(
        self,
        image_input: torch.Tensor,
        proprio: torch.Tensor,
        steps: int = 10,
        precomputed: bool = False,
    ) -> torch.Tensor:
        B = image_input.shape[0]
        obs_cond = self.encode_visual(image_input, precomputed)
        obs_cond = obs_cond.to(self.transformer.dtype)
        proprio = proprio.to(self.transformer.dtype)

        x1 = torch.randn(
            B, self.num_actions, self.dim_action,
            device=proprio.device, dtype=proprio.dtype,
        )
        action = torch.zeros_like(x1)
        for i in range(steps, 0, -1):
            t = torch.full((B,), i / steps, device=proprio.device, dtype=proprio.dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t.view(-1, 1, 1))
            # 对齐 X-VLA generate_actions —— preprocess 在每次迭代进 transformer 前应用
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            proprio_tokens = proprio_m.unsqueeze(1).expand(B, self.num_actions, self.dim_propio)
            sample = torch.cat([x_t_m, proprio_tokens], dim=-1)
            action = self.transformer(sample, t, cond=obs_cond)
        return self.action_space.postprocess(action)
