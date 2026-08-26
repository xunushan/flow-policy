"""
一次性把指定 episodes 的三相机视频编码成 DINOv2 patch 特征，按相机各存一个 npy。

用法（GPU 服务器）：
    python flow_policy/precompute.py --config flow_policy/configs/train.yaml \
        --out_dir data_precompute/goai_task1

输出（每相机一个文件，fp32）：
    {out_dir}/cam_high.npy            [N_total, 256, 384]
    {out_dir}/cam_left_wrist.npy      [N_total, 256, 384]
    {out_dir}/cam_right_wrist.npy     [N_total, 256, 384]
    {out_dir}/precompute_meta.json    行索引元数据（训练时 PrecomputedObs 读它定位帧）

帧行序 = datalist（episode_index 升序）下的 episode 顺序累积；episode 内帧序不变。
处理范围 = meta.json 的 episodes 字段（全量训练留 null = 全部），或 --episodes 覆盖。
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from models.vision import DinoV2Encoder
from datasets.domain_handler.registry import get_handler_cls


def main(cfg, out_dir: str, episodes=None, batch_size: int = 128):
    with open(cfg["metas_path"]) as f:
        meta = json.load(f)
    robot_type = meta.get("robot_type", "arx_x5_ee")
    Handler = get_handler_cls(robot_type)
    meta.setdefault("datalist", Handler.build_datalist(meta))
    if episodes is not None:
        allowed = set(int(e) for e in episodes)
        meta["episodes"] = sorted(allowed)
        meta["datalist"] = [i for i in meta["datalist"] if i in allowed]
    if not meta["datalist"]:
        raise ValueError("no episodes to precompute (check meta.json 'episodes' / --episodes)")

    handler = Handler(meta=meta, num_views=len(meta.get("camera_keys", [])))
    camera_keys = handler.camera_keys

    # 冻结 DINOv2 编码器（与训练模型同一实现；eval 模式关闭 dropout）
    enc = DinoV2Encoder(name="dinov2_vits14", feature_key="x_norm_patchtokens").eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = enc.to(device)
    n_patches, emb_dim = 256, enc.emb_dim
    # 与训练 on-the-fly 路径完全一致的预处理：PIL BICUBIC Resize + ToTensor(0-1)。
    # 不要用 tensor F.interpolate——bicubic 会轻微越出 [0,1] 触发 DINOv2 断言，且与 PIL 核不一致。
    to_tensor = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
    )

    # episode -> 全局行偏移（datalist 顺序累积）
    offsets = {}
    total = 0
    for ep_idx in meta["datalist"]:
        length = int(handler.episodes[ep_idx]["length"])
        offsets[ep_idx] = {"start": total, "frames": length}
        total += length
    print(f"precompute {len(meta['datalist'])} episodes, {total} frames, {len(camera_keys)} cameras -> {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    cam_files = {}
    for cam in camera_keys:
        fname = f"{cam.rsplit('.', 1)[-1]}.npy"
        cam_files[cam] = fname
        mm = np.lib.format.open_memmap(
            os.path.join(out_dir, fname), mode="w+", dtype="float32", shape=(total, n_patches, emb_dim)
        )
        for ep_idx in meta["datalist"]:
            ep = handler.episodes[ep_idx]
            off = offsets[ep_idx]["start"]
            video = handler._decode_episode_video(cam, ep)  # [T, H, W, C] uint8
            T = video.shape[0]
            frames = [to_tensor(Image.fromarray(f).convert("RGB")) for f in video]  # [3,224,224]
            t = torch.stack(frames, dim=0)  # [T,3,224,224] 0-1
            embs = []
            for b in range(0, T, batch_size):
                chunk = t[b : b + batch_size].to(device)
                with torch.no_grad():
                    embs.append(enc(chunk).float().cpu().numpy())  # [B,256,384]
            arr = np.concatenate(embs, axis=0)
            mm[off : off + T] = arr
            print(f"  cam={cam.rsplit('.', 1)[-1]} ep={ep_idx} frames={T} rows=[{off},{off + T})")
        mm.flush()
        del mm

    meta_out = {
        "n_patches": n_patches,
        "emb_dim": emb_dim,
        "dtype": "float32",
        "feature_key": "x_norm_patchtokens",
        "camera_files": cam_files,
        "episode_offsets": {str(k): v for k, v in offsets.items()},
        "total_frames": total,
        "fps": handler.fps,
        "query_duration": handler.qdur,
    }
    with open(os.path.join(out_dir, "precompute_meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"done. sidecar -> {out_dir}/precompute_meta.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to train.yaml (读取 metas_path)")
    parser.add_argument("--out_dir", type=str, required=True, help="输出目录（每相机一个 npy + sidecar）")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="覆盖 meta.json 的 episodes 过滤")
    parser.add_argument("--batch_size", type=int, default=128, help="DINOv2 编码批大小")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg, args.out_dir, episodes=args.episodes, batch_size=args.batch_size)
