"""
PrecomputedObs — 预计算 DINOv2 embedding 的 mmap 读取器（每相机一个 npy）。

文件布局（由 precompute.py 生成）：
    {dir}/precompute_meta.json        行索引元数据
    {dir}/{cam_short}.npy             [N_total, n_patches, emb_dim] fp32，episode 按序连续行

帧映射：episode_offsets[ep_idx] = {"start": 首行全局偏移, "frames": 帧数}，
行号 = start + 帧内局部索引。与数据表的 dataset_from/to_index 无关（按 datalist 顺序排）。

每个 DataLoader worker 各自打开自己的 memmap（np.memmap 按需分页，无一次性大内存拷贝）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def cam_short_name(cam_key: str) -> str:
    """observation.images.cam_high -> cam_high（npy 文件名用）。"""
    return cam_key.rsplit(".", 1)[-1]


class PrecomputedObs:
    def __init__(self, precomputed_dir: str, camera_keys, num_views: int):
        self.dir = Path(precomputed_dir)
        with open(self.dir / "precompute_meta.json") as f:
            meta = json.load(f)
        self.n_patches = int(meta["n_patches"])
        self.emb_dim = int(meta["emb_dim"])
        self.dtype = np.dtype(meta["dtype"])
        self.camera_files = meta["camera_files"]  # {cam_key: filename}
        # {ep_idx: {"start": int, "frames": int}}
        self.episode_offsets = {
            int(k): (int(v["start"]), int(v["frames"])) for k, v in meta["episode_offsets"].items()
        }
        self.num_views = num_views
        self.camera_keys = list(camera_keys)
        self._maps = {}

    def _mmap(self, cam_key: str) -> np.ndarray:
        if cam_key not in self._maps:
            path = self.dir / self.camera_files[cam_key]
            self._maps[cam_key] = np.load(str(path), mmap_mode="r")
        return self._maps[cam_key]

    def get_window(self, cam_keys, ep_idx: int, frame_idx: int, n_obs_steps: int) -> np.ndarray:
        """取 (ep_idx, frame_idx) 及其前 n_obs_steps-1 帧的 embedding 窗口。

        返回 [n_obs_steps, V, P, E] fp32；越界（frame_idx-k<0）时重复首帧
        （与 patch_policy 的 repeat_start_to_length 语义一致）。
        """
        start, frames = self.episode_offsets[int(ep_idx)]
        rows = []
        for k in range(n_obs_steps - 1, -1, -1):
            fidx = max(0, frame_idx - k)
            row = start + fidx
            if row < start + frames:
                arr = [np.asarray(self._mmap(c)[row]) for c in cam_keys]  # [V, P, E]
            else:  # 防御：越界帧退化为最后一帧（正常采样不会发生）
                arr = [np.asarray(self._mmap(c)[start + frames - 1]) for c in cam_keys]
            rows.append(arr)
        return np.stack(rows, axis=0).astype(np.float32)  # [T_obs, V, P, E]
