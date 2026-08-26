# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations
import io, json, numpy as np, pyarrow.parquet as pq, av, cv2
from mmengine import fileio
from PIL import Image
from scipy.spatial.transform import Rotation as R
import h5py
from typing import Sequence, Dict
import torch

def load_episode_indices(path, split: str = "train") -> list[int]:
    """从 splits 索引文件读取指定划分（train/val）的 episode 索引列表。

    兼容常见格式：
      - JSON dict：取 `split` 键（如 {"train": [...], "val": [...]}，RoboDojo splits 格式）；
      - JSON 数组：直接作为索引列表（忽略 split）；
      - JSONL / 每行一个整数：行内为含 `episode_index` 的 dict 或单个整数。
    返回排序后的去重整数列表。缺 split 键 / 空列表时抛 ValueError。
    """
    from pathlib import Path
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        idxs = obj.get(split)
        if idxs is None:
            raise ValueError(f"splits file {p} has no '{split}' key (keys: {list(obj)[:10]})")
        idxs = list(idxs)
    elif isinstance(obj, list):
        idxs = obj
    else:
        idxs = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                entry = line
            if isinstance(entry, dict):
                idxs.append(entry["episode_index"])
            elif isinstance(entry, (int, float)):
                idxs.append(int(entry))
            elif isinstance(entry, str):
                idxs.append(int(entry))
            else:
                raise ValueError(f"unrecognized splits line: {line!r}")
    result = sorted({int(i) for i in idxs})
    if not result:
        raise ValueError(f"splits file {p} yielded empty '{split}' index list")
    return result


def read_bytes(path: str) -> bytes:
    return fileio.get(path)

def open_h5(path: str) -> h5py.File:
    try: return h5py.File(path, "r")
    except OSError: return h5py.File(io.BytesIO(read_bytes(path)), "r")

def read_video_to_frames(path: str) -> np.ndarray:
    buf = io.BytesIO(read_bytes(path)); container = av.open(buf, options={'threads': '2'})
    frames = []
    for packet in container.demux(video=0):
        for f in packet.decode(): frames.append(f.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames, axis=0)

def read_parquet(path: str) -> dict:
    buf = io.BytesIO(read_bytes(path))
    return pq.read_table(buf).to_pydict()

def decode_image_from_bytes(x) -> Image.Image:
    if isinstance(x, (bytes, bytearray)): x = np.frombuffer(x, dtype=np.uint8)
    rgb = cv2.imdecode(x, cv2.IMREAD_COLOR)
    if rgb is None:
        rgb = np.frombuffer(x, dtype=np.uint8)
        if rgb.size == 2764800: rgb = rgb.reshape(720, 1280, 3)
        elif rgb.size == 921600: rgb = rgb.reshape(480, 640, 3)
    return Image.fromarray(rgb)

def quat_to_rotate6d(q: np.ndarray, scalar_first = False) -> np.ndarray:
    return R.from_quat(q, scalar_first = scalar_first).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

def euler_to_rotate6d(q: np.ndarray, pattern: str = "xyz") -> np.ndarray:
    return R.from_euler(pattern, q, degrees=False).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))


def rotate6d_to_xyz(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    return R.from_matrix(rot_mats).as_euler('xyz')

def rotate6d_to_quat(v6: np.ndarray, scalar_first = False) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    return R.from_matrix(rot_mats).as_quat(scalar_first = scalar_first)


def ee16_to_xvla20(value: np.ndarray, *, invert_gripper: bool = True) -> np.ndarray:
    """16 维 end-effector 布局 -> 20 维 X-VLA 布局（每臂 [xyz, quat_wxyz, g] -> [xyz, rot6d, g']）。

    gripper 默认反转（1-g，X-VLA EE6D 约定 1=闭合）；20 维输入原样返回。
    （自 LeRobotV3RoboDojoHandler._to_20d 上移，供 handler 与评估共用。）
    """
    value = np.asarray(value, dtype=np.float32)
    if value.shape[-1] == 20:
        return value
    if value.shape[-1] != 16:
        raise ValueError(f"unsupported last dim {value.shape[-1]}; expected 16 or 20")
    left, right = value[..., :8], value[..., 8:]
    l = np.concatenate(
        [left[..., :3],
         quat_to_rotate6d(left[..., 3:7], scalar_first=True),
         1.0 - left[..., 7:8] if invert_gripper else left[..., 7:8]], -1
    )
    r = np.concatenate(
        [right[..., :3],
         quat_to_rotate6d(right[..., 3:7], scalar_first=True),
         1.0 - right[..., 7:8] if invert_gripper else right[..., 7:8]], -1
    )
    return np.concatenate([l, r], -1).astype(np.float32)


def xvla20_to_ee16(value: np.ndarray, *, invert_gripper: bool = False, clip_gripper: bool = True) -> np.ndarray:
    """20 维 X-VLA 动作/状态 -> 16 维 end-effector 布局（评估指标用）。

    每臂 [xyz, rot6d, g] -> [xyz, quat_wxyz, g]；gripper 默认保留 20 维值并 clip 到 [0,1]，
    与先前评估（eval_results/*/metrics.json 的 convert_20d_to_16d=true）语义一致。
    """
    value = np.asarray(value, dtype=np.float32)
    if value.shape[-1] != 20:
        raise ValueError(f"unsupported last dim {value.shape[-1]}; expected 20")
    left_gripper = value[..., 9:10]
    right_gripper = value[..., 19:20]
    if clip_gripper:
        left_gripper = np.clip(left_gripper, 0.0, 1.0)
        right_gripper = np.clip(right_gripper, 0.0, 1.0)
    if invert_gripper:
        left_gripper = 1.0 - left_gripper
        right_gripper = 1.0 - right_gripper
    return np.concatenate(
        (
            value[..., 0:3],
            rotate6d_to_quat(value[..., 3:9], scalar_first=True),
            left_gripper,
            value[..., 10:13],
            rotate6d_to_quat(value[..., 13:19], scalar_first=True),
            right_gripper,
        ),
        axis=-1,
    ).astype(np.float32)


def action_slice(abs_traj: torch.Tensor, 
                 idx_for_delta: Sequence[int] = (),
                 idx_for_mask_proprio: Sequence[int] = ()
                ) -> Dict[str, torch.Tensor]:
    if not isinstance(abs_traj, torch.Tensor):
        raise TypeError("abs_traj must be a torch.Tensor")
    if abs_traj.ndim != 2 or abs_traj.size(0) < 2:
        raise ValueError("abs_traj must be [H+1, D] with H>=1")

    proprio = abs_traj[0]         # [D]
    action = abs_traj[1:].clone() # [H, D]

    if idx_for_delta:
        idx = torch.as_tensor(idx_for_delta, dtype=torch.long, device=abs_traj.device)
        action[:, idx] -= proprio[idx]
    if idx_for_mask_proprio:
        idx = torch.as_tensor(idx_for_mask_proprio, dtype=torch.long, device=abs_traj.device)
        proprio[idx] = 0.0
    return {"proprio": proprio, "action": action}