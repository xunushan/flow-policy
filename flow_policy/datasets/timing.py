# ------------------------------------------------------------------------------
# DataLoader worker 侧计时累计：统计视频解码耗时占比。
#
# 训练主进程在开启时设 env XVLA_TIMING_DIR=<dir>（仅冒烟测试/瓶颈量化用，生产默认关闭）。
# worker（DataLoader 子进程）每累积 _FLUSH_EVERY 个样本 flush 一次到
# <dir>/decode_<pid>.jsonl（各 worker 独立文件，无需锁）。训练结束时主进程聚合，
# 得到：
#   - decode_ms / 样本：每个训练样本平摊的视频解码毫秒数
#   - decode_pct：解码时间占 worker 侧处理墙钟时间比例（近似"数据预处理中视频解码占比"）
# 未设 env 时全部 no-op（每次解码一次 env 查询，开销可忽略）。
# ------------------------------------------------------------------------------
from __future__ import annotations

import json
import os
import time

_FLUSH_EVERY = 64  # 每累计这么多样本 flush 一次（减少 IO）

_state = {"decode_s": 0.0, "frames": 0, "samples": 0, "wall0": None}


def record_decode(dt: float, frames: int) -> None:
    """记录一次 episode 视频段解码耗时（秒）与解码帧数（进程内累计）。"""
    _state["decode_s"] += dt
    _state["frames"] += frames


def record_sample() -> None:
    """标记产出一个训练样本；每 _FLUSH_EVERY 个样本 flush 一次。"""
    if _state["wall0"] is None:
        _state["wall0"] = time.time()
    _state["samples"] += 1
    if _state["samples"] % _FLUSH_EVERY == 0:
        flush()


def flush() -> None:
    """把当前累计状态追加写入 <XVLA_TIMING_DIR>/decode_<pid>.jsonl（未开启时 no-op）。"""
    d = os.environ.get("XVLA_TIMING_DIR")
    if not d:
        return
    if _state["samples"] == 0 and _state["decode_s"] == 0:
        return
    os.makedirs(d, exist_ok=True)
    wall = time.time() - _state["wall0"] if _state["wall0"] else 0.0
    with open(os.path.join(d, f"decode_{os.getpid()}.jsonl"), "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "samples": _state["samples"],
                    "decode_s": round(_state["decode_s"], 4),
                    "frames": _state["frames"],
                    "wall_s": round(wall, 4),
                }
            )
            + "\n"
        )
