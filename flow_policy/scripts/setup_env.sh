#!/usr/bin/env bash
# =============================================================================
# flow_policy 训练环境一键配置（Linux + NVIDIA GPU 服务器）
#
# 用法:
#     bash flow_policy/scripts/setup_env.sh              # 默认参数
#     CUDA_TAG=cu126 bash flow_policy/scripts/setup_env.sh  # 指定 CUDA wheel
#
# 可配置环境变量（默认值见下）:
#     ENV_NAME        conda 环境名（默认 goai_flow）
#     PYTHON_VERSION  Python 版本（默认 3.11）
#     CUDA_TAG        PyTorch wheel 的 CUDA 标签（默认 cu128）
#     PIN_TORCH       固定 torch 版本（默认装 index 内最新，如设 2.8.0 则装 2.8.0）
#
# CUDA_TAG 选择参考（取决于 NVIDIA 驱动版本，nvidia-smi 可查）:
#     cu118  -> 老驱动（>= 450.80.02）          torch 2.2.x，兼容 RTX3090/T4/V100 等
#     cu124  -> 驱动 >= 545.23.07               torch 2.5.x，Ampere+/Hopper 主流
#     cu126  -> 驱动 >= 560.x                   torch 2.6.x
#     cu128  -> 驱动 >= 570.x（默认）           torch 2.8.x，H100/B100 等最新卡
#     驱动很新时也可不设 index，直接 pip install torch torchvision（PyPI 自带 CUDA）。
# =============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-goai_flow}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_TAG="${CUDA_TAG:-cu128}"
PIN_TORCH="${PIN_TORCH:-}"
CONDA_HOME_DEF="${CONDA_HOME_DEF:-$HOME/miniconda3}"

echo "==> 目标: env=$ENV_NAME python=$PYTHON_VERSION torch-index=$CUDA_TAG"

# 0. 探测 GPU 与驱动（仅提示，不阻塞）
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "==> GPU 检测:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/    /'
else
    echo "!! 未检测到 nvidia-smi。训练需要 NVIDIA GPU；若服务器有卡请先安装驱动/加入 PATH。"
fi

# 1. conda（缺则装 Miniconda 到 $HOME/miniconda3）
if ! command -v conda >/dev/null 2>&1 && [ ! -x "$CONDA_HOME_DEF/bin/conda" ]; then
    echo "==> 未检测到 conda，安装 Miniconda -> $CONDA_HOME_DEF ..."
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_HOME_DEF"
fi
if ! command -v conda >/dev/null 2>&1 && [ -x "$CONDA_HOME_DEF/bin/conda" ]; then
    export PATH="$CONDA_HOME_DEF/bin:$PATH"
fi
command -v conda >/dev/null 2>&1 || { echo "!! conda 仍不可用，请手动安装 Miniconda 后重试"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
echo "==> conda: $(conda --version)"

# 2. 创建环境（幂等，已存在则跳过）
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "==> 环境 $ENV_NAME 已存在，跳过创建（直接复用）"
else
    echo "==> 创建环境 $ENV_NAME (python=$PYTHON_VERSION) ..."
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi
conda activate "$ENV_NAME"

# 3. 安装依赖
echo "==> 安装 PyTorch ($CUDA_TAG) ..."
pip install -q --upgrade pip
INDEX="https://download.pytorch.org/whl/$CUDA_TAG"
if [ -n "$PIN_TORCH" ]; then
    pip install -q "torch==$PIN_TORCH" torchvision --index-url "$INDEX"
else
    pip install -q torch torchvision --index-url "$INDEX"
fi

echo "==> 安装其余训练依赖 ..."
pip install -q \
    "numpy<2" einops pyyaml pillow scipy \
    av pyarrow h5py mmengine accelerate tensorboard

# 4. 验证（在 flow_policy/ 下 import 三方包）
echo "==> 验证安装 ..."
cd "$(dirname "$0")/.."
python - <<'PY'
import sys
import torch
print(f"python       : {sys.version.split()[0]}")
print(f"torch        : {torch.__version__}")
print(f"cuda avail   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu          : {torch.cuda.get_device_name(0)}")
import torchvision, numpy, scipy, yaml, PIL, av, pyarrow, h5py, mmengine, accelerate
print(f"torchvision  : {torchvision.__version__}")
print(f"numpy        : {numpy.__version__} | scipy {scipy.__version__} | PIL {PIL.__version__}")
print(f"av           : {av.__version__} | pyarrow {pyarrow.__version__} | h5py {h5py.__version__}")
print(f"mmengine     : {mmengine.__version__} | accelerate {accelerate.__version__}")
PY

echo
echo "=================================================="
echo "完成。每次训练前:"
echo "  conda activate $ENV_NAME"
echo "  cd flow_policy"
echo "  accelerate launch --mixed_precision=bf16 train.py --config configs/train.yaml"
echo "=================================================="
