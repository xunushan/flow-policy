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

# flow_policy 只训练单个 ARX 双臂数据集：无多数据集加权。
DATA_WEIGHTS = {}

# dataset.py 用 DATA_DOMAIN_ID.get(robot_type, 0) 注入 domain_id。
# 模型侧无 domain 概念，该字段会被训练循环 pop 掉；这里保留映射即可。
DATA_DOMAIN_ID = {
    "arx_x5_ee": 0,
}
