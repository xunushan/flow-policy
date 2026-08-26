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
from typing import Dict, Type
from .base import DomainHandler

# Handlers
from .lerobot_v3_robodojo import LeRobotV3RoboDojoHandler

# Exact registry only (no heuristics). flow_policy 只训练 ARX 双臂 v3.0 数据。
_REGISTRY: Dict[str, Type[DomainHandler]] = {
    "arx_x5_ee": LeRobotV3RoboDojoHandler,
}

def get_handler_cls(dataset_name: str) -> Type[DomainHandler]:
    """Strict lookup: require explicit registration."""
    try:
        return _REGISTRY[dataset_name]
    except KeyError:
        raise KeyError(
            f"No handler registered for dataset '{dataset_name}'. "
            f"Only 'arx_x5_ee' is registered in flow_policy."
        )
