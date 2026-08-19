# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict

from gr00t.data.dataset import ModalityConfig
from gr00t.eval.service import BaseInferenceClient, BaseInferenceServer
from gr00t.model.policy import BasePolicy


def _cast_obs_float64_to_float32(value: Any) -> Any:
    """Normalize eval observations so policy transforms see consistent float32 tensors."""
    import numpy as np
    import torch

    if isinstance(value, dict):
        return {k: _cast_obs_float64_to_float32(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cast_obs_float64_to_float32(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cast_obs_float64_to_float32(v) for v in value)
    if isinstance(value, np.ndarray) and value.dtype == np.float64:
        return value.astype(np.float32, copy=False)
    if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
        return value.to(dtype=torch.float32)
    return value


class RobotInferenceServer(BaseInferenceServer):
    """
    Server with three endpoints for real robot policies
    """

    def __init__(self, model, host: str = "*", port: int = 5555, api_token: str = None):
        super().__init__(host, port, api_token)
        def _wrapped_get_action(obs):
            obs = _cast_obs_float64_to_float32(obs)
            return model.get_action(obs)

        self.register_endpoint("get_action", _wrapped_get_action)
        self.register_endpoint(
            "get_modality_config", model.get_modality_config, requires_input=False
        )

    @staticmethod
    def start_server(policy: BasePolicy, port: int, api_token: str = None):
        server = RobotInferenceServer(policy, port=port, api_token=api_token)
        server.run()


class RobotInferenceClient(BaseInferenceClient, BasePolicy):
    """
    Client for communicating with the RealRobotServer
    """

    def __init__(self, host: str = "localhost", port: int = 5555, api_token: str = None):
        super().__init__(host=host, port=port, api_token=api_token)

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        return self.call_endpoint("get_action", observations)

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        return self.call_endpoint("get_modality_config", requires_input=False)
