
#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""
Lightweight registry of available devices and dataset tooling for the data-collection
focused LeRobot package.
"""

from lerobot.__version__ import __version__  # noqa: F401

# TODO(rcadene): Improve policies and envs. As of now, an item in `available_policies`
# refers to a yaml file AND a modeling name. Same for `available_envs` which refers to
# a yaml file AND a environment name. The difference should be more obvious.
available_tasks_per_env = {}
available_envs: list[str] = []
available_datasets_per_env: dict[str, list[str]] = {}
available_real_world_datasets: list[str] = []
available_datasets: list[str] = []

# Policies are not included in the data-collection subset.
available_policies: list[str] = []

# lists all available robots from `lerobot/robots`
available_robots = [
    "bi_arx5",
    "flexiv_rizon4",
    "xense_flare",
]

# lists all available cameras from `lerobot/cameras`
available_cameras = [
    "opencv",
    "intelrealsense",
    "xense",
]

# lists all available motors from `lerobot/motors`
available_motors = [
    "dynamixel",
    "feetech",
]

available_policies_per_env: dict[str, list[str]] = {}
