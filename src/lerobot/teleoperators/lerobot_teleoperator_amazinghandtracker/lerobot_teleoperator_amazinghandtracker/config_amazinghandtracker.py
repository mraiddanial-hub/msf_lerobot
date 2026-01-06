#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass ,field

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras import CameraConfig


@TeleoperatorConfig.register_subclass("lerobot_teleoperator_amazinghandtracker")
@dataclass
class AmazingHandTrackerConfig(TeleoperatorConfig):
    camera_id: int = field(default=0, metadata={"description": "Webcam device index"})
    side: str = field(default="right", metadata={"description": "Right or Left hand"})
    show_preview: bool = field(default=True, metadata={"description": "Show OpenCV preview window (may freeze on Windows with threading)"})
    show_overlay: bool = field(default=True, metadata={"description": "Draw hand landmarks on frames"})
    save_debug_frames: bool = field(default=False, metadata={"description": "Save frames with landmarks to disk for debugging (check output folder)"})
    debug_frame_interval: int = field(default=30, metadata={"description": "Save every Nth frame when save_debug_frames is enabled"})
    use_ik: bool = field(default=True, metadata={"description": "Use MuJoCo IK solver (requires mujoco+mink). Falls back to approximation if False."})
    mujoco_model_path: str | None = field(default=None, metadata={"description": "Path to MuJoCo scene.xml (auto-detects local copy if None)"})
    ik_dt: float = field(default=0.001, metadata={"description": "IK integration timestep in seconds (smaller = more accurate)"})
    ik_iterations: int = field(default=10, metadata={"description": "IK solve iterations per frame (more = better convergence)"})
    range_expansion: float = field(default=1.4, metadata={"description": "Expand calibrated range by this factor to reduce saturation"})
    def __post_init__(self):
            if self.side not in ["right", "left"]:
                raise ValueError(self.side)
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_1": OpenCVCameraConfig(
                index_or_path=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )



