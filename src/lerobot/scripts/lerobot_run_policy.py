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
"""Run a trained policy on a robot continuously without recording.

This script allows you to deploy a trained policy on a robot for continuous execution,
useful for demonstrations, testing, or production deployment without the overhead of
recording episodes to a dataset.

Example usage:

```shell
lerobot-run-policy \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --policy.path=outputs/train/act_so100_test/checkpoints/last/pretrained_model \
    --fps=30
```

Example with AmazingHand robot:
```shell
lerobot-run-policy \
    --robot.type=lerobot_robot_amazinghand \
    --robot.port=COM5 \
    --robot.side=left \
    --robot.baudrate=1000000 \
    --robot.cameras="{cam_0: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --policy.path=outputs/train/act/checkpoints/last/pretrained_model \
    --fps=30
```

Press Ctrl+C to stop execution.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import importlib

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import RobotConfig, make_robot_from_config, lerobot_robot_amazinghand  # noqa: F401
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# Import config module to register amazinghand with draccus
try:
    importlib.import_module(
        "lerobot.robots.lerobot_robot_amazinghand.lerobot_robot_amazinghand.config_amazinghand"
    )
except Exception:
    pass


@dataclass
class RunPolicyConfig:
    """Configuration for running a policy on a robot."""
    
    robot: RobotConfig
    # Policy to run on the robot
    policy: PreTrainedConfig | None = None
    # Control loop frequency in Hz
    fps: int = 30
    # Task description (optional, used by some policies like VLA)
    task: str | None = None
    # Display extra information during execution
    display_data: bool = False
    
    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        
        if self.policy is None:
            raise ValueError("A policy is required to run this command")
    
    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


@parser.wrap()
def run_policy(cfg: RunPolicyConfig):
    """Run a trained policy continuously on a robot without recording."""
    init_logging()
    
    policy_path = Path(cfg.policy.pretrained_path)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy path not found: {policy_path}")
    
    logging.info(f"Loading policy from: {policy_path}")
    
    # Load training config to get dataset info
    train_config_path = policy_path / "train_config.json"
    if not train_config_path.exists():
        raise FileNotFoundError(f"Training config not found: {train_config_path}")
    
    with open(train_config_path) as f:
        train_config = json.load(f)
    
    # Load dataset metadata
    dataset_repo_id = train_config["dataset"]["repo_id"]
    logging.info(f"Loading dataset metadata from: {dataset_repo_id}")
    dataset = LeRobotDataset(dataset_repo_id)
    
    logging.info(f"Policy type: {cfg.policy.type}")
    logging.info(f"Control frequency: {cfg.fps} Hz")
    
    if cfg.display_data:
        init_rerun(session_name="run_policy")
    
    # Initialize robot
    logging.info(f"Connecting to {cfg.robot.type} robot")
    robot_instance = make_robot_from_config(cfg.robot)
    robot_instance.connect()
    
    # Load policy with dataset metadata
    policy_model = make_policy(cfg.policy, ds_meta=dataset.meta)
    policy_model.eval()
    
    # Create preprocessor and postprocessor pipelines
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, rename_map={}),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
        },
    )
    
    logging.info("Starting policy execution. Press Ctrl+C to stop.")
    
    dt = 1 / cfg.fps
    iteration = 0
    device = get_safe_torch_device(cfg.policy.device)
    start_time_total = time.perf_counter()
    
    try:
        while True:
            start_time = time.perf_counter()
            
            # Get observation from robot
            observation = robot_instance.get_observation()
            
            # Convert to dataset frame format
            observation_frame = build_dataset_frame(dataset.features, observation, prefix=OBS_STR)
            
            # Get action from policy using the full pipeline
            action_values = predict_action(
                observation=observation_frame,
                policy=policy_model,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=cfg.policy.use_amp,
                task=cfg.task,
                robot_type=cfg.robot.type,
            )
            
            # Convert action tensor to robot action dictionary
            action = make_robot_action(action_values, dataset.features)
            
            # Send action to robot
            robot_instance.send_action(action)
            
            # Log to Rerun if enabled
            if cfg.display_data:
                log_rerun_data(observation=observation, robot_action=action)
            
            iteration += 1
            
            # Log progress periodically
            if iteration % 100 == 0:
                elapsed_total = time.perf_counter() - start_time_total
                logging.info(f"Iteration {iteration}: Policy running (avg freq: {iteration / elapsed_total:.1f} Hz)")
            
            # Maintain control loop frequency
            elapsed = time.perf_counter() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    except KeyboardInterrupt:
        logging.info("Policy execution stopped by user")
    
    finally:
        robot_instance.disconnect()
        logging.info("Robot disconnected")


def main():
    run_policy()


if __name__ == "__main__":
    main()
