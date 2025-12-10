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

import logging
import time
from functools import cached_property
from typing import Any
from pathlib import Path

import numpy as np
import mujoco
import mink

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode, MotorCalibration
from lerobot.motors.calibration_gui import RangeFinderGUI
from lerobot.motors.feetech import (
    FeetechMotorsBus,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.robots.robot import Robot
from .config_amazinghand import AmazingHandConfig
from lerobot.robots.config import RobotConfig

logger = logging.getLogger(__name__)

class AmazingHand(Robot):
    config_class = AmazingHandConfig
    name = "lerobot_robot_amazinghand"

    def __init__(self, config: AmazingHandConfig):
        super().__init__(config)
        self.config = config
        import os
        
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                # Index
                "index_m1": Motor(15, "scs0009", MotorNormMode.RANGE_0_100),
                "index_m2": Motor(16, "scs0009", MotorNormMode.RANGE_0_100),
                # Middle
                "middle_m1": Motor(13, "scs0009", MotorNormMode.RANGE_0_100),
                "middle_m2": Motor(14, "scs0009", MotorNormMode.RANGE_0_100),
                # Ring
                "ring_m1": Motor(11, "scs0009", MotorNormMode.RANGE_0_100),
                "ring_m2": Motor(12, "scs0009", MotorNormMode.RANGE_0_100),
                # Thumb
                "thumb_m1": Motor(17, "scs0009", MotorNormMode.RANGE_0_100),
                "thumb_m2": Motor(18, "scs0009", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
            protocol_version=1,
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        # Demo has NO motor inversions - all motors are invert=false
        self.inverted_motors = []
        
        # Initialize MuJoCo simulation + IK solver
        self._setup_simulation()
        logger.info("Robot smoothing: DISABLED (raw IK output sent directly)")
        
        # MuJoCo viewer for visualization (optional, controlled by env var)
        self.viewer = None
        self._viewer_thread = None
        self._viewer_stop = False
        self._viewer_enabled = os.getenv("LEROBOT_SHOW_MUJOCO_VIEWER", "0").lower() in ("1", "true", "yes")

        # Optional per-finger axis sign flips before IK (for quick field tuning)
        # Format: LEROBOT_TIP_AXIS_FLIPS="index:x:-1;middle:x:-1;thumb:y:-1" (axis in {x,y,z}, sign in {1,-1})
        self.axis_flips: dict[str, np.ndarray] = {}
        flips_env = os.getenv("LEROBOT_TIP_AXIS_FLIPS")
        if flips_env:
            try:
                for entry in flips_env.split(";"):
                    entry = entry.strip()
                    if not entry:
                        continue
                    # Allow compact form finger:axis:sign or finger:x:-1,y:1,z:1
                    if "," in entry:
                        # finger:axis:sign,axis:sign,axis:sign
                        finger, rest = entry.split(":", 1)
                        signs = {"x":1, "y":1, "z":1}
                        for part in rest.split(","):
                            a,s = part.split(":")
                            signs[a.strip()] = int(s.strip())
                    else:
                        finger, axis, sign = entry.split(":")
                        signs = {"x":1, "y":1, "z":1}
                        signs[axis.strip()] = int(sign.strip())
                    self.axis_flips[finger.strip()] = np.array([signs["x"], signs["y"], signs["z"]], dtype=float)
                logger.info(f"Axis flips configured: {self.axis_flips}")
            except Exception as e:
                logger.warning(f"Failed to parse LEROBOT_TIP_AXIS_FLIPS='{flips_env}': {e}")

        # Optional per-finger axis permutation (map hand-frame components to robot frame order)
        # Format: LEROBOT_TIP_AXIS_PERM="index=2,1,0;middle=0,2,1" where values are indices for x,y,z
        self.axis_perm: dict[str, np.ndarray] = {}
        perm_env = os.getenv("LEROBOT_TIP_AXIS_PERM")
        if perm_env:
            try:
                for entry in perm_env.split(";"):
                    entry = entry.strip()
                    if not entry:
                        continue
                    finger, perm = entry.split("=")
                    idxs = [int(x.strip()) for x in perm.split(",")]
                    if len(idxs) != 3 or any(i not in (0,1,2) for i in idxs):
                        raise ValueError("Permutation must be three indices 0,1,2")
                    self.axis_perm[finger.strip()] = np.array(idxs, dtype=int)
                logger.info(f"Axis permutations configured: {self.axis_perm}")
            except Exception as e:
                logger.warning(f"Failed to parse LEROBOT_TIP_AXIS_PERM='{perm_env}': {e}")

        # Optional per-finger motor swap (swap m1/m2 logical mapping) before sending goals
        swaps_env = os.getenv("LEROBOT_SWAP_FINGERS")  # e.g. "index,middle"
        self.swap_fingers: set[str] = set()
        if swaps_env:
            try:
                self.swap_fingers = {s.strip() for s in swaps_env.split(",") if s.strip()}
                if self.swap_fingers:
                    logger.info(f"Motor swap enabled for fingers: {sorted(self.swap_fingers)}")
            except Exception:
                pass
    
    def _setup_simulation(self):
        """Initialize MuJoCo simulation and mink IK solver with persistent state."""
        # Load MuJoCo model: prefer env override LEROBOT_MJCF_PATH if provided, else bundled MJCF
        import os
        env_mjcf = os.getenv("LEROBOT_MJCF_PATH")
        mjcf_path: Path
        if env_mjcf:
            p = Path(env_mjcf)
            if not p.exists():
                raise FileNotFoundError(f"LEROBOT_MJCF_PATH set but file not found: {p}")
            mjcf_path = p
            logger.info(f"Using MuJoCo model from LEROBOT_MJCF_PATH: {mjcf_path}")
        else:
            # Try both the original teleoperator location and new configs location
            robot_package_dir = Path(__file__).parent.parent.parent.parent
            # First try configs/robot/amazinghand/mjcf/scene.xml (copied from Demo)
            mjcf_path = robot_package_dir / "configs" / "robot" / "amazinghand" / "mjcf" / "scene.xml"
            if not mjcf_path.exists():
                # Fallback to teleoperator package
                mjcf_path = robot_package_dir / "teleoperators" / "lerobot_teleoperator_amazinghandtracker" / "lerobot_teleoperator_amazinghandtracker" / "mjcf" / "scene.xml"
            if not mjcf_path.exists():
                raise FileNotFoundError(f"MuJoCo model not found at {mjcf_path}")
            logger.info(f"Using bundled MuJoCo model: {mjcf_path}")
        
        self.mj_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.mj_data = mujoco.MjData(self.mj_model)
        self.configuration = mink.Configuration(self.mj_model)
        self.data = self.configuration.data
        
        # Initialize from zero keyframe for a stable starting pose
        try:
            self.configuration.update_from_keyframe("zero")
        except Exception:
            # Some models may not have a 'zero' keyframe; ignore if missing
            pass
        
        # Posture and equality constraint tasks to stabilize IK
        self.posture_task = mink.PostureTask(self.mj_model, cost=1e-2)
        self.posture_task.set_target_from_configuration(self.configuration)
        self.eq_task = mink.EqualityConstraintTask(self.mj_model, cost=1000.0)
        
        # Create persistent IK tasks for each fingertip (position-only control)
        self.tip_tasks = {}
        for finger_idx in [1, 2, 3, 4]:
            site_name = f"tip{finger_idx}"
            site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id >= 0:
                task = mink.FrameTask(
                    frame_name=site_name,
                    frame_type="site",
                    position_cost=1.0,
                    orientation_cost=0.0,
                    lm_damping=1.0,
                )
                self.tip_tasks[finger_idx] = task
        
        # Initialize mocap bodies at their respective sites (Demo pattern)
        for finger_idx in [1, 2, 3, 4]:
            mocap_body = f"finger{finger_idx}_target"
            site_name = f"tip{finger_idx}"
            try:
                mink.move_mocap_to_frame(self.mj_model, self.data, mocap_body, site_name, "site")
            except Exception as e:
                logger.warning(f"Could not initialize mocap {mocap_body} from {site_name}: {e}")
        
        # Per-finger scaling factors and offsets from original mj_mink_left.py
        # Map logical finger names to MJCF indices (finger1..finger4)
        # Increased scale for better range of motion in teleoperation
        # Can be overridden with LEROBOT_TIP_SCALE env var
        self.tip_scale = float(os.getenv("LEROBOT_TIP_SCALE", "2.5"))
        # All fingers use same scale for consistency with Demo
        self.tip_scales = {1: self.tip_scale, 2: self.tip_scale, 3: self.tip_scale, 4: self.tip_scale}
        logger.info(f"Using tip vector scale factor: {self.tip_scale} (set LEROBOT_TIP_SCALE to adjust)")
        self.finger_name_to_index = {"index": 1, "middle": 2, "ring": 3, "thumb": 4}
        self.tip_offsets = {
            1: np.array([0.025, -0.022, 0.098]),  # index
            2: np.array([0.025, 0.009, 0.092]),   # middle
            3: np.array([0.025, 0.040, 0.082]),  # ring
            4: np.array([0.024, -0.019, 0.017]),  # thumb
        }
        
        # Per-motor angular offsets (applied post-IK, in radians)
        # These are CRITICAL - they center the robot's neutral position
        # Can be loaded from TOML file via LEROBOT_MOTOR_OFFSETS_PATH env var
        self.motor_offsets = self._load_motor_offsets()
        logger.info("=" * 60)
        logger.info("MOTOR OFFSETS (CRITICAL FOR ACCURACY):")
        for motor_name in sorted(self.motor_offsets.keys()):
            offset = self.motor_offsets[motor_name]
            logger.info(f"  {motor_name}: {offset:+.4f} rad ({offset * 180/np.pi:+.2f}°)")
        logger.info("=" * 60)   
        
        # Simulation configuration parameters (matching Demo's rate limiter)
        self.sim_dt = self.mj_model.opt.timestep  # Use model timestep (typically 0.001)
        self.ik_solver = "quadprog"
        self.ik_tolerance = 1e-5
        
        # Optional angle scaling for more range (EXPERIMENTAL - use carefully)
        self._angle_scale = float(os.getenv("LEROBOT_ANGLE_SCALE", "1.05"))
        if self._angle_scale != 1.0:
            logger.warning(f"⚠️  ANGLE_SCALE={self._angle_scale} (EXPERIMENTAL - may strain servos)")
        
        logger.info(f"Simulation initialized with dt={self.sim_dt}, solver={self.ik_solver}")
    
    def _load_motor_offsets(self) -> dict[str, float]:
        """Load motor offsets from TOML file or use defaults.
        
        Checks in order:
        1. LEROBOT_MOTOR_OFFSETS_PATH env var (absolute path)
        2. config/motor_offsets.toml (next to this file)
        3. Default values from Demo
        
        Expected TOML format:
        [[motors]]
        finger_name = "l_finger1"
        [motors.motor1]
        offset = 0.031
        [motors.motor2]
        offset = 0.173
        """
        import os
        from pathlib import Path
        
        # Check env var first
        toml_path_str = os.getenv("LEROBOT_MOTOR_OFFSETS_PATH")
        
        if toml_path_str:
            toml_path = Path(toml_path_str)
        else:
            # Check standard location: config/l_hand.toml next to this file (Demo format)
            toml_path = Path(__file__).parent / "config" / "l_hand.toml"
            if not toml_path.exists():
                # Fallback to motor_offsets.toml if custom file exists
                toml_path = Path(__file__).parent / "config" / "motor_offsets.toml"
                if not toml_path.exists():
                    toml_path = None
        
        if toml_path:
            try:
                import tomli
                if not toml_path.exists():
                    logger.warning(f"Motor offsets TOML not found: {toml_path}, using defaults")
                else:
                    with open(toml_path, "rb") as f:
                        config = tomli.load(f)
                    
                    # Parse TOML structure
                    offsets = {}
                    finger_map = {
                        "l_finger1": ("index_m1", "index_m2"),
                        "l_finger2": ("middle_m1", "middle_m2"),
                        "l_finger3": ("ring_m1", "ring_m2"),
                        "l_finger4": ("thumb_m1", "thumb_m2"),
                    }
                    
                    for motor_entry in config.get("motors", []):
                        finger_name = motor_entry.get("finger_name")
                        if finger_name in finger_map:
                            m1_name, m2_name = finger_map[finger_name]
                            
                            motor1 = motor_entry.get("motor1", {})
                            motor2 = motor_entry.get("motor2", {})
                            
                            if "offset" in motor1:
                                offsets[m1_name] = float(motor1["offset"])
                            if "offset" in motor2:
                                offsets[m2_name] = float(motor2["offset"])
                    
                    logger.info(f"Loaded motor offsets from {toml_path}: {offsets}")
                    return offsets
                    
            except ImportError:
                logger.warning("tomli not installed, using default offsets. Install with: pip install tomli")
            except Exception as e:
                logger.warning(f"Failed to load motor offsets from {toml_path}: {e}, using defaults")
        
        # Default offsets from Demo's l_hand.toml
        defaults = {
            "index_m1": 0.0311,      # finger1 motor1
            "index_m2": 0.1731,      # finger1 motor2
            "middle_m1": -0.6009,    # finger2 motor1
            "middle_m2": 0.6137,     # finger2 motor2
            "ring_m1": -0.9901,      # finger3 motor1
            "ring_m2": 0.9146,       # finger3 motor2
            "thumb_m1": -0.1009,     # finger4 motor1
            "thumb_m2": 0.5113,      # finger4 motor2
        }
        logger.info(f"Using default motor offsets from Demo config")
        return defaults
    
    def _run_viewer(self):
        """Run MuJoCo viewer in separate thread (for older MuJoCo API)."""
        try:
            # glfw-based viewer requires running in main thread on some platforms,
            # but we'll try in background thread for convenience
            import glfw
            if not glfw.init():
                logger.error(f"{self}: Failed to initialize GLFW for viewer")
                return
            
            # Create window
            window = glfw.create_window(1200, 900, "MuJoCo - AmazingHand IK Simulation", None, None)
            if not window:
                logger.error(f"{self}: Failed to create GLFW window")
                glfw.terminate()
                return
            
            glfw.make_context_current(window)
            glfw.swap_interval(1)
            
            # Create MuJoCo rendering context
            import mujoco
            context = mujoco.MjrContext(self.mj_model, mujoco.mjtFontScale.mjFONTSCALE_150)
            scene = mujoco.MjvScene(self.mj_model, maxgeom=10000)
            camera = mujoco.MjvCamera()
            option = mujoco.MjvOption()
            
            # Set camera for better view
            camera.azimuth = 90
            camera.elevation = -20
            camera.distance = 0.5
            camera.lookat[:] = [0.025, 0.01, 0.06]
            
            viewport = mujoco.MjrRect(0, 0, 1200, 900)
            
            while not self._viewer_stop and not glfw.window_should_close(window):
                # Update scene
                mujoco.mjv_updateScene(
                    self.mj_model, self.data, option, None, camera,
                    mujoco.mjtCatBit.mjCAT_ALL, scene
                )
                
                # Render
                mujoco.mjr_render(viewport, scene, context)
                
                glfw.swap_buffers(window)
                glfw.poll_events()
                
                time.sleep(0.01)  # ~100 FPS max
            
            glfw.terminate()
            logger.info(f"{self}: Viewer window closed.")
            
        except Exception as e:
            logger.error(f"{self}: Error in viewer thread: {e}", exc_info=True)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        # Record motor angles (what we actually control)
        # Tip vectors from teleoperator are converted to motor angles via IK
        return {f"{motor}.pos": float for motor in self.bus.motors.keys()}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        # Connect the cameras
        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        
        # Launch MuJoCo viewer if enabled
        if self._viewer_enabled:
            logger.info(f"{self}: Launching MuJoCo viewer...")
            try:
                # Try new API (mujoco >= 3.0)
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.data)
                logger.info(f"{self}: MuJoCo viewer launched (new API). You can see the simulation in real-time!")
            except (AttributeError, ImportError) as e:
                # Fallback to older API (mujoco < 3.0) - passive viewer using context manager pattern
                logger.info(f"{self}: New viewer API not available ({e}), trying older API with threading...")
                try:
                    import threading
                    self._viewer_thread = threading.Thread(target=self._run_viewer, daemon=True)
                    self._viewer_thread.start()
                    logger.info(f"{self}: MuJoCo viewer thread started. Window should appear shortly.")
                except Exception as viewer_err:
                    logger.warning(f"{self}: Could not start viewer: {viewer_err}. Continuing without visualization.")
                    logger.info(f"{self}: You can still see IK results in the logs and on the physical robot.")
        
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        logger.info("🔧 Starting calibration for teleoperation...")
        logger.info("Move each finger through its full range of motion.")
        
        fingers = {}
        for finger in ["thumb", "index", "middle", "ring"]:
            fingers[finger] = [motor for motor in self.bus.motors if motor.startswith(finger)]

        self.calibration = RangeFinderGUI(self.bus, fingers).run()
        
        # Set homing_offset to 0 for all motors (Protocol 1 doesn't support homing offset)
        # Demo uses motor offsets applied to joint angles, not homing_offset
        for motor in self.calibration:
            self.calibration[motor].homing_offset = 0
        
        # Demo has ALL motors with invert=false (drive_mode=0)
        # Set all motors to normal direction
        for motor in self.calibration:
            self.calibration[motor].drive_mode = 0
        
        self._save_calibration()
        logger.info(f"✅ Calibration saved to {self.calibration_fpath}")
        logger.info("Motor ranges recorded. These will be used for teleoperation.")

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            
            # Set speed and acceleration for faster, more responsive movement
            # Maximum_Acceleration: 0-254, higher = faster acceleration (default ~50)
            # Goal_Speed: 0-2047, higher = faster movement (default varies, typical ~100-300)
            import os
            max_accel = int(os.getenv("LEROBOT_MAX_ACCELERATION", "254"))  # Max acceleration
            goal_speed = int(os.getenv("LEROBOT_GOAL_SPEED", "2047"))      # Maximum speed for fastest response
            
            logger.info(f"Setting motor speed parameters: Max_Accel={max_accel}, Goal_Speed={goal_speed}")
            
            for motor_name in self.bus.motors:
                try:
                    # Set maximum acceleration for faster response
                    self.bus.write("Maximum_Acceleration", motor_name, max_accel)
                    
                    # Set goal speed for faster movement
                    self.bus.write("Goal_Speed", motor_name, goal_speed)
                    
                    logger.info(f"  {motor_name}: Max_Accel={max_accel}, Goal_Speed={goal_speed}")
                except Exception as e:
                    logger.warning(f"  {motor_name}: Could not set speed parameters: {e}")
        
        # Explicitly enable torque after configuration
        logger.info("Enabling torque on all motors...")
        for motor_name in self.bus.motors:
            self.bus.write("Torque_Enable", motor_name, 1)
            torque_state = self.bus.read("Torque_Enable", motor_name)
            logger.info(f"  {motor_name}: Torque_Enable = {torque_state}")
        logger.info("✅ All motors torque enabled")

    def setup_motors(self) -> None:
        # TODO: add docstring
        for motor in self.bus.motors:
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        # Read hand position - skip motors with errors
        start = time.perf_counter()
        for motor in self.bus.motors:
            try:
                obs_dict[f"{motor}.pos"] = self.bus.read("Present_Position", motor)
            except RuntimeError as e:
                if "voltage error" in str(e).lower():
                    logger.warning(f"{self}: Motor {motor} has voltage error, skipping read. Check power supply!")
                    obs_dict[f"{motor}.pos"] = 0.0  # Default value
                else:
                    raise
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            frame = cam.async_read()
            # Flip camera horizontally for mirror view in Rerun
            if frame is not None:
                import cv2
                frame = cv2.flip(frame, 1)
            obs_dict[cam_key] = frame
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        import os

        # Check if action contains motor angles (replay mode) or tip vectors (teleop mode)
        motor_angles = {}
        for motor_name in self.bus.motors.keys():
            angle_key = f"{motor_name}.pos"
            if angle_key in action:
                motor_angles[motor_name] = float(action[angle_key])
        
        # If motor angles are present (replay mode), use them directly
        if motor_angles:
            # Convert motor angles (radians) to motor positions (0-100 percent)
            goal_pos = {}
            IK_TO_DELTA_SCALE = float(os.getenv("LEROBOT_IK_DELTA_SCALE", "2.0"))
            
            for motor_name, angle_rad in motor_angles.items():
                if motor_name in self.bus.motors:
                    # Get neutral position for YOUR hand assembly
                    neutral_offset = self.motor_offsets.get(motor_name, 0.0)
                    
                    # Treat recorded angle as DELTA from neutral
                    delta_from_neutral = angle_rad * IK_TO_DELTA_SCALE
                    
                    # Final angle = neutral + delta
                    final_angle = neutral_offset + delta_from_neutral
                    
                    # Radians to percent: [-π, π] → [0, 100]
                    motor_percent = ((final_angle + np.pi) / (2 * np.pi)) * 100.0
                    goal_pos[motor_name] = motor_percent
            
            if goal_pos:
                try:
                    self.bus.sync_write("Goal_Position", goal_pos, normalize=True)
                except Exception as e:
                    logger.error(f"sync_write failed: {e}")
            
            return action
        
        # Extract tip vectors from action (teleop mode - raw meters)
        tip_vectors = {}
        for finger in ["index", "middle", "ring", "thumb"]:
            x_key = f"{finger}_tip_x.pos"
            y_key = f"{finger}_tip_y.pos"
            z_key = f"{finger}_tip_z.pos"
            if x_key in action and y_key in action and z_key in action:
                tip_vectors[finger] = np.array([float(action[x_key]), float(action[y_key]), float(action[z_key])])
        
        if not tip_vectors:
            return action
        
        # Run IK solver to convert tip vectors to motor angles
        motor_angles = self._solve_ik(tip_vectors)
        if not motor_angles:
            return action
        
        # Convert motor angles (radians) to motor positions (0-100 percent)
        # KEY INSIGHT: Your hand assembly has different neutral position than MuJoCo model
        # - IK produces angles relative to MuJoCo's neutral (0 rad)
        # - Your offsets define YOUR hand's neutral position
        # - We want: IK angle = DELTA from neutral, not absolute angle
        # Solution: Send (neutral_offset + IK_delta) where IK is scaled to be a delta
        goal_pos = {}
        raw_percents = {}  # Track for logging
        
        # Scale factor to treat IK output as delta from neutral (not absolute)
        # IK typically outputs -1.5 to +1.5 rad range
        # We want this to be ±movement around neutral, not absolute position
        IK_TO_DELTA_SCALE = float(os.getenv("LEROBOT_IK_DELTA_SCALE", "2.0"))  # Increased for more range (was 1.5)
        
        for motor_name, angle_rad in motor_angles.items():
            if motor_name in self.bus.motors:
                # Get neutral position for YOUR hand assembly
                neutral_offset = self.motor_offsets.get(motor_name, 0.0)
                
                # Treat IK output as DELTA from neutral
                delta_from_neutral = angle_rad * IK_TO_DELTA_SCALE
                
                # Final angle = neutral + delta
                final_angle = neutral_offset + delta_from_neutral
                
                # Radians to percent: [-π, π] → [0, 100]
                motor_percent = ((final_angle + np.pi) / (2 * np.pi)) * 100.0
                raw_percents[motor_name] = motor_percent
                
                # NO EMA SMOOTHING - send directly
                goal_pos[motor_name] = motor_percent
        
        if goal_pos:
            try:
                self.bus.sync_write("Goal_Position", goal_pos, normalize=True)
            except Exception as e:
                logger.error(f"sync_write failed: {e}")
        

        
        # Replace tip vectors with motor angles for recording/visualization
        # This is what gets recorded during dataset collection
        # Remove tip vector keys (they were just intermediate data)
        keys_to_remove = [k for k in action.keys() if 'tip' in k]
        for k in keys_to_remove:
            del action[k]
        
        # Add motor angles (this is what we actually control and want to record)
        for motor_name, angle_rad in motor_angles.items():
            action[f"{motor_name}.pos"] = angle_rad
        
        return action
    
    def _update_mocap_targets(self, tip_vectors: dict[str, np.ndarray]):
        """
        Update mocap body positions from tracked tip vectors (Demo write_mocap_pos pattern).
        This pushes new target positions into the simulation without resetting state.
        """
        scaled_positions = {}
        for finger_name, tip_vec in tip_vectors.items():
            # Apply per-finger axis permutation if configured
            if finger_name in self.axis_perm:
                perm = self.axis_perm[finger_name]
                tip_vec = tip_vec[perm]
            # Apply per-finger axis flips if configured
            if finger_name in self.axis_flips:
                tip_vec = tip_vec * self.axis_flips[finger_name]
            if finger_name in self.finger_name_to_index:
                idx = self.finger_name_to_index[finger_name]
                scale = self.tip_scales.get(idx, self.tip_scale)
                scaled_pos = tip_vec * scale + self.tip_offsets[idx]
                scaled_positions[finger_name] = scaled_pos
                
                # Update mocap body position directly (as in mj_mink_left.py write_mocap_pos)
                mocap_body = f"finger{idx}_target"
                body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, mocap_body)
                if body_id >= 0:
                    mocap_id = self.mj_model.body_mocapid[body_id]
                    if mocap_id >= 0:
                        self.data.mocap_pos[mocap_id] = scaled_pos

    def _solve_ik(self, tip_vectors: dict[str, np.ndarray]) -> dict[str, float]:
        """
        Run one IK integration step to smoothly move configuration toward mocap targets.
        This is the core simulation loop from Demo: update mocap → set task targets → solve → integrate.
        
        Args:
            tip_vectors: Dict mapping finger names to 3D tip position vectors
            
        Returns:
            Dict mapping motor names to joint angles in radians
        """
        # Step 1: Update mocap body positions from new tip vectors
        self._update_mocap_targets(tip_vectors)
        
        # Step 2: Update each IK task target from its corresponding mocap body (Demo pattern)
        for finger_idx, task in self.tip_tasks.items():
            mocap_body = f"finger{finger_idx}_target"
            task.set_target(
                mink.SE3.from_mocap_name(self.mj_model, self.data, mocap_body)
            )
        
        # Step 3: Build task list with stabilizing tasks
        tasks = [self.eq_task, self.posture_task] + list(self.tip_tasks.values())
        
        # Step 4: Solve IK and integrate configuration (Demo uses single integration per tick)
        vel = mink.solve_ik(
            self.configuration,
            tasks,
            self.sim_dt,
            self.ik_solver,
            self.ik_tolerance
        )
        self.configuration.integrate_inplace(vel, self.sim_dt)
        
        # Update viewer if running (sync the visual to show current state)
        if self.viewer is not None:
            try:
                if hasattr(self.viewer, 'is_running') and self.viewer.is_running():
                    self.viewer.sync()
            except Exception:
                pass  # Viewer in separate thread handles its own updates
        
        # Step 5: Read joint positions from integrated configuration
        motor_angles = {}
        name_map = {
            1: ("index_m1", "index_m2"),
            2: ("middle_m1", "middle_m2"),
            3: ("ring_m1", "ring_m2"),
            4: ("thumb_m1", "thumb_m2"),
        }
        for finger_idx, (m1, m2) in name_map.items():
            # Optionally swap motors for this finger if requested
            finger_label = {1:"index",2:"middle",3:"ring",4:"thumb"}[finger_idx]
            if finger_label in self.swap_fingers:
                m1, m2 = m2, m1
            for motor_name, joint_suffix in ((m1, "motor1"), (m2, "motor2")):
                joint_name = f"finger{finger_idx}_{joint_suffix}"
                joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id >= 0:
                    # Read joint angle from IK-integrated configuration
                    angle = self.data.joint(joint_id).qpos[0]
                    
                    # Optional angle amplification for more range (EXPERIMENTAL)
                    if hasattr(self, '_angle_scale') and self._angle_scale != 1.0:
                        angle *= self._angle_scale
                    
                    motor_angles[motor_name] = angle
        
        return motor_angles

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Close MuJoCo viewer if running
        if self.viewer is not None:
            try:
                self.viewer.close()
                logger.info(f"{self}: MuJoCo viewer closed.")
            except Exception as e:
                logger.warning(f"{self}: Error closing MuJoCo viewer: {e}")
            self.viewer = None
        
        # Stop viewer thread if running
        if self._viewer_thread is not None:
            self._viewer_stop = True
            self._viewer_thread.join(timeout=2.0)
            logger.info(f"{self}: Viewer thread stopped.")
            self._viewer_thread = None

        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
