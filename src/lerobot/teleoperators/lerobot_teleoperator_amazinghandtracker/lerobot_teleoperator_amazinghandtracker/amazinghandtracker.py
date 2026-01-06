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
import os
import threading
from collections import deque
from pprint import pformat

import serial

from lerobot.motors import MotorCalibration
from lerobot.motors.motors_bus import MotorNormMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.utils import enter_pressed, move_cursor_up

from lerobot.teleoperators import Teleoperator
from .config_amazinghandtracker import AmazingHandTrackerConfig
from .inverse_kinematics import AmazingHandIK
logger = logging.getLogger(__name__)

#!/usr/bin/env python

import logging
import threading
from collections import deque
from pprint import pformat
import serial

from lerobot.motors import MotorCalibration
from lerobot.motors.motors_bus import MotorNormMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.utils import enter_pressed, move_cursor_up

from lerobot.teleoperators import Teleoperator
from .config_amazinghandtracker import AmazingHandTrackerConfig
from lerobot.teleoperators.config import TeleoperatorConfig

logger = logging.getLogger(__name__)

# Tip vector components for hand tracking (sent as raw values to robot)
HAND_JOINTS = [
    "index_tip_x", "index_tip_y", "index_tip_z",
    "middle_tip_x", "middle_tip_y", "middle_tip_z",
    "ring_tip_x", "ring_tip_y", "ring_tip_z",
    "thumb_tip_x", "thumb_tip_y", "thumb_tip_z",
]

class AmazingHandTracker(Teleoperator):
    config_class = AmazingHandTrackerConfig
    name = "lerobot_teleoperator_amazinghandtracker"

    def __init__(self, config: AmazingHandTrackerConfig):
        super().__init__(config)
        self.config = config
        # If a MuJoCo model path is provided in teleop config, propagate via env for the robot
        try:
            mjcf = getattr(config, "mujoco_model_path", None)
            if mjcf:
                from pathlib import Path
                p = Path(mjcf)
                if p.exists():
                    os.environ["LEROBOT_MJCF_PATH"] = str(p.resolve())
                    logger.info(f"{self}: Set LEROBOT_MJCF_PATH to {p}")
                else:
                    logger.warning(f"{self}: Provided mujoco_model_path does not exist: {p}")
        except Exception as e:
            logger.warning(f"{self}: Failed to set LEROBOT_MJCF_PATH: {e}")
        self._frame = None
        self._frame_ready = threading.Event()
        self._stop_event = threading.Event()
        self._state = None
        self._state_lock = threading.Lock()
        
        # Initialize camera
        from lerobot.cameras.utils import make_cameras_from_configs
        self.cameras = make_cameras_from_configs(config.cameras)

        # Initialize trackers for each finger tip component (raw metric values, no normalization)
        self.joints = {joint: float for joint in HAND_JOINTS}

        # NO EMA smoothing - send raw MediaPipe tracking data (like Demo does)
        # Demo only has PID smoothing in Rust controller, not in Python tracking
        logger.info("Tracker smoothing: DISABLED (raw MediaPipe data sent directly)")
        self._last_raw = dict.fromkeys(self.joints, None)  # Store last raw values for fallback

        # Start processing thread
        self._thread = threading.Thread(target=self._tracking_loop, daemon=True)

    # ------------------------------------------------------------
    # LeRobot interface properties
    # ------------------------------------------------------------
    @property
    def action_features(self) -> dict:
        return {f"{joint}.pos": float for joint in self.joints}
    
    @property
    def observation_features(self) -> dict:
        # Include camera dimensions for rerun visualization
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) 
            for cam in self.cameras
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return all(cam.is_connected for cam in self.cameras.values()) and self._thread.is_alive()

    # ------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------
    def connect(self, calibrate=True):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()
        
        # Start tracking thread
        self._thread.start()

        # Try to prime the camera/frame source and verify we can get frames
        try:
            first_cam = next(iter(self.cameras.values()))
            logger.info(f"{self}: Attempting to read first frame from camera...")
            
            # Try synchronous read first to ensure camera works
            frame = first_cam.read()
            if frame is not None:
                logger.info(f"{self}: Successfully got first frame with shape {frame.shape}")
            else:
                logger.warning(f"{self}: Got None frame from synchronous read")
            
            # Now try async read to prime the background thread
            frame = first_cam.async_read(timeout_ms=2000)
            if frame is not None:
                logger.info(f"{self}: Successfully got first async frame with shape {frame.shape}")
            else:
                logger.warning(f"{self}: Got None frame from async read")
                
        except Exception as e:
            logger.warning(f"{self}: Error priming camera: {e}")

        # Wait for first successful tracking with longer timeout
        logger.info(f"{self}: Waiting for first frame to be processed...")
        if not self._frame_ready.wait(timeout=10):  # Increased timeout
            logger.error(f"{self}: Timed out waiting for first frame to be processed")
            raise TimeoutError(f"{self}: Timed out waiting for first frame.")

        # Hand tracking doesn't need calibration - skip it
        if calibrate:
            logger.info(f"{self}: Calibration not required for hand tracking (sends raw metric vectors)")

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # Hand tracking doesn't need calibration - always return True
        return True
    
    def _load_calibration(self):
        """Override parent method - hand tracking doesn't need calibration."""
        # Check if a calibration file exists and warn that it will be ignored
        from pathlib import Path
        calib_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "teleoperators" / "lerobot_teleoperator_amazinghandtracker"
        calib_file = calib_dir / "None.json"
        
        if calib_file.exists():
            logger.warning(f"⚠️  Found calibration file at {calib_file}")
            logger.warning(f"   This file LIMITS tip vector range and should be deleted for teleoperation!")
            logger.warning(f"   Ignoring calibration - using RAW MediaPipe tracking data")
        
        # Set calibration to empty dict to satisfy interface
        self.calibration = {}
        logger.info(f"{self}: Skipping calibration load (not required for hand tracking)")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Stop tracking thread
        self._stop_event.set()
        self._thread.join(timeout=1)

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()
            
        logger.info(f"{self} disconnected.")

    # ------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------
    def calibrate(self):
        """Calibration is NOT needed for hand tracking.
        
        Unlike robot motors that need range calibration, hand tracking produces
        raw metric tip vectors that are fed directly to the IK solver.
        The IK solver handles the mapping to robot joint space.
        
        This method exists for interface compatibility but does nothing.
        """
        print("=" * 60)
        print("HAND TRACKING CALIBRATION NOT REQUIRED")
        print("=" * 60)
        print("\nHand tracking sends raw metric tip vectors (in meters)")
        print("directly to the robot's IK solver. No calibration needed.")
        print("\nIf the robot doesn't follow your hand correctly:")
        print("1. Check robot motor calibration (robot.calibrate())")
        print("2. Verify MJCF model matches your robot")
        print("3. Check axis flips/permutations in robot code")
        print("=" * 60)
        print("\nSkipping calibration - not required for hand tracking.\n")

    def _get_current_hand_state(self, wait_for_frame: bool = False) -> dict[str, float]:
        """Get the current hand joint positions (3D tip positions in hand frame)
        
        Args:
            wait_for_frame: If True, wait up to 1 second for a frame (used during calibration).
                           If False, return immediately with current state (used during teleoperation).
        """
        if wait_for_frame and not self._frame_ready.wait(timeout=1):
            raise TimeoutError(f"{self}: Timed out waiting for frame")
        
        with self._state_lock:
            state = self._state.copy() if self._state else {}
            
        return state
        
    # ------------------------------------------------------------
    # Tracking & Smoothing
    # ------------------------------------------------------------
    # NOTE: _apply_ema removed - no smoothing on Python side (Demo only has PID on Rust side)
    
    def _tracking_loop(self):
        """Main tracking loop that processes frames and updates joint positions"""
        logger.info(f"{self}: Starting tracking loop thread...")
        
        try:
            import cv2
            import mediapipe as mp
            import numpy as np
            import time

            # Initialize MediaPipe Hands
            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            
            logger.info(f"{self}: Initializing MediaPipe Hands...")
            # Get confidence thresholds from config or use defaults
            min_detection_conf = float(getattr(self.config, "min_detection_confidence", 0.7))
            min_tracking_conf = float(getattr(self.config, "min_tracking_confidence", 0.7))
            model_complexity = int(getattr(self.config, "model_complexity", 0))
            
            hands = mp_hands.Hands(
                static_image_mode=False,  # Video mode for better tracking
                model_complexity=model_complexity,  # 0=fastest, 1=balanced (default changed to 0.7 for better tracking)
                max_num_hands=1,
                min_detection_confidence=min_detection_conf,
                min_tracking_confidence=min_tracking_conf
            )
            logger.info(f"{self}: MediaPipe Hands initialized (detection_conf={min_detection_conf}, tracking_conf={min_tracking_conf}, model={model_complexity})")

            # Create window for visualization (optional)
            window_name = f"MediaPipe Hands - {self.config.side}"
            preview_enabled = bool(getattr(self.config, "show_preview", False))
            overlay_enabled = bool(getattr(self.config, "show_overlay", True))
            save_debug_frames = bool(getattr(self.config, "save_debug_frames", False))
            debug_frame_interval = int(getattr(self.config, "debug_frame_interval", 30))
            frame_counter = 0
            
            # Create debug output directory if saving frames
            if save_debug_frames:
                import os
                from pathlib import Path
                debug_dir = Path("outputs/hand_tracking_debug")
                debug_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"{self}: Saving debug frames to {debug_dir.absolute()}")
            
            if preview_enabled:
                try:
                    # startWindowThread can help on some platforms; no-op on others
                    try:
                        cv2.startWindowThread()
                    except Exception:
                        pass
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                except Exception as e:
                    logger.warning(f"{self}: Failed to create OpenCV window: {e}. Disabling preview.")
                    preview_enabled = False

            while not self._stop_event.is_set():
                try:
                    loop_start = time.time()
                    frame_counter += 1
                    
                    # Get frame from camera
                    frame = next(iter(self.cameras.values())).async_read()
                    if frame is None:
                        continue
                    
                    # Flip for selfie view
                    frame = cv2.flip(frame, 1)

                    # Process frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb.flags.writeable = False
                    results = hands.process(frame_rgb)
                    frame_rgb.flags.writeable = True
                    
                    # Convert back to BGR for display with correct colors
                    display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Extract hand landmarks
                    if results.multi_hand_landmarks and results.multi_handedness:
                        hand_detected = False
                        for index, handedness_classif in enumerate(results.multi_handedness):
                            score = handedness_classif.classification[0].score
                            hand_label = handedness_classif.classification[0].label
                            
                            # Filter: only process the hand that matches our configured side
                            if hand_label.lower() != self.config.side.lower():
                                continue
                            
                            if score > 0.8:
                                hand_detected = True
                                # Get world landmarks (metric 3D coordinates)
                                hand_landmarks = results.multi_hand_world_landmarks[index]
                                # Get normalized landmarks for drawing
                                hand_landmarks_norm = results.multi_hand_landmarks[index]
                                
                                # Draw landmarks and connections using MediaPipe's default styles (same as Demo)
                                # This provides rich, multi-colored visualization where different parts are easily distinguishable
                                mp_drawing.draw_landmarks(
                                    display_frame,
                                    hand_landmarks_norm,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                    mp_drawing_styles.get_default_hand_connections_style()
                                )
                                
                                # Convert landmarks to joint positions using original algorithm
                                joint_positions = self._landmarks_to_joint_positions(
                                    hand_landmarks, hand_landmarks_norm, hand_label
                                )
                                
                                # Store RAW joint positions - NO smoothing (like Demo)
                                with self._state_lock:
                                    self._state = joint_positions
                                    self._frame = display_frame
                                    self._last_raw = joint_positions
                                
                                break  # Only process first valid hand
                        
                        # Store frame even if no hands detected
                        with self._state_lock:
                            self._frame = display_frame
                    
                    # Display the frame (optional). On Windows, HighGUI can freeze in non-main thread; catch and disable.
                    if preview_enabled:
                        try:
                            cv2.imshow(window_name, display_frame)
                            # waitKey with very short timeout (1ms) to prevent blocking
                            cv2.waitKey(1)
                        except Exception as e:
                            logger.warning(f"{self}: OpenCV preview error: {e}. Disabling preview to prevent freezing.")
                            preview_enabled = False
                            try:
                                cv2.destroyAllWindows()
                            except Exception:
                                pass
                    
                    # Save debug frames (for external viewing without threading issues)
                    if save_debug_frames and frame_counter % debug_frame_interval == 0:
                        try:
                            frame_path = debug_dir / f"frame_{frame_counter:06d}.jpg"
                            cv2.imwrite(str(frame_path), display_frame)
                        except Exception:
                            pass
                    
                    # Signal that we're getting frames regardless of hand detection
                    self._frame_ready.set()
                    
                    # Frame rate limiter: target 30 FPS
                    elapsed = time.time() - loop_start
                    sleep_time = max(0, (1.0 / 30.0) - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                except Exception as e:
                    logger.warning(f"{self}: Error in tracking loop iteration: {e}", exc_info=True)
                    # Signal frame ready even on error to prevent deadlock
                    self._frame_ready.set()
                    time.sleep(0.1)  # Brief pause before retry
            
            # Clean up
            hands.close()
            if preview_enabled:
                try:
                    cv2.destroyWindow(window_name)
                except Exception:
                    pass
                    
        except Exception as e:
            logger.error(f"{self}: Fatal error in tracking loop: {e}", exc_info=True)
            raise

    def _landmarks_to_joint_positions(self, hand_landmarks, hand_landmarks_norm, hand_label) -> dict[str, float]:
        """Convert MediaPipe hand landmarks to joint positions using original AmazingHand algorithm
        
        This function follows the exact same logic as the original process_img() from
        d:\SHRDC\AmazingHand-main\Demo\HandTracking\HandTracking\main.py
        """
        import numpy as np
        import os
        import mediapipe as mp
        
        mp_hands = mp.solutions.hands
        
        # Calculate tip vectors (TIP - MCP) for each finger in WORLD coordinates (metric)
        tip1_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
        tip1_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
        tip1_z = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z

        tip2_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x - hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
        tip2_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y - hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
        tip2_z = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z - hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z

        tip3_x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x - hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x
        tip3_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y - hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
        tip3_z = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z - hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z

        tip4_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x
        tip4_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
        tip4_z = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z
        
        # Define hand coordinate frame using NORMALIZED coordinates (matching original Demo)
        # This is critical: Demo uses hand_landmarks_norm for frame, not world coordinates
        # Wrist origin (normalized)
        origin = np.array([
            hand_landmarks_norm.landmark[mp_hands.HandLandmark.WRIST].x,
            hand_landmarks_norm.landmark[mp_hands.HandLandmark.WRIST].y,
            hand_landmarks_norm.landmark[mp_hands.HandLandmark.WRIST].z,
        ])

        # Middle finger MCP (base) in normalized
        mid_mcp = np.array([
            hand_landmarks_norm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
            hand_landmarks_norm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
            hand_landmarks_norm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z,
        ])

        # Z-axis: unit vector from wrist towards middle MCP (normalized)
        unit_z = mid_mcp - origin
        unit_z = unit_z / np.linalg.norm(unit_z)

        # Base of the pinky finger (normalized)
        pinky_mcp = np.array([
            hand_landmarks_norm.landmark[mp_hands.HandLandmark.PINKY_MCP].x,
            hand_landmarks_norm.landmark[mp_hands.HandLandmark.PINKY_MCP].y,
            hand_landmarks_norm.landmark[mp_hands.HandLandmark.PINKY_MCP].z,
        ])

        # Base of the index finger (normalized)
        index_mcp = np.array([
            hand_landmarks_norm.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
            hand_landmarks_norm.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
            hand_landmarks_norm.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z,
        ])
        
        # Vector towards Y direction depends on hand side (same as original)
        if hand_label == 'Right':
            vec_towards_y = pinky_mcp - origin  # vector from wrist base towards pinky base
        if hand_label == 'Left':
            vec_towards_y = index_mcp - origin  # vector from wrist base towards index base
        
        # X-axis: cross product of z and the vector towards pinky/index (same as original)
        unit_x = np.cross(vec_towards_y, unit_z)
        unit_x = unit_x / np.linalg.norm(unit_x)
        
        # Y-axis: complete coordinate system
        unit_y = np.cross(unit_z, unit_x)
        
        # Rotation matrix; Demo mirrors Y (uses -unit_y). Allow override via env var.
        mirror_env = os.getenv("LEROBOT_MIRROR_Y", "1")  # "1" or "true" to mirror (default), "0" to disable
        mirror_y = str(mirror_env).lower() in ("1", "true", "yes", "on")
        R = np.array([unit_x, (-unit_y if mirror_y else unit_y), unit_z]).reshape((3, 3))
        
        # Transform tip vectors into hand coordinate frame (same as original)
        tip1 = R @ np.array([tip1_x, tip1_y, tip1_z])
        tip2 = R @ np.array([tip2_x, tip2_y, tip2_z])
        tip3 = R @ np.array([tip3_x, tip3_y, tip3_z])
        tip4 = R @ np.array([tip4_x, tip4_y, tip4_z])

        # Prepare tip positions for IK solver
        tip_positions = {
            'tip1': tip1,
            'tip2': tip2,
            'tip3': tip3,
            'tip4': tip4,
        }

        # Return raw tip vectors - robot will handle motor conversion
        # This simplifies the teleoperator and puts all robot-specific logic on robot side
        positions = {
            'index_tip_x': float(tip1[0]),
            'index_tip_y': float(tip1[1]),
            'index_tip_z': float(tip1[2]),
            'middle_tip_x': float(tip2[0]),
            'middle_tip_y': float(tip2[1]),
            'middle_tip_z': float(tip2[2]),
            'ring_tip_x': float(tip3[0]),
            'ring_tip_y': float(tip3[1]),
            'ring_tip_z': float(tip3[2]),
            'thumb_tip_x': float(tip4[0]),
            'thumb_tip_y': float(tip4[1]),
            'thumb_tip_z': float(tip4[2]),
        }
        return positions

    # ------------------------------------------------------------
    # Action interface
    # ------------------------------------------------------------
    def get_action(self) -> dict[str, float]:
        """
        Returns raw tip vectors (in meters) for robot IK solver.
        No normalization is applied - robot expects metric coordinates.
        """
        state = self._read()
        # Return raw tip vectors without normalization
        action_dict = {f"{j}.pos": v for j, v in state.items()}
        
        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # Not used for tracking-only devices
        pass

    def configure(self) -> None:
        # No additional configuration needed for this device
        pass
        
    def _read(self) -> dict[str, float]:
        """Get the latest RAW hand state in meters (no normalization).
        The robot expects raw metric tip vectors for its IK solver.
        Always returns a full dict so downstream never sees an empty action.
        Fallback order: current state -> last raw -> neutral zeros.
        """
        raw = self._get_current_hand_state()
        if not raw:
            # Fallback to last raw values or neutral zeros
            raw = {j: (self._last_raw[j] if self._last_raw[j] is not None else 0.0) for j in self.joints}
        
        # NO EMA smoothing - send raw tracking data directly (like Demo does)
        return raw
