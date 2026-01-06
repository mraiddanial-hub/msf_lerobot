#!/usr/bin/env python
"""
Inverse Kinematics for AmazingHand fingers using MuJoCo and mink.

This module provides IK solving to convert fingertip positions (from hand tracking)
to joint angles (for motor control), matching the original AHSimulation behavior.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Optional dependencies - will be imported lazily
mujoco = None
mink = None


class AmazingHandIK:
    """Inverse kinematics solver for AmazingHand using MuJoCo and mink"""
    
    def __init__(self, model_path: Optional[str] = None, use_ik: bool = True, dt: float = 0.001, max_iters: int = 10):
        """
        Initialize the IK solver.
        
        Args:
            model_path: Path to the MuJoCo scene.xml file. If None, tries to find it automatically.
            use_ik: If False, falls back to magnitude-based approximation without MuJoCo/mink
            dt: IK integration timestep in seconds (smaller = more accurate, default 0.001)
            max_iters: Number of IK solve iterations per frame (more = better convergence, default 10)
        """
        self.use_ik = use_ik
        self.model_path = None  # will be set to resolved path when model loads
        self.model = None
        self.configuration = None
        self.tasks = None
        self.solver = "quadprog"
        self.dt = dt
        self.max_iters = max_iters
        
        if not use_ik:
            logger.info("IK disabled, using magnitude-based approximation")
            return
            
        # Try to import optional dependencies
        global mujoco, mink
        try:
            import mujoco as mj
            import mink as mk
            mujoco = mj
            mink = mk
        except ImportError as e:
            logger.warning(f"Could not import mujoco/mink: {e}. Falling back to approximation.")
            self.use_ik = False
            return
        
        # Find model path
        if model_path is None:
            # Prefer a local copy shipped with the package first
            local_scene = Path(__file__).parent / "mjcf" / "scene.xml"
            possible_paths = [
                local_scene,
                # Fallback to the original AmazingHand repo locations
                Path("d:/SHRDC/AmazingHand-main/Demo/AHSimulation/AHSimulation/AH_Left/mjcf/scene.xml"),
                Path.home() / "AmazingHand" / "Demo" / "AHSimulation" / "AHSimulation" / "AH_Left" / "mjcf" / "scene.xml",
            ]
            for path in possible_paths:
                if path.exists():
                    model_path = str(path)
                    break
        
        if model_path is None or not Path(model_path).exists():
            logger.warning(f"MuJoCo model not found at {model_path}. Falling back to approximation.")
            self.use_ik = False
            return
        
        try:
            # Load MuJoCo model
            logger.info(f"Loading MuJoCo model from: {model_path}")
            self.model_path = str(model_path)
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.configuration = mink.Configuration(self.model)
            
            # Create IK tasks for each fingertip
            posture_task = mink.PostureTask(self.model, cost=1e-2)
            
            # Frame tasks for fingertips (position control only, no orientation)
            task1 = mink.FrameTask(
                frame_name='tip1',
                frame_type="site",
                position_cost=1.0,
                orientation_cost=0.0,
                lm_damping=1.0,
            )
            task2 = mink.FrameTask(
                frame_name='tip2',
                frame_type="site",
                position_cost=1.0,
                orientation_cost=0.0,
                lm_damping=1.0,
            )
            task3 = mink.FrameTask(
                frame_name='tip3',
                frame_type="site",
                position_cost=1.0,
                orientation_cost=0.0,
                lm_damping=1.0,
            )
            task4 = mink.FrameTask(
                frame_name='tip4',
                frame_type="site",
                position_cost=1.0,
                orientation_cost=0.0,
                lm_damping=1.0,
            )
            
            # Equality constraint task
            eq_task = mink.EqualityConstraintTask(self.model, cost=1000.0)
            
            self.tasks = [eq_task, posture_task, task1, task2, task3, task4]
            
            # Initialize from zero keyframe
            self.configuration.update_from_keyframe("zero")
            posture_task.set_target_from_configuration(self.configuration)
            
            # Store task references for setting targets
            self.task1 = task1
            self.task2 = task2
            self.task3 = task3
            self.task4 = task4
            self.posture_task = posture_task
            
            # Get joint IDs for reading motor positions
            self.joint_ids = {
                'finger1_motor1': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger1_motor1"),
                'finger1_motor2': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger1_motor2"),
                'finger2_motor1': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger2_motor1"),
                'finger2_motor2': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger2_motor2"),
                'finger3_motor1': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger3_motor1"),
                'finger3_motor2': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger3_motor2"),
                'finger4_motor1': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger4_motor1"),
                'finger4_motor2': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger4_motor2"),
            }
            
            self.data = self.configuration.data
            
            logger.info(f"IK solver initialized successfully with dt={self.dt}, max_iters={self.max_iters}")
            
        except Exception as e:
            logger.error(f"Failed to initialize IK solver: {e}", exc_info=True)
            self.use_ik = False
    
    def solve(self, tip_positions: dict[str, np.ndarray]) -> dict[str, float]:
        """
        Solve inverse kinematics to get joint angles from tip positions.
        
        Args:
            tip_positions: Dictionary with keys 'tip1', 'tip2', 'tip3', 'tip4'
                          Each value is a 3D position vector [x, y, z] in the hand frame
        
        Returns:
            Dictionary with motor positions:
            - 'index_m1', 'index_m2' (from tip1)
            - 'middle_m1', 'middle_m2' (from tip2)
            - 'ring_m1', 'ring_m2' (from tip3)
            - 'thumb_m1', 'thumb_m2' (from tip4)
        """
        if not self.use_ik:
            # Fallback to magnitude-based approximation
            logger.debug("Using magnitude-based approximation (IK disabled)")
            return self._solve_magnitude(tip_positions)
        
        try:
            # Log input tip positions
            logger.debug(f"IK input tip positions:")
            for tip_name, tip_pos in tip_positions.items():
                logger.debug(f"  {tip_name}: [{tip_pos[0]:.4f}, {tip_pos[1]:.4f}, {tip_pos[2]:.4f}]")
            
            # Scale and offset tip positions to match MuJoCo coordinate frame
            # These values match the original mj_mink_left.py scaling
            scale = 1.5
            offsets = {
                'tip1': [0.025, -0.022, 0.098],  # index
                'tip2': [0.025, 0.009, 0.092],   # middle
                'tip3': [0.025, 0.040, 0.082],   # ring
                'tip4': [0.024, -0.019, 0.017],  # thumb
            }
            
            # Set target positions for each fingertip task
            for i, (tip_name, tip_pos) in enumerate(tip_positions.items(), 1):
                offset = offsets[tip_name]
                scaled_pos = tip_pos * scale + np.array(offset)
                
                logger.debug(f"  {tip_name} scaled: [{scaled_pos[0]:.4f}, {scaled_pos[1]:.4f}, {scaled_pos[2]:.4f}]")
                
                # Create SE3 transform for the target position (position only, no rotation)
                target_transform = mink.SE3.from_translation(scaled_pos)
                
                # Set target for corresponding task
                task = getattr(self, f'task{i}')
                task.set_target(target_transform)
            
            # Solve IK with multiple iterations for better convergence
            # Original runs continuously at 1000Hz, we approximate by iterating here
            for _ in range(self.max_iters):
                vel = mink.solve_ik(self.configuration, self.tasks, self.dt, self.solver, 1e-5)
                self.configuration.integrate_inplace(vel, self.dt)
            
            # Read joint positions
            motor_positions = {
                'index_m1': self.data.joint(self.joint_ids['finger1_motor1']).qpos[0],
                'index_m2': self.data.joint(self.joint_ids['finger1_motor2']).qpos[0],
                'middle_m1': self.data.joint(self.joint_ids['finger2_motor1']).qpos[0],
                'middle_m2': self.data.joint(self.joint_ids['finger2_motor2']).qpos[0],
                'ring_m1': self.data.joint(self.joint_ids['finger3_motor1']).qpos[0],
                'ring_m2': self.data.joint(self.joint_ids['finger3_motor2']).qpos[0],
                'thumb_m1': self.data.joint(self.joint_ids['finger4_motor1']).qpos[0],
                'thumb_m2': self.data.joint(self.joint_ids['finger4_motor2']).qpos[0],
            }
            
            # Apply motor offsets from original l_hand.toml config
            # These offsets calibrate the robot hand to match the simulation
            motor_offsets = {
                'index_m1': 0.031,
                'index_m2': 0.173,
                'middle_m1': -0.601,
                'middle_m2': 0.614,
                'ring_m1': -0.990,
                'ring_m2': 0.915,
                'thumb_m1': -0.101,
                'thumb_m2': 0.511,
            }
            
            for motor in motor_positions:
                motor_positions[motor] += motor_offsets[motor]
            
            # Log output joint angles (in radians) AFTER offsets
            logger.debug(f"IK output joint angles (radians) AFTER offsets:")
            logger.debug(f"  index: m1={motor_positions['index_m1']:.4f}, m2={motor_positions['index_m2']:.4f}")
            logger.debug(f"  middle: m1={motor_positions['middle_m1']:.4f}, m2={motor_positions['middle_m2']:.4f}")
            logger.debug(f"  ring: m1={motor_positions['ring_m1']:.4f}, m2={motor_positions['ring_m2']:.4f}")
            logger.debug(f"  thumb: m1={motor_positions['thumb_m1']:.4f}, m2={motor_positions['thumb_m2']:.4f}")
            
            return motor_positions
            
        except Exception as e:
            logger.error(f"IK solve failed: {e}", exc_info=True)
            # Fallback to approximation
            return self._solve_magnitude(tip_positions)
    
    def _solve_magnitude(self, tip_positions: dict[str, np.ndarray]) -> dict[str, float]:
        """
        Fallback approximation using vector magnitude.
        This is what we use when MuJoCo/mink is not available.
        """
        tip1_mag = np.linalg.norm(tip_positions['tip1'])
        tip2_mag = np.linalg.norm(tip_positions['tip2'])
        tip3_mag = np.linalg.norm(tip_positions['tip3'])
        tip4_mag = np.linalg.norm(tip_positions['tip4'])
        
        return {
            'index_m1': tip1_mag,
            'index_m2': tip1_mag,
            'middle_m1': tip2_mag,
            'middle_m2': tip2_mag,
            'ring_m1': tip3_mag,
            'ring_m2': tip3_mag,
            'thumb_m1': tip4_mag,
            'thumb_m2': tip4_mag,
        }
