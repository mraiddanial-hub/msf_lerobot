# AmazingHand Robot Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
# Install base LeRobot package with AmazingHand dependencies
pip install -r requirements-amazinghand.txt
```

### 2. Hardware Setup

- Connect AmazingHand robot to serial port (e.g., COM5 on Windows)
- Connect two cameras:
  - Camera 0: Robot observation camera
  - Camera 1: Hand tracking camera for teleoperation

### 3. Configure Hand Side

The robot supports both left and right hand configurations:

```bash
# For left hand (default)
--robot.side=left --robot.port=COM5

# For right hand
--robot.side=right --robot.port=COM6
```

**Motor ID Mapping:**
- Left hand: IDs 15, 16, 13, 14, 11, 12, 17, 18
- Right hand: IDs 5, 6, 3, 4, 1, 2, 7, 8 (left hand IDs - 10)

### 4. Teleoperation

**Windows PowerShell:**
```powershell
lerobot-teleoperate `
  --robot.type=lerobot_robot_amazinghand `
  --teleop.type=lerobot_teleoperator_amazinghandtracker `
  --robot.port=<COM_PORT> `
  --robot.side=<left|right> `
  --robot.baudrate=1000000 `
  --robot.cameras="{ cam_0: {type: opencv, index_or_path: <ROBOT_CAM_ID>, width: 640, height: 480, fps: 30}}" `
  --teleop.side=<left|right> `
  --teleop.camera_id=<TELEOP_CAM_ID> `
  --teleop.cameras="{ cam_1: {type: opencv, index_or_path: <TELEOP_CAM_ID>, width: 640, height: 480, fps: 30}}" `
  --display_data=true
```

**Linux/Mac:**
```bash
lerobot-teleoperate \
  --robot.type=lerobot_robot_amazinghand \
  --teleop.type=lerobot_teleoperator_amazinghandtracker \
  --robot.port=<SERIAL_PORT> \
  --robot.side=<left|right> \
  --robot.baudrate=1000000 \
  --robot.cameras="{ cam_0: {type: opencv, index_or_path: <ROBOT_CAM_ID>, width: 640, height: 480, fps: 30}}" \
  --teleop.side=<left|right> \
  --teleop.camera_id=<TELEOP_CAM_ID> \
  --teleop.cameras="{ cam_1: {type: opencv, index_or_path: <TELEOP_CAM_ID>, width: 640, height: 480, fps: 30}}" \
  --display_data=true
```

**Example:**
- Windows: `--robot.port=COM5`, camera IDs: `0` and `1`
- Linux: `--robot.port=/dev/ttyUSB0`, camera IDs: `0` and `1`

### 5. Recording Episodes

**Windows PowerShell:**
```powershell
lerobot-record `
  --robot.type=lerobot_robot_amazinghand `
  --teleop.type=lerobot_teleoperator_amazinghandtracker `
  --robot.port=<COM_PORT> `
  --robot.side=<left|right> `
  --robot.baudrate=1000000 `
  --robot.cameras='{ "cam_0": {"type": "opencv", "index_or_path": <ROBOT_CAM_ID>, "width": 640, "height": 480, "fps": 30} }' `
  --teleop.side=<left|right> `
  --teleop.camera_id=<TELEOP_CAM_ID> `
  --teleop.cameras='{ "cam_1": {"type": "opencv", "index_or_path": <TELEOP_CAM_ID>, "width": 640, "height": 480, "fps": 30} }' `
  --display_data=true `
  --dataset.repo_id="<username>/<dataset-name>" `
  --dataset.num_episodes=<NUM_EPISODES> `
  --dataset.single_task="<Task description>"
```

**Linux/Mac:**
```bash
lerobot-record \
  --robot.type=lerobot_robot_amazinghand \
  --teleop.type=lerobot_teleoperator_amazinghandtracker \
  --robot.port=<SERIAL_PORT> \
  --robot.side=<left|right> \
  --robot.baudrate=1000000 \
  --robot.cameras='{ "cam_0": {"type": "opencv", "index_or_path": <ROBOT_CAM_ID>, "width": 640, "height": 480, "fps": 30} }' \
  --teleop.side=<left|right> \
  --teleop.camera_id=<TELEOP_CAM_ID> \
  --teleop.cameras='{ "cam_1": {"type": "opencv", "index_or_path": <TELEOP_CAM_ID>, "width": 640, "height": 480, "fps": 30} }' \
  --display_data=true \
  --dataset.repo_id="<username>/<dataset-name>" \
  --dataset.num_episodes=<NUM_EPISODES> \
  --dataset.single_task="<Task description>"
```

**Example:**
- `--dataset.repo_id="raiddanial/grasp-cube"`
- `--dataset.num_episodes=50`
- `--dataset.single_task="Grasp the purple cube"`

### 6. Training

#### ACT Policy (Recommended for 2GB VRAM)

**Windows PowerShell:**
```powershell
lerobot-train `
  --dataset.repo_id="<username>/<dataset-name>" `
  --policy.type=act `
  --output_dir=<OUTPUT_DIR> `
  --job_name=<JOB_NAME> `
  --policy.device=cuda `
  --wandb.enable=<true|false> `
  --batch_size=2 `
  --save_freq=2000 `
  --steps=<NUM_STEPS>
```

**Linux/Mac:**
```bash
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="<username>/<dataset-name>" \
  --policy.type=act \
  --output_dir=<OUTPUT_DIR> \
  --job_name=<JOB_NAME> \
  --policy.device=cuda \
  --wandb.enable=<true|false> \
  --batch_size=2 \
  --save_freq=2000 \
  --steps=<NUM_STEPS>
```

**Example:**
- Windows output: `D:\lerobot_train\act_grasp_cube`
- Linux output: `./outputs/train/act_grasp_cube`
- Steps: `50000` (typical for ACT)

#### SmolVLA Policy (Requires more VRAM)

**Windows PowerShell:**
```powershell
lerobot-train `
  --dataset.repo_id="<username>/<dataset-name>" `
  --policy.type=smolvla `
  --output_dir=<OUTPUT_DIR> `
  --job_name=<JOB_NAME> `
  --policy.device=cuda `
  --wandb.enable=<true|false> `
  --batch_size=1 `
  --save_freq=2000 `
  --steps=<NUM_STEPS> `
  --policy.push_to_hub=false
```

**Linux/Mac:**
```bash
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="<username>/<dataset-name>" \
  --policy.type=smolvla \
  --output_dir=<OUTPUT_DIR> \
  --job_name=<JOB_NAME> \
  --policy.device=cuda \
  --wandb.enable=<true|false> \
  --batch_size=1 \
  --save_freq=2000 \
  --steps=<NUM_STEPS> \
  --policy.push_to_hub=false
```

**Example:**
- Batch size: `1` (for 2GB VRAM GPUs like GTX 1050 Ti)
- Steps: `10000-50000` (depending on dataset size)

### 7. Replay Episodes

**Windows PowerShell:**
```powershell
lerobot-replay `
  --robot.type=lerobot_robot_amazinghand `
  --robot.port=<COM_PORT> `
  --robot.side=<left|right> `
  --robot.baudrate=1000000 `
  --robot.cameras='{ "cam_0": {"type": "opencv", "index_or_path": <ROBOT_CAM_ID>, "width": 640, "height": 480, "fps": 30} }' `
  --dataset.repo_id="<username>/<dataset-name>" `
  --dataset.episode=<EPISODE_INDEX>
```

**Linux/Mac:**
```bash
lerobot-replay \
  --robot.type=lerobot_robot_amazinghand \
  --robot.port=<SERIAL_PORT> \
  --robot.side=<left|right> \
  --robot.baudrate=1000000 \
  --robot.cameras='{ "cam_0": {"type": "opencv", "index_or_path": <ROBOT_CAM_ID>, "width": 640, "height": 480, "fps": 30} }' \
  --dataset.repo_id="<username>/<dataset-name>" \
  --dataset.episode=<EPISODE_INDEX>
```

**Example:**
- Episode indices start at `0`
- Use this to test recorded episodes before training

### 8. Evaluation with Policy

**Windows PowerShell:**
```powershell
lerobot-record `
  --robot.type=lerobot_robot_amazinghand `
  --robot.port=<COM_PORT> `
  --robot.side=<left|right> `
  --robot.baudrate=1000000 `
  --robot.cameras="{ cam_0: {type: opencv, index_or_path: <ROBOT_CAM_ID>, width: 640, height: 480, fps: 30}}" `
  --display_data=true `
  --dataset.repo_id="<username>/<eval-dataset-name>" `
  --dataset.num_episodes=<NUM_EVAL_EPISODES> `
  --dataset.episode_time_s=<TIMEOUT_SECONDS> `
  --dataset.reset_time_s=<RESET_TIME> `
  --dataset.single_task="<Task description>" `
  --policy.path=<CHECKPOINT_PATH> `
  --dataset.push_to_hub=false
```

**Linux/Mac:**
```bash
lerobot-record \
  --robot.type=lerobot_robot_amazinghand \
  --robot.port=<SERIAL_PORT> \
  --robot.side=<left|right> \
  --robot.baudrate=1000000 \
  --robot.cameras="{ cam_0: {type: opencv, index_or_path: <ROBOT_CAM_ID>, width: 640, height: 480, fps: 30}}" \
  --display_data=true \
  --dataset.repo_id="<username>/<eval-dataset-name>" \
  --dataset.num_episodes=<NUM_EVAL_EPISODES> \
  --dataset.episode_time_s=<TIMEOUT_SECONDS> \
  --dataset.reset_time_s=<RESET_TIME> \
  --dataset.single_task="<Task description>" \
  --policy.path=<CHECKPOINT_PATH>
```

**Example:**
- Windows checkpoint: `D:\lerobot_train\act_grasp_cube\checkpoints\last\pretrained_model`
- Linux checkpoint: `./outputs/train/act/checkpoints/last/pretrained_model`
- Episode time: `600` seconds (10 minutes)
- Reset time: `60` seconds (1 minute between episodes)
- Num episodes: `10` (typical evaluation run)

## Optional Features

### MuJoCo Visualization

Enable real-time MuJoCo simulation viewer:

```bash
$env:LEROBOT_SHOW_MUJOCO_VIEWER="1"
```

Then run teleoperation/recording as usual. A 3D viewer window will show the simulated hand.

### Performance Tuning

The robot is configured for maximum speed:
- Goal_Speed: 2047 (max)
- Maximum_Acceleration: 254 (max)
- IK_TO_DELTA_SCALE: 2.0

To adjust, modify the values in `amazinghand.py` or use environment variables.

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError: No module named 'mujoco'`:
```bash
pip install mujoco mink mediapipe
```

### CUDA Not Detected
Use the full Python path when training:
```bash
python -m lerobot.scripts.lerobot_train ...
```
Not: `lerobot-train ...` (executable may use wrong Python environment)

### Disk Space Issues
Training outputs can be large. Use a drive with sufficient space:
```bash
--output_dir=D:\lerobot_train\your_experiment
```

### Motor Connection Issues
1. Check COM port in Device Manager
2. Verify baudrate (default: 1000000)
3. Test with: `python -c "from lerobot.robots.lerobot_robot_amazinghand.lerobot_robot_amazinghand import AmazingHand"`

## File Structure

```
msf_lerobot/
├── requirements-amazinghand.txt          # This file's dependencies
├── src/lerobot/robots/
│   └── lerobot_robot_amazinghand/
│       └── lerobot_robot_amazinghand/
│           ├── AH_Left/                  # Left hand MJCF model
│           │   └── mjcf/scene.xml
│           ├── AH_Right/                 # Right hand MJCF model
│           │   └── mjcf/scene.xml
│           ├── amazinghand.py            # Robot driver
│           └── config_amazinghand.py     # Robot config
└── src/lerobot/teleoperators/
    └── lerobot_teleoperator_amazinghandtracker/
        └── lerobot_teleoperator_amazinghandtracker/
            ├── amazinghandtracker.py     # Teleoperation tracker
            └── config_amazinghandtracker.py
```

## Citation

If you use this code, please cite the LeRobot project and acknowledge the AmazingHand hardware.
