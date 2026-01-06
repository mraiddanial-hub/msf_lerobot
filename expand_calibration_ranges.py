#!/usr/bin/env python3
"""
Expand calibration ranges for AmazingHand teleoperation.

The calibration GUI records min/max positions during calibration.
If you didn't move fingers through their full range, the recorded ranges are too small.
This script expands the ranges while keeping a safety margin from hardware limits (0-4095).
"""

import json
from pathlib import Path

# Path to calibration file
calib_path = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "lerobot_robot_amazinghand" / "None.json"

if not calib_path.exists():
    print(f"❌ Calibration file not found: {calib_path}")
    print("Run teleoperation first to generate calibration.")
    exit(1)

# Load current calibration
with open(calib_path) as f:
    calib = json.load(f)

print("Current calibration ranges:")
print("=" * 60)
for motor, config in calib.items():
    current_range = config["range_max"] - config["range_min"]
    current_pct = (current_range / 4095) * 100
    print(f"{motor:12s}: {config['range_min']:4d} - {config['range_max']:4d}  (range={current_range:4d}, {current_pct:5.1f}% of full)")

print("\n" + "=" * 60)
print("Expanding ranges to ~80% of full range (safety margin)...")
print("This gives max range while protecting against hardware limits.")
print("=" * 60)

# Expand ranges: use center ± 40% of full range (80% total)
# Full range is 0-4095, so 80% = 3276 positions
EXPANSION_FACTOR = 0.8  # Use 80% of full 0-4095 range
FULL_RANGE = 4095
EXPANDED_RANGE = int(FULL_RANGE * EXPANSION_FACTOR)
MARGIN = (FULL_RANGE - EXPANDED_RANGE) // 2

for motor, config in calib.items():
    # Find current center
    current_center = (config["range_min"] + config["range_max"]) // 2
    
    # Expand around center
    new_min = max(0, current_center - EXPANDED_RANGE // 2)
    new_max = min(FULL_RANGE, current_center + EXPANDED_RANGE // 2)
    
    # Ensure we don't exceed bounds
    if new_min < MARGIN:
        new_min = MARGIN
    if new_max > FULL_RANGE - MARGIN:
        new_max = FULL_RANGE - MARGIN
    
    old_range = config["range_max"] - config["range_min"]
    new_range = new_max - new_min
    
    config["range_min"] = new_min
    config["range_max"] = new_max
    
    print(f"{motor:12s}: {new_min:4d} - {new_max:4d}  (range={new_range:4d}, {old_range:4d} → {new_range:4d} = +{new_range-old_range:4d})")

# Backup original
backup_path = calib_path.with_suffix(".json.backup")
with open(backup_path, "w") as f:
    json.dump(calib, f, indent=2)
print(f"\n✅ Original calibration backed up to: {backup_path}")

# Save expanded calibration
with open(calib_path, "w") as f:
    json.dump(calib, f, indent=2)
print(f"✅ Expanded calibration saved to: {calib_path}")
print("\nRestart teleoperation to use the expanded ranges!")
print("If motors hit limits or behave strangely, restore backup:")
print(f"  Copy-Item '{backup_path}' '{calib_path}' -Force")
