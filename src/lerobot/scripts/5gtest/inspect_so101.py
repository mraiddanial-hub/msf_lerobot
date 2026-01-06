import lerobot
import inspect

print("LeRobot package path:", lerobot.__file__)

# 1) Import robot configs module
from lerobot.common.robot_devices.robots import configs

# 2) List what’s inside configs
print("\nAttributes in lerobot.common.robot_devices.robots.configs:")
for name in dir(configs):
    if "So101" in name or "so101" in name:
        print("  ", name)

# 3) Try to import the So101RobotConfig directly
from lerobot.common.robot_devices.robots.configs import So101RobotConfig

print("\nSo101RobotConfig definition:")
print(inspect.getsource(So101RobotConfig)[:600], "...\n")
