# test_import_so101.py
import lerobot

print("LeRobot path:", lerobot.__file__)

# SO-101 leader (teleoperator)
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

# SO-101 follower (robot)
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

print("SO101Leader:", SO101Leader)
print("SO101LeaderConfig:", SO101LeaderConfig)
print("SO101Follower:", SO101Follower)
print("SO101FollowerConfig:", SO101FollowerConfig)
