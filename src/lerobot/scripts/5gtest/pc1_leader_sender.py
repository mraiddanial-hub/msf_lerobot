# pc1_leader_sender.py
import socket
import json
import time

from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig

# ========= CONFIG (EDIT THESE!) =========
FOLLOWER_IP = "100.107.126.39"   # <-- IP of PC2
UDP_PORT = 5005
SEND_HZ = 50                    # 50 Hz command rate

LEADER_PORT = "COM8"            # <-- change to your leader SO-101 COM port
LEADER_ID = "leader"    # arbitrary name

# ========= INITIALISE UDP SOCKET =========
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
dt = 1.0 / SEND_HZ

# ========= INITIALISE LEADER SO-101 =========
leader_cfg = SO101LeaderConfig(
    port=LEADER_PORT,
    id=LEADER_ID,
)
leader = SO101Leader(leader_cfg)
leader.connect()

print("Leader SO-101 initialised on", LEADER_PORT)


def get_leader_action() -> dict:
    """
    Get the current action from the SO-101 leader.

    SO101Leader.get_action() returns a dict like:
      {"joint1.pos": value, "joint2.pos": value, ...}

    We also cast all values to plain float so JSON can encode them.
    """
    raw_action = leader.get_action()  # <-- correct API for SO101Leader

    # Make sure everything is JSON-serialisable (no numpy.float32 etc.)
    action = {key: float(value) for key, value in raw_action.items()}
    return action


print(f"Sending leader actions to {FOLLOWER_IP}:{UDP_PORT} at {SEND_HZ} Hz")

try:
    while True:
        action = get_leader_action()
        msg = {
            "ts": time.time(),   # timestamp on PC1
            "action": action,    # dict of joint commands
        }
        data = json.dumps(msg).encode("utf-8")
        sock.sendto(data, (FOLLOWER_IP, UDP_PORT))
        time.sleep(dt)

finally:
    try:
        leader.disconnect()
    except Exception:
        pass
    sock.close()
    print("Leader teleop sender stopped.")
