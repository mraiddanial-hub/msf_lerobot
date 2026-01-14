"""
PC1 (Leader) - Robot Control Sender (UDP)
Sends bi-SO100 leader commands via UDP to the follower.
This is kept separate from WebRTC camera streaming.
"""
import socket
import json
import time
from lerobot.teleoperators.bi_so100_leader.bi_so100_leader import BiSO100Leader
from lerobot.teleoperators.bi_so100_leader.config_bi_so100_leader import BiSO100LeaderConfig

# ========= CONFIG =========
FOLLOWER_IP = "192.168.0.90"  # PC2 IP address
UDP_PORT = 5005
SEND_HZ = 50

LEADER_LEFT_PORT = "COM7"  # Change to your left arm leader COM port
LEADER_RIGHT_PORT = "COM8"  # Change to your right arm leader COM port
LEADER_ID = "leader"

# ==========================

# Initialize UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
dt = 1.0 / SEND_HZ

# Initialize bi-SO100 leader (dual arms)
leader_cfg = BiSO100LeaderConfig(
    left_arm_port=LEADER_LEFT_PORT,
    right_arm_port=LEADER_RIGHT_PORT,
    id=LEADER_ID,
)
leader = BiSO100Leader(leader_cfg)
leader.connect()

print(f"Leader bi-SO100 initialized on {LEADER_LEFT_PORT} (left) and {LEADER_RIGHT_PORT} (right)")
print(f"Sending commands to {FOLLOWER_IP}:{UDP_PORT} at {SEND_HZ} Hz")

try:
    while True:
        # Get action from leader
        raw_action = leader.get_action()
        action = {key: float(value) for key, value in raw_action.items()}
        
        # Send via UDP
        msg = {
            "ts": time.time(),
            "action": action,
        }
        data = json.dumps(msg).encode("utf-8")
        sock.sendto(data, (FOLLOWER_IP, UDP_PORT))
        
        time.sleep(dt)

except KeyboardInterrupt:
    print("\nStopping sender...")
finally:
    try:
        leader.disconnect()
    except:
        pass
    sock.close()
    print("Leader sender stopped")
