"""
PC1 (Leader) - Robot Control Sender (UDP)
Sends SO-101 leader commands via UDP to the follower.
This is kept separate from WebRTC camera streaming.
"""
import socket
import json
import time
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig

# ========= CONFIG =========
FOLLOWER_IP = "100.127.72.97"  # PC2 IP address
UDP_PORT = 5005
SEND_HZ = 50

LEADER_PORT = "COM9"  # Change to your leader COM port
LEADER_ID = "leader"

# ==========================

# Initialize UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
dt = 1.0 / SEND_HZ

# Initialize SO-101 leader
leader_cfg = SO101LeaderConfig(
    port=LEADER_PORT,
    id=LEADER_ID,
)
leader = SO101Leader(leader_cfg)
leader.connect()

print(f"Leader SO-101 initialized on {LEADER_PORT}")
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
