"""
PC2 (Follower) - Robot Control Receiver (UDP)
Receives robot commands via UDP and controls the bi-SO100 follower arm.
This is kept separate from WebRTC camera streaming.

Features:
- Echo packets back to leader for RTT measurement
- Receive timestamped commands
"""
import socket
import json
from lerobot.robots.bi_so100.bi_so100 import BiSO100Robot
from lerobot.robots.bi_so100.config_bi_so100 import BiSO100RobotConfig

# ========= CONFIG =========
UDP_PORT = 5005
FOLLOWER_LEFT_PORT = "COM10"  # Change to your left arm COM port
FOLLOWER_RIGHT_PORT = "COM11"  # Change to your right arm COM port
FOLLOWER_ID = "follower"

# ==========================

# Initialize UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", UDP_PORT))

# Initialize bi-SO100 follower robot (dual arms)
follower_cfg = BiSO100RobotConfig(
    left_arm_port=FOLLOWER_LEFT_PORT,
    right_arm_port=FOLLOWER_RIGHT_PORT,
    id=FOLLOWER_ID,
)
follower = BiSO100Robot(follower_cfg)
follower.connect()

print(f"Follower robot initialized on {FOLLOWER_LEFT_PORT} (left) and {FOLLOWER_RIGHT_PORT} (right)")
print(f"Listening for commands on UDP port {UDP_PORT}")
print("Echoing packets for RTT measurement")

try:
    while True:
        data, addr = sock.recvfrom(4096)
        msg = json.loads(data.decode("utf-8"))
        
        # Echo back sequence number for RTT measurement
        if "seq" in msg:
            echo = {"seq": msg["seq"]}
            echo_data = json.dumps(echo).encode("utf-8")
            sock.sendto(echo_data, addr)
        
        action = msg.get("action")
        if action:
            # Send action to follower robot
            follower.send_action(action)

except KeyboardInterrupt:
    print("\nStopping receiver...")
finally:
    try:
        follower.disconnect()
    except:
        pass
    sock.close()
    print("Robot receiver stopped")
