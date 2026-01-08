"""
PC2 (Follower) - Robot Control Receiver (UDP)
Receives robot commands via UDP and controls the SO-101 follower arm.
This is kept separate from WebRTC camera streaming.
"""
import socket
import json
from lerobot.robots.so101.so101 import SO101Robot
from lerobot.robots.so101.config_so101 import SO101RobotConfig

# ========= CONFIG =========
UDP_PORT = 5005
FOLLOWER_PORT = "COM10"  # Change to your follower robot COM port
FOLLOWER_ID = "follower"

# ==========================

# Initialize UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", UDP_PORT))

# Initialize SO-101 follower robot
follower_cfg = SO101RobotConfig(
    port=FOLLOWER_PORT,
    id=FOLLOWER_ID,
)
follower = SO101Robot(follower_cfg)
follower.connect()

print(f"Follower robot initialized on {FOLLOWER_PORT}")
print(f"Listening for commands on UDP port {UDP_PORT}")

try:
    while True:
        data, addr = sock.recvfrom(4096)
        msg = json.loads(data.decode("utf-8"))
        
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
