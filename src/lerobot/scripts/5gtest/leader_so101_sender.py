import socket
import json
import time

# ==== CONFIG ====
FOLLOWER_IP = "192.168.8.252"   # IP of PC2
UDP_PORT = 5005
SEND_HZ = 50                   # send at 50 Hz (20 ms)

# --- TODO: import and init your SO-101 leader driver here ---
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

leader_cfg = SO101LeaderConfig(port="COM9", id="my_leader")  # change COM7
leader = SO101Leader(leader_cfg)
leader.connect()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
dt = 1.0 / SEND_HZ

def get_leader_joints():
    obs = leader.get_observation()   # returns a dict
    # TODO: pick the right key from obs that contains the 6 joint values
    q = obs["state"]                 # example – we’ll adjust once we see obs
    return list(q)
    

print(f"Sending leader joints to {FOLLOWER_IP}:{UDP_PORT} at {SEND_HZ} Hz")

while True:
    q = get_leader_joints()
    msg = {
        "ts": time.time(),   # timestamp on PC1
        "q": q
    }
    data = json.dumps(msg).encode("utf-8")
    sock.sendto(data, (FOLLOWER_IP, UDP_PORT))
    time.sleep(dt)
