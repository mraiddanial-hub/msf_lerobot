"""
PC1 (Leader) - Robot Control Sender (UDP)
Sends bi-SO100 leader commands via UDP to the follower.
This is kept separate from WebRTC camera streaming.

Features:
- RTT (Round-Trip Time) measurement
- Jitter calculation
- Packet loss detection
- Bandwidth monitoring
- CSV logging with timestamp
"""
import socket
import json
import time
import csv
from datetime import datetime
from collections import deque
import statistics
from lerobot.teleoperators.bi_so100_leader.bi_so100_leader import BiSO100Leader
from lerobot.teleoperators.bi_so100_leader.config_bi_so100_leader import BiSO100LeaderConfig

# ========= CONFIG =========
FOLLOWER_IP = "100.78.30.123"  # PC2 IP address
UDP_PORT = 5005
SEND_HZ = 50

LEADER_LEFT_PORT = "COM7"  # Change to your left arm leader COM port
LEADER_RIGHT_PORT = "COM8"  # Change to your right arm leader COM port
LEADER_ID = "leader"

# Measurement settings
LOG_INTERVAL = 1.0  # Log metrics every 1 second
RTT_WINDOW = 100  # Keep last 100 RTT measurements for jitter calculation
TIMEOUT = 0.1  # Socket timeout for RTT measurement (100ms)

# ==========================

# Create CSV log file with timestamp
log_filename = f"teleoperation_control_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_file = open(log_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'Timestamp', 'RTT_ms', 'Jitter_ms', 'Packet_Loss_%', 
    'Bandwidth_kbps', 'Packets_Sent', 'Packets_Lost'
])

# Initialize UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(TIMEOUT)
dt = 1.0 / SEND_HZ

# Initialize bi-SO100 leader (dual arms)
leader_cfg = BiSO100LeaderConfig(
    left_arm_port=LEADER_LEFT_PORT,
    right_arm_port=LEADER_RIGHT_PORT,
    id=LEADER_ID,
)
leader = BiSO100Leader(leader_cfg)
leader.connect()

# Measurement variables
sequence_num = 0
packets_sent = 0
packets_lost = 0
bytes_sent = 0
rtt_measurements = deque(maxlen=RTT_WINDOW)
last_log_time = time.time()

print(f"Leader bi-SO100 initialized on {LEADER_LEFT_PORT} (left) and {LEADER_RIGHT_PORT} (right)")
print(f"Sending commands to {FOLLOWER_IP}:{UDP_PORT} at {SEND_HZ} Hz")
print(f"Logging to: {log_filename}")
print(f"Metrics: RTT, Jitter, Packet Loss, Bandwidth")

try:
    while True:
        loop_start = time.time()
        
        # Get action from leader
        raw_action = leader.get_action()
        action = {key: float(value) for key, value in raw_action.items()}
        
        # Send via UDP with sequence number for RTT measurement
        msg = {
            "ts": time.time(),
            "seq": sequence_num,
            "action": action,
        }
        data = json.dumps(msg).encode("utf-8")
        
        send_time = time.time()
        sock.sendto(data, (FOLLOWER_IP, UDP_PORT))
        bytes_sent += len(data)
        packets_sent += 1
        
        # Try to receive echo for RTT measurement
        try:
            echo_data, _ = sock.recvfrom(256)
            recv_time = time.time()
            echo_msg = json.loads(echo_data.decode("utf-8"))
            
            if echo_msg.get("seq") == sequence_num:
                rtt = (recv_time - send_time) * 1000  # Convert to ms
                rtt_measurements.append(rtt)
        except socket.timeout:
            # No echo received - count as lost packet
            packets_lost += 1
        except Exception:
            packets_lost += 1
        
        sequence_num += 1
        
        # Log metrics periodically
        current_time = time.time()
        if current_time - last_log_time >= LOG_INTERVAL:
            # Calculate metrics
            avg_rtt = statistics.mean(rtt_measurements) if rtt_measurements else 0
            jitter = statistics.stdev(rtt_measurements) if len(rtt_measurements) > 1 else 0
            packet_loss = (packets_lost / packets_sent * 100) if packets_sent > 0 else 0
            bandwidth = (bytes_sent * 8) / (current_time - last_log_time) / 1000  # kbps
            
            # Write to CSV
            csv_writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                f"{avg_rtt:.2f}",
                f"{jitter:.2f}",
                f"{packet_loss:.2f}",
                f"{bandwidth:.2f}",
                packets_sent,
                packets_lost
            ])
            csv_file.flush()
            
            # Print to console
            print(f"RTT: {avg_rtt:.1f}ms | Jitter: {jitter:.1f}ms | Loss: {packet_loss:.1f}% | BW: {bandwidth:.1f}kbps")
            
            # Reset counters
            last_log_time = current_time
            bytes_sent = 0
        
        # Sleep to maintain send rate
        elapsed = time.time() - loop_start
        sleep_time = max(0, dt - elapsed)
        time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nStopping sender...")
finally:
    try:
        leader.disconnect()
    except:
        pass
    sock.close()
    csv_file.close()
    print(f"Leader sender stopped. Log saved to: {log_filename}")
