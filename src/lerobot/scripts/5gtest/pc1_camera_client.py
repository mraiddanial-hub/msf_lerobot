# pc1_camera_client.py
import socket
import cv2
import struct
import numpy as np

# ====== CONFIG ======
FOLLOWER_IP = "100.107.126.39"   # <-- IP of PC2 (same as in your teleop sender)
SERVER_PORT = 6000

# ====================
def recvall(sock, count):
    """Receive exactly 'count' bytes from the socket."""
    buf = b""
    while len(buf) < count:
        newbuf = sock.recv(count - len(buf))
        if not newbuf:
            return None
        buf += newbuf
    return buf

print(f"Connecting to camera server at {FOLLOWER_IP}:{SERVER_PORT}...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((FOLLOWER_IP, SERVER_PORT))
print("Connected to camera server.")

try:
    while True:
        # Read 4-byte length header
        header = recvall(sock, 4)
        if not header:
            print("Connection closed by server")
            break

        size = struct.unpack(">I", header)[0]

        # Read the JPEG data
        data = recvall(sock, size)
        if data is None:
            print("Failed to receive frame data")
            break

        # Decode JPEG to image
        np_data = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        # Show on leader PC (if GUI available)
        try:
            cv2.imshow("Follower Camera (from PC2)", frame)
            # Press 'q' to quit the viewer
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit signal received, closing viewer.")
                break
        except cv2.error:
            # If GUI not available, just print frame info
            print(f"Received frame: {frame.shape}")

finally:
    sock.close()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass  # GUI not available, ignore
