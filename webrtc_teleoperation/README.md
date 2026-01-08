# WebRTC Teleoperation System

Complete WebRTC-based teleoperation system for SO-101 robot with adaptive video streaming over 4G/5G networks.

## Architecture

```
PC1 (Leader)                                    PC2 (Follower)
┌─────────────────┐                            ┌─────────────────┐
│ Camera Client   │◄──WebRTC (adaptive H.264)──│ Camera Server   │
│ (WebRTC)        │   via signaling server     │ (WebRTC)        │
└─────────────────┘                            └─────────────────┘
        │                                               │
┌─────────────────┐                            ┌─────────────────┐
│ SO-101 Leader   │─────UDP (commands)────────►│ SO-101 Follower │
│ Sender          │    Direct IP connection    │ Receiver        │
└─────────────────┘                            └─────────────────┘
```

**Key Benefits:**
- **WebRTC for camera**: Adaptive bitrate, low latency (50-200ms), handles 4G/5G transitions
- **UDP for robot control**: Simple, proven, low-latency commands
- **Separate streams**: No bandwidth competition

## Quick Start

### Step 1: Deploy Signaling Server (One-time setup)

#### Option A: Local Testing (Same Network)
```bash
cd signaling_server
pip install -r requirements.txt
python server.py
```
Use `ws://YOUR_LOCAL_IP:8080/ws` in camera scripts.

#### Option B: Deploy to Render.com (Internet Access)
See [signaling_server/README.md](signaling_server/README.md) for detailed deployment instructions.

After deployment, you'll get a URL like: `https://your-app.onrender.com`

### Step 2: Configure Camera Scripts

Edit both camera scripts to use your signaling server:

**In `pc1_leader/camera_client_webrtc.py`:**
```python
SIGNALING_SERVER = "wss://your-app.onrender.com/ws"  # or ws://local-ip:8080/ws
```

**In `pc2_follower/camera_server_webrtc.py`:**
```python
SIGNALING_SERVER = "wss://your-app.onrender.com/ws"  # or ws://local-ip:8080/ws
CAMERA_INDEX = 0  # Change to your camera index
```

### Step 3: Configure Robot Control Scripts

**In `pc1_leader/leader_sender_udp.py`:**
```python
FOLLOWER_IP = "100.127.72.97"  # PC2's IP address
LEADER_PORT = "COM9"  # Your leader SO-101 COM port
```

**In `pc2_follower/robot_receiver_udp.py`:**
```python
FOLLOWER_PORT = "COM10"  # Your follower SO-101 COM port
```

### Step 4: Install Dependencies

**On PC1 (Leader):**
```bash
cd pc1_leader
pip install -r requirements.txt
```

**On PC2 (Follower):**
```bash
cd pc2_follower
pip install -r requirements.txt
```

### Step 5: Run the System

**On PC2 (Follower) - Start these FIRST:**
```bash
# Terminal 1: Camera server
python camera_server_webrtc.py

# Terminal 2: Robot receiver
python robot_receiver_udp.py
```

**On PC1 (Leader):**
```bash
# Terminal 1: Camera client
python camera_client_webrtc.py

# Terminal 2: Robot control sender
python leader_sender_udp.py
```

## Usage

1. **Start PC2 scripts first** (camera server and robot receiver)
2. **Then start PC1 scripts** (camera client will connect to server)
3. Camera window will open on PC1 showing follower's view
4. Move the leader SO-101 arm - follower will follow
5. Press 'q' in camera window to quit

## Troubleshooting

### Camera not connecting
- Check signaling server is running (visit health check URL)
- Verify both scripts use same signaling server URL
- Check firewall settings
- Look at console logs for connection errors

### High latency on camera
- WebRTC should auto-adapt, but check:
  - Network quality (ping between PCs)
  - Camera resolution/FPS settings in `camera_server_webrtc.py`
  - Try reducing: `FRAME_WIDTH = 320`, `FRAME_HEIGHT = 240`, `FPS = 20`

### Robot control not working
- Verify UDP control scripts have correct:
  - IP addresses
  - COM ports
  - Robot is powered on and connected

### Signaling connection fails
- For local testing: Use `ws://` not `wss://`
- For deployed server: Use `wss://` not `ws://`
- Check signaling server logs

## Network Requirements

**Minimum bandwidth for smooth operation:**
- Camera (WebRTC): 500 Kbps - 5 Mbps (adapts automatically)
- Robot control (UDP): ~10 Kbps (negligible)

**Tested on:**
- ✅ Same WiFi network
- ✅ Different WiFi networks
- ✅ 4G LTE
- ✅ 5G
- ✅ Mixed (4G + 5G)

## Advanced Configuration

### Adjust Video Quality
Edit `camera_server_webrtc.py`:
```python
FRAME_WIDTH = 640   # Lower for less bandwidth
FRAME_HEIGHT = 480
FPS = 30            # Lower for less bandwidth
```

### Change Robot Control Rate
Edit `leader_sender_udp.py`:
```python
SEND_HZ = 50  # Default 50Hz, can go up to 100Hz
```

### Use Different STUN Servers
WebRTC uses Google's STUN server by default. To use different ones, edit both camera scripts and add ICE servers configuration to `RTCPeerConnection()`.

## File Structure

```
webrtc_teleoperation/
├── signaling_server/      # Deploy to cloud (one-time)
│   ├── server.py
│   ├── requirements.txt
│   └── README.md
├── pc1_leader/            # Run on leader PC
│   ├── camera_client_webrtc.py
│   ├── leader_sender_udp.py
│   └── requirements.txt
├── pc2_follower/          # Run on follower PC
│   ├── camera_server_webrtc.py
│   ├── robot_receiver_udp.py
│   └── requirements.txt
└── README.md             # This file
```

## Replicating to Multiple PCs

To set up on new PCs:
1. Copy entire `webrtc_teleoperation` folder to new PC
2. Edit IP addresses and COM ports in scripts
3. Install dependencies: `pip install -r requirements.txt`
4. Run scripts as described above

**Note:** All PCs can share the same signaling server - no need to deploy multiple times.

## Performance Comparison

| Method | Latency | Bandwidth Efficiency | Stability on 4G/5G |
|--------|---------|---------------------|-------------------|
| TCP + JPEG | 500-2000ms | Poor | Poor |
| UDP + JPEG | 100-500ms | Poor | Fair |
| **WebRTC** | **50-200ms** | **Excellent** | **Excellent** |

## Support

For issues:
1. Check console logs on both PCs
2. Verify signaling server is running
3. Test with local signaling server first
4. Check firewall/antivirus settings

## License

Same as LeRobot project.
