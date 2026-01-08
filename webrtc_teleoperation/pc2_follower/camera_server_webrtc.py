"""
PC2 (Follower) - WebRTC Camera Server
Streams camera feed via WebRTC with adaptive bitrate.
"""
import asyncio
import json
import cv2
import logging
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer
from av import VideoFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========= CONFIG =========
SIGNALING_SERVER = "ws://localhost:8080/ws"  # Change to your deployed server URL
CAMERA_INDEX = 0  # Change to your camera index
PEER_ID = "follower"
TARGET_PEER = "leader"

# Video settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# =========================


class CameraVideoTrack(VideoStreamTrack):
    """Custom video track that captures from OpenCV camera."""
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera = cv2.VideoCapture(camera_index)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, FPS)
        
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_index}")
        
        logger.info(f"Camera opened: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FPS}fps")
    
    async def recv(self):
        """Capture and return a video frame."""
        pts, time_base = await self.next_timestamp()
        
        ret, frame = self.camera.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            return None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create VideoFrame
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        
        return video_frame
    
    def __del__(self):
        if hasattr(self, 'camera'):
            self.camera.release()


class WebRTCCameraServer:
    """Manages WebRTC connection and camera streaming."""
    
    def __init__(self, signaling_url, peer_id, target_peer):
        self.signaling_url = signaling_url
        self.peer_id = peer_id
        self.target_peer = target_peer
        self.pc = None
        self.ws = None
        self.video_track = None
    
    async def connect_signaling(self):
        """Connect to signaling server."""
        logger.info(f"Connecting to signaling server: {self.signaling_url}")
        self.ws = await websockets.connect(self.signaling_url)
        
        # Register with server
        await self.ws.send(json.dumps({
            "type": "register",
            "peer_id": self.peer_id
        }))
        
        response = await self.ws.recv()
        data = json.loads(response)
        if data.get("type") == "registered":
            logger.info(f"Registered as: {self.peer_id}")
    
    async def create_offer(self):
        """Create and send WebRTC offer."""
        self.pc = RTCPeerConnection()
        
        # Add video track
        self.video_track = CameraVideoTrack(CAMERA_INDEX)
        self.pc.addTrack(self.video_track)
        logger.info("Added camera track to peer connection")
        
        # Create offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        
        # Send offer to target peer
        await self.ws.send(json.dumps({
            "type": "offer",
            "target": self.target_peer,
            "data": {
                "sdp": self.pc.localDescription.sdp,
                "type": self.pc.localDescription.type
            }
        }))
        logger.info(f"Sent offer to {self.target_peer}")
    
    async def handle_signaling_messages(self):
        """Handle incoming signaling messages."""
        try:
            async for message in self.ws:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "answer":
                    # Received answer from leader
                    answer_data = data.get("data")
                    answer = RTCSessionDescription(
                        sdp=answer_data["sdp"],
                        type=answer_data["type"]
                    )
                    await self.pc.setRemoteDescription(answer)
                    logger.info("Received and set answer - WebRTC connection established!")
                
                elif msg_type == "ice-candidate":
                    # Handle ICE candidates if needed
                    logger.info("Received ICE candidate")
                
                elif msg_type == "peers":
                    peers = data.get("peers", [])
                    logger.info(f"Active peers: {peers}")
                    if self.target_peer in peers and self.pc is None:
                        # Target peer is available, create offer
                        await self.create_offer()
        
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Signaling connection closed")
    
    async def run(self):
        """Main run loop."""
        try:
            await self.connect_signaling()
            
            # Wait a moment for peer list
            await asyncio.sleep(2)
            
            # If target peer already connected, send offer
            await self.create_offer()
            
            # Handle signaling messages
            await self.handle_signaling_messages()
        
        except Exception as e:
            logger.error(f"Error: {e}")
        
        finally:
            if self.pc:
                await self.pc.close()
            if self.ws:
                await self.ws.close()
            if self.video_track:
                del self.video_track
            logger.info("Camera server stopped")


async def main():
    """Entry point."""
    logger.info("Starting WebRTC Camera Server (Follower)")
    logger.info(f"Signaling: {SIGNALING_SERVER}")
    logger.info(f"Camera: {CAMERA_INDEX}")
    
    server = WebRTCCameraServer(SIGNALING_SERVER, PEER_ID, TARGET_PEER)
    await server.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped by user")
