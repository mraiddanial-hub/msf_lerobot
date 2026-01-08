"""
PC1 (Leader) - WebRTC Camera Client
Receives camera feed via WebRTC and displays it.
"""
import asyncio
import json
import cv2
import logging
import websockets
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from av import VideoFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========= CONFIG =========
SIGNALING_SERVER = "ws://localhost:8080/ws"  # Change to your deployed server URL
PEER_ID = "leader"
TARGET_PEER = "follower"

# =========================


class WebRTCCameraClient:
    """Manages WebRTC connection and camera receiving."""
    
    def __init__(self, signaling_url, peer_id, target_peer):
        self.signaling_url = signaling_url
        self.peer_id = peer_id
        self.target_peer = target_peer
        self.pc = None
        self.ws = None
        self.video_frames = asyncio.Queue(maxsize=1)
    
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
    
    async def handle_offer(self, offer_data):
        """Handle incoming offer and send answer."""
        self.pc = RTCPeerConnection()
        
        # Handle incoming video track
        @self.pc.on("track")
        async def on_track(track):
            logger.info(f"Receiving {track.kind} track")
            if track.kind == "video":
                asyncio.create_task(self.process_video_track(track))
        
        # Set remote description (offer)
        offer = RTCSessionDescription(
            sdp=offer_data["sdp"],
            type=offer_data["type"]
        )
        await self.pc.setRemoteDescription(offer)
        
        # Create and send answer
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        
        await self.ws.send(json.dumps({
            "type": "answer",
            "target": self.target_peer,
            "data": {
                "sdp": self.pc.localDescription.sdp,
                "type": self.pc.localDescription.type
            }
        }))
        logger.info(f"Sent answer to {self.target_peer}")
    
    async def process_video_track(self, track):
        """Process incoming video frames."""
        logger.info("Started receiving video frames")
        try:
            while True:
                frame = await track.recv()
                
                # Convert av.VideoFrame to numpy array
                img = frame.to_ndarray(format="bgr24")
                
                # Put frame in queue (drop old frame if queue full)
                if self.video_frames.full():
                    try:
                        self.video_frames.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                
                await self.video_frames.put(img)
        
        except Exception as e:
            logger.error(f"Error processing video: {e}")
    
    async def handle_signaling_messages(self):
        """Handle incoming signaling messages."""
        try:
            async for message in self.ws:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "offer":
                    # Received offer from follower
                    logger.info("Received offer from follower")
                    await self.handle_offer(data.get("data"))
                
                elif msg_type == "ice-candidate":
                    # Handle ICE candidates if needed
                    logger.info("Received ICE candidate")
                
                elif msg_type == "peers":
                    peers = data.get("peers", [])
                    logger.info(f"Active peers: {peers}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Signaling connection closed")
    
    async def display_video(self):
        """Display received video frames."""
        logger.info("Starting video display...")
        while True:
            try:
                # Get frame from queue with timeout
                frame = await asyncio.wait_for(self.video_frames.get(), timeout=1.0)
                
                # Display frame
                cv2.imshow("Follower Camera (WebRTC)", frame)
                
                # Process OpenCV events
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit signal received")
                    break
            
            except asyncio.TimeoutError:
                # No frame received in 1 second - continue waiting
                cv2.waitKey(1)  # Keep window responsive
                continue
            
            except Exception as e:
                logger.error(f"Display error: {e}")
                break
        
        cv2.destroyAllWindows()
    
    async def run(self):
        """Main run loop."""
        try:
            await self.connect_signaling()
            
            # Start tasks concurrently
            await asyncio.gather(
                self.handle_signaling_messages(),
                self.display_video()
            )
        
        except Exception as e:
            logger.error(f"Error: {e}")
        
        finally:
            if self.pc:
                await self.pc.close()
            if self.ws:
                await self.ws.close()
            cv2.destroyAllWindows()
            logger.info("Camera client stopped")


async def main():
    """Entry point."""
    logger.info("Starting WebRTC Camera Client (Leader)")
    logger.info(f"Signaling: {SIGNALING_SERVER}")
    logger.info("Press 'q' in video window to quit")
    
    client = WebRTCCameraClient(SIGNALING_SERVER, PEER_ID, TARGET_PEER)
    await client.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped by user")
