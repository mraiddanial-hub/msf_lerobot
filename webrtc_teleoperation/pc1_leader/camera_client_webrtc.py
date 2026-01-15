"""
PC1 (Leader) - WebRTC Camera Client
Receives camera feed via WebRTC and displays it.

Features:
- Video latency measurement
- Frame rate tracking
- Bandwidth monitoring
- CSV logging with timestamp
"""
import asyncio
import json
import cv2
import logging
import websockets
import numpy as np
import csv
import time
from datetime import datetime
from collections import deque
import statistics
from aiortc import RTCPeerConnection, RTCSessionDescription
from av import VideoFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========= CONFIG =========
SIGNALING_SERVER = "ws://localhost:8080/ws"  # Change to your deployed server URL
PEER_ID = "leader"
TARGET_PEER = "follower"

# Measurement settings
LOG_INTERVAL = 1.0  # Log metrics every 1 second
LATENCY_WINDOW = 100  # Keep last 100 latency measurements

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
        
        # Measurement tracking
        self.frame_count = 0
        self.frames_received = 0
        self.frames_dropped = 0
        self.bytes_received = 0
        self.latency_measurements = deque(maxlen=LATENCY_WINDOW)
        self.last_log_time = time.time()
        
        # Create CSV log file
        log_filename = f"teleoperation_video_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.csv_file = open(log_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'Timestamp', 'Video_Latency_ms', 'Jitter_ms', 'FPS', 
            'Frames_Dropped_%', 'Bandwidth_kbps'
        ])
        logger.info(f"Logging to: {log_filename}")
    
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
                recv_time = time.time()
                frame = await track.recv()
                
                # Calculate latency (frame.time is in seconds since epoch)
                if hasattr(frame, 'time') and frame.time:
                    latency = (recv_time - frame.time) * 1000  # Convert to ms
                    self.latency_measurements.append(latency)
                
                # Convert av.VideoFrame to numpy array
                img = frame.to_ndarray(format="bgr24")
                
                # Track bytes (approximate)
                self.bytes_received += img.nbytes
                self.frames_received += 1
                
                # Put frame in queue (drop old frame if queue full)
                if self.video_frames.full():
                    try:
                        self.video_frames.get_nowait()
                        self.frames_dropped += 1
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
        frames_displayed = 0
        display_start = time.time()
        
        while True:
            try:
                # Get frame from queue with timeout
                frame = await asyncio.wait_for(self.video_frames.get(), timeout=1.0)
                frames_displayed += 1
                self.frame_count += 1
                
                # Calculate and log metrics periodically
                current_time = time.time()
                if current_time - self.last_log_time >= LOG_INTERVAL:
                    elapsed = current_time - self.last_log_time
                    
                    # Calculate metrics
                    avg_latency = statistics.mean(self.latency_measurements) if self.latency_measurements else 0
                    jitter = statistics.stdev(self.latency_measurements) if len(self.latency_measurements) > 1 else 0
                    fps = self.frame_count / elapsed
                    drop_rate = (self.frames_dropped / self.frames_received * 100) if self.frames_received > 0 else 0
                    bandwidth = (self.bytes_received * 8) / elapsed / 1000  # kbps
                    
                    # Write to CSV
                    self.csv_writer.writerow([
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                        f"{avg_latency:.2f}",
                        f"{jitter:.2f}",
                        f"{fps:.1f}",
                        f"{drop_rate:.2f}",
                        f"{bandwidth:.2f}"
                    ])
                    self.csv_file.flush()
                    
                    # Print to console
                    logger.info(f"Video - Latency: {avg_latency:.1f}ms | Jitter: {jitter:.1f}ms | FPS: {fps:.1f} | Drop: {drop_rate:.1f}% | BW: {bandwidth:.1f}kbps")
                    
                    # Reset counters
                    self.last_log_time = current_time
                    self.frame_count = 0
                    self.bytes_received = 0
                
                # Add metrics overlay on frame
                if self.latency_measurements:
                    avg_latency = statistics.mean(self.latency_measurements)
                    fps = frames_displayed / (current_time - display_start) if (current_time - display_start) > 0 else 0
                    
                    cv2.putText(frame, f"Latency: {avg_latency:.1f}ms", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
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
            if hasattr(self, 'csv_file'):
                self.csv_file.close()
                logger.info(f"Video log saved")
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
