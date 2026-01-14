"""
WebRTC Signaling Server
Simple WebSocket server to facilitate WebRTC peer connection setup.
Deploy this to Render.com, Railway.app, or any cloud hosting.
"""
import asyncio
import json
import logging
import aiohttp
from aiohttp import web
import aiohttp_cors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store connected peers
peers = {}


async def websocket_handler(request):
    """Handle WebSocket connections for signaling."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    peer_id = None
    
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    msg_type = data.get("type")
                    
                    if msg_type == "register":
                        # Register peer
                        peer_id = data.get("peer_id")
                        peers[peer_id] = ws
                        logger.info(f"Peer registered: {peer_id}")
                        await ws.send_json({"type": "registered", "peer_id": peer_id})
                        
                        # Notify all peers about active peers
                        peer_list = list(peers.keys())
                        for pid, peer_ws in peers.items():
                            await peer_ws.send_json({
                                "type": "peers",
                                "peers": peer_list
                            })
                    
                    elif msg_type in ["offer", "answer", "ice-candidate"]:
                        # Forward signaling messages to target peer
                        target_id = data.get("target")
                        if target_id in peers:
                            await peers[target_id].send_json({
                                "type": msg_type,
                                "from": peer_id,
                                "data": data.get("data")
                            })
                            logger.info(f"Forwarded {msg_type} from {peer_id} to {target_id}")
                        else:
                            logger.warning(f"Target peer {target_id} not found")
                    
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
            
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")
    
    finally:
        # Clean up on disconnect
        if peer_id and peer_id in peers:
            del peers[peer_id]
            logger.info(f"Peer disconnected: {peer_id}")
            
            # Notify remaining peers
            peer_list = list(peers.keys())
            for pid, peer_ws in peers.items():
                try:
                    await peer_ws.send_json({
                        "type": "peers",
                        "peers": peer_list
                    })
                except:
                    pass
    
    return ws


async def health_check(request):
    """Health check endpoint for monitoring."""
    return web.Response(text="OK")


def create_app():
    """Create and configure the aiohttp application."""
    app = web.Application()
    
    # Configure CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*"
        )
    })
    
    # Add routes
    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/health", health_check)
    
    # Configure CORS for all routes
    for route in list(app.router.routes()):
        cors.add(route)
    
    return app


if __name__ == "__main__":
    app = create_app()
    port = 8080  # Render/Railway use PORT env variable
    logger.info(f"Starting signaling server on port {port}")
    web.run_app(app, host="0.0.0.0", port=port)
