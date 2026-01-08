# WebRTC Signaling Server

This server facilitates the initial WebRTC handshake between peers. It only handles connection setup - actual video streams go directly between PCs.

## Deployment Options

### Option 1: Render.com (Recommended - Free & Easy)

1. Create account at [render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repo OR use "Deploy from Git URL"
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python server.py`
   - **Instance Type**: Free
5. Click "Create Web Service"
6. Copy the URL (e.g., `https://your-app.onrender.com`)
7. Use this URL in your camera scripts (replace `ws://` with `wss://`)

### Option 2: Railway.app

1. Create account at [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select this folder
4. Railway auto-detects Python and deploys
5. Copy the provided URL

### Option 3: Local Testing

For testing on your local network:

```bash
pip install -r requirements.txt
python server.py
```

Then use `ws://YOUR_LOCAL_IP:8080/ws` in camera scripts.

## Usage

Once deployed, you'll get a URL like:
- **Render**: `https://your-signaling-server.onrender.com`
- **Railway**: `https://your-app.railway.app`

Use this in your camera scripts:
```python
SIGNALING_SERVER = "wss://your-signaling-server.onrender.com/ws"
```

## Health Check

Visit `https://your-server-url/health` to verify the server is running.

## Logs

Check your hosting platform's dashboard to view connection logs and debug issues.
