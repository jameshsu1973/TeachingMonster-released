# API Server for Teaching Video Generation

This FastAPI server exposes a POST endpoint `/v1/video/generate` for the competition platform.

## Features
- Accepts JSON input with `request_id`, `course_requirement`, `student_persona`.
- Returns a JSON with `video_url` (points to a static MP4 file).
- Serves the video file from the `/static/` route for direct download.
- Ready for Vercel deployment (see vercel.json).

## Usage (Local Test)
1. Place your generated video at `output/final_video_debug_test.mp4` (relative to project root).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the server:
   ```
   uvicorn main:app --reload
   ```
4. Test POST request:
   ```
   curl -X POST http://localhost:8000/v1/video/generate -H "Content-Type: application/json" -d '{"request_id":"1","course_requirement":"...","student_persona":"..."}'
   ```

## Deployment (Vercel)
- vercel.json is preconfigured for Python/ASGI (FastAPI).
- The `/static` route exposes the output directory for video download.

## Output Example
```
{
  "video_url": "https://<your-vercel-domain>/static/final_video_debug_test.mp4"
}
```

## Notes
- Only `video_url` is returned for now. Subtitle and supplementary URLs can be added later.
- The video file must exist and be accessible for download after response.

## Local One-Click Test (direct_test.sh)

You can use the provided `direct_test.sh` script for a one-click local test:

1. **Make the script executable** (only needed once):
   ```bash
   chmod +x direct_test.sh
   ```
2. **Run the script**:
   ```bash
   ./direct_test.sh
   ```
   - This will automatically install Python dependencies, check ngrok, launch the FastAPI server, and open an ngrok tunnel.
   - On first use, if ngrok is not authenticated, you will see a message like:
     ```
     ngrok authtoken not found.
     Please sign up at https://dashboard.ngrok.com/signup, copy your authtoken, and run:
       ngrok config add-authtoken <your-token>
     ```
   - After authentication, the script will display a URL like:
     ```
     https://xxxx.ngrok-free.dev/v1/video/generate
     ```
     This is the endpoint you should provide to the competition platform.

3. **Test the API**
   You can use curl or Postman to test:
   ```bash
   curl -X POST https://xxxx.ngrok-free.dev/v1/video/generate \
     -H "Content-Type: application/json" \
     -d '{"request_id":"1","course_requirement":"...","student_persona":"..."}'
   ```

4. **Stop the service**
   Press Ctrl+C to stop both ngrok and the FastAPI server.

---

## ngrok Setup Guide

1. **Sign up for a free ngrok account:**
   - Go to https://dashboard.ngrok.com/signup and register.
2. **Get your authtoken:**
   - After logging in, visit https://dashboard.ngrok.com/get-started/your-authtoken
   - Copy your unique authtoken string.
3. **Authenticate ngrok on your machine:**
   - Run the following command in your terminal (replace <your-token> with your actual token):
     ```bash
     ngrok config add-authtoken <your-token>
     ```
   - You only need to do this once per machine.
4. **Verify ngrok installation:**
   - Run `ngrok version` to check if ngrok is installed and authenticated.

---

## Dependencies
- `fastapi`, `uvicorn`: Required for the API server.
- `requests`: Sometimes used for local test scripts or health checks.
- `ngrok`: Used for exposing your local server to the public internet for platform testing.

---

## Notes
- The file `output/final_video_debug_test.mp4` must exist in the `output/` directory at the project root.
- If you encounter ngrok authtoken issues, follow the setup guide above.
- Free ngrok accounts have bandwidth and connection limits, suitable for local testing only.
