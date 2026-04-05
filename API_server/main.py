# main.py
# FastAPI server for Vercel, serving teaching video generation API
# Only returns video_url for now, pointing to a static file

import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Serve the output directory as static files (for video download)
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output'))
VIDEO_FILENAME = 'my_video.mp4'
VIDEO_PATH = os.path.join(OUTPUT_DIR, VIDEO_FILENAME)

# Mount /static for file serving
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

@app.post("/v1/video/generate")
async def generate_video(request: Request):
    # Parse JSON input (request_id, course_requirement, student_persona)
    data = await request.json()
    # In this baseline, we always return the same video file
    video_url = request.url_for("static", path=VIDEO_FILENAME)
    # Compose response (only video_url for now)
    return JSONResponse({
        "video_url": str(video_url),
        # "subtitle_url": None,  # Optional, not implemented
        # "supplementary_url": []  # Optional, not implemented
    })

# For Vercel compatibility (entrypoint)
# vercel.json should point to "main.handler"
handler = app
