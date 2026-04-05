# Kill any previous uvicorn or ngrok processes
pkill -f 'uvicorn main:app' 2>/dev/null || true
pkill -f 'ngrok http' 2>/dev/null || true
#!/bin/bash
# direct_test.sh
# One-click script to set up environment, install dependencies, launch FastAPI server and ngrok for local API testing.

set -e

# 1. Check and install Python dependencies
REQ_FILE="$(dirname "$0")/requirements.txt"
if ! command -v pip &> /dev/null; then
    echo "pip not found. Please install Python and pip first."
    exit 1
fi

echo "[1/4] Installing Python dependencies..."
pip install --upgrade pip
pip install -r "$REQ_FILE"

# 2. Check and install ngrok if not present
if ! command -v ngrok &> /dev/null; then
    echo "[2/4] ngrok not found. Installing ngrok..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        wget -O ngrok.zip https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
        unzip ngrok.zip
        chmod +x ngrok
        sudo mv ngrok /usr/local/bin/
        rm ngrok.zip
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install --cask ngrok
    elif [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* ]]; then
        echo "Please manually install ngrok from https://ngrok.com/download and add it to PATH."
        exit 1
    else
        echo "Unsupported OS. Please install ngrok manually."
        exit 1
    fi
else
    echo "[2/4] ngrok already installed."
fi


# 3. Check ngrok authtoken in both common config locations
NGROK_AUTH_FOUND=0
if grep -q 'authtoken:' ~/.ngrok2/ngrok.yml 2>/dev/null; then
    NGROK_AUTH_FOUND=1
elif grep -q 'authtoken:' ~/.config/ngrok/ngrok.yml 2>/dev/null; then
    NGROK_AUTH_FOUND=1
fi
if [ "$NGROK_AUTH_FOUND" -eq 0 ]; then
    echo "[3/4] ngrok authtoken not found."
    echo "Please sign up at https://dashboard.ngrok.com/signup, copy your authtoken, and run:"
    echo "  ngrok config add-authtoken <your-token>"
    exit 1
else
    echo "[3/4] ngrok authtoken found."
fi

# 4. Launch FastAPI server (background)
echo "[4/4] Launching FastAPI server..."
cd "$(dirname "$0")"
uvicorn main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!
sleep 2

# 5. Launch ngrok (foreground) and print endpoint
echo "[INFO] Launching ngrok tunnel for http://localhost:8000 ..."
echo "[INFO] Press Ctrl+C to stop ngrok and the FastAPI server."

# Try normal ngrok first (redirect both stdout and stderr)
ngrok http 8000 --log=stdout > ngrok.log 2>&1 &
NGROK_PID=$!
sleep 5

# If ngrok.log is still empty, try snap run ngrok (for snap installs)
if [ ! -s ngrok.log ]; then
    echo "[WARN] ngrok.log is empty, trying snap run ngrok..."
    snap run ngrok http 8000 --log=stdout > ngrok.log 2>&1 &
    NGROK_PID=$!
    sleep 5
fi

# Wait for ngrok public URL (max 20s)
NGROK_URL=""
for i in {1..20}; do
    sleep 1
    NGROK_URL=$(grep -oE 'https://[a-zA-Z0-9\-]+\.ngrok(-free)?\.dev' ngrok.log | head -n 1)
    if [ -n "$NGROK_URL" ]; then
        break
    fi
done

if [ -n "$NGROK_URL" ]; then
    echo "[SUCCESS] Your public API endpoint is:"
    echo "    $NGROK_URL/v1/video/generate"
else
    echo "[ERROR] Could not detect ngrok public URL after waiting. Please check ngrok.log."
fi

# Wait for ngrok to exit (user Ctrl+C)
wait $NGROK_PID

# Cleanup: kill FastAPI server when ngrok exits
kill $SERVER_PID
rm -f ngrok.log
