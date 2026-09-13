#!/bin/bash
# Wildlife Cam with Vision Server — run on Raspberry Pi
# Usage:
#   ./wildlife_vision.sh start    — start in background
#   ./wildlife_vision.sh stop     — stop background process
#   ./wildlife_vision.sh status   — check if running
#   ./wildlife_vision.sh logs     — tail the log
#   ./wildlife_vision.sh run      — run in foreground (see output live)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/wildlife_cam.pid"
LOG_FILE="$SCRIPT_DIR/wildlife_cam.log"

# ---- CONFIGURE THESE ----
VISION_SERVER="http://10.0.0.81:8000"
VISION_API_KEY="wildlife-vision-secret-2026"
EXPERIMENT="wildlife002"
# --------------------------

CMD="python -u -m wildlife.main \
  --live \
  --rotate 180 \
  --experiment $EXPERIMENT \
  --capture-interval 3 \
  --heartbeat-minutes 30 \
  --min-solidity 0.2 \
  --confirm-frames 1 \
  --use-classifier \
  --classifier-model yolov8n.onnx \
  --classifier-confidence 0.35 \
  --vision-server $VISION_SERVER \
  --vision-api-key $VISION_API_KEY"

start_cam() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Already running (PID $PID)"
            return
        fi
    fi
    echo "Starting Wildlife Cam..."
    echo "  Vision server: $VISION_SERVER"
    echo "  Experiment:    $EXPERIMENT"
    cd "$SCRIPT_DIR"
    nohup $CMD > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Started (PID $(cat "$PID_FILE"))"
    echo "Logs: tail -f $LOG_FILE"
}

stop_cam() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            echo "Stopped (PID $PID)"
        else
            echo "Was not running (stale PID)"
        fi
        rm -f "$PID_FILE"
    else
        echo "No PID file. Killing any wildlife.main processes..."
        pkill -f "wildlife.main" && echo "Killed." || echo "Nothing running."
    fi
}

case "$1" in
    start)  start_cam ;;
    stop)   stop_cam ;;
    restart) stop_cam; sleep 2; start_cam ;;
    run)
        echo "Running in foreground (Ctrl+C to stop)..."
        echo "  Vision server: $VISION_SERVER"
        cd "$SCRIPT_DIR"
        $CMD
        ;;
    status)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "RUNNING (PID $(cat "$PID_FILE"))"
        else
            echo "NOT running."
        fi
        ;;
    logs)
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "No log file yet. Start the cam first."
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|run|status|logs}"
        echo ""
        echo "  start   — run in background"
        echo "  stop    — stop background process"
        echo "  restart — stop then start"
        echo "  run     — foreground (see output live)"
        echo "  status  — check if running"
        echo "  logs    — tail background log"
        ;;
esac
