#!/bin/bash
# WildlifeCam — start/stop/status helper

cd "$(dirname "$0")"
source venv/bin/activate

ACTION="${1:-start}"
EXPERIMENT="sep2026_48hr"

case "$ACTION" in
  start)
    # Kill any existing run
    pkill -f "wildlife.main" 2>/dev/null
    sleep 1
    echo "Starting WildlifeCam (experiment: $EXPERIMENT)..."
    nohup python -u -m wildlife.main \
        --live \
        --rotate 180 \
        --experiment "$EXPERIMENT" \
        > wildlife.log 2>&1 &
    echo "PID: $!"
    echo "Running in background. Check with: ./wildlife_start.sh status"
    ;;
  stop)
    echo "Stopping WildlifeCam..."
    pkill -f "wildlife.main"
    echo "Stopped."
    ;;
  status)
    if pgrep -f "wildlife.main" > /dev/null; then
      echo "WildlifeCam is RUNNING (PID: $(pgrep -f 'wildlife.main'))"
    else
      echo "WildlifeCam is NOT running"
    fi
    echo ""
    echo "=== Disk ==="
    df -h .
    echo ""
    echo "=== Images ==="
    MOTION=$(find "data/wildlife/$EXPERIMENT/wildlife_motion" -name '*.jpg' 2>/dev/null | wc -l)
    HEARTBEAT=$(find "data/wildlife/$EXPERIMENT/heartbeat" -name '*.jpg' 2>/dev/null | wc -l)
    echo "Motion captures: $MOTION"
    echo "Heartbeats:      $HEARTBEAT"
    echo ""
    echo "=== Last 5 log lines ==="
    tail -5 wildlife.log 2>/dev/null || echo "No log yet"
    ;;
  log)
    tail -f wildlife.log
    ;;
  *)
    echo "Usage: ./wildlife_start.sh [start|stop|status|log]"
    ;;
esac
