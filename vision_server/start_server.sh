#!/bin/bash
# Wildlife Vision Server — start/stop/status
# Usage:
#   ./start_server.sh start    — start server in background
#   ./start_server.sh stop     — stop the background server
#   ./start_server.sh status   — check if server is running
#   ./start_server.sh logs     — tail the log file

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/server.pid"
LOG_FILE="$SCRIPT_DIR/server.log"

start_server() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Server already running (PID $PID)"
            return
        fi
    fi

    echo "Starting Wildlife Vision Server..."
    cd "$SCRIPT_DIR"
    nohup python -u server.py > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Server started (PID $(cat "$PID_FILE"))"
    echo "Log file: $LOG_FILE"
}

stop_server() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            echo "Server stopped (PID $PID)"
        else
            echo "Server was not running (stale PID)"
        fi
        rm -f "$PID_FILE"
    else
        echo "No PID file. Killing any server.py processes..."
        pkill -f "python.*server.py" && echo "Killed." || echo "Nothing to kill."
    fi
}

case "$1" in
    start)   start_server ;;
    stop)    stop_server ;;
    restart) stop_server; sleep 2; start_server ;;
    status)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "Server is RUNNING (PID $(cat "$PID_FILE"))"
        else
            echo "Server is NOT running."
        fi
        ;;
    logs)
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "No log file found."
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|status|logs|restart}"
        ;;
esac
