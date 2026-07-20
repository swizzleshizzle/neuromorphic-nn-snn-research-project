## Live monitoring (Phase 2)

Watch a running sim stream into the dashboard via the file-tail WebSocket server.

1. Simulate a live sim (or run a real experiment that writes JSONL via FileSink):
   `python scripts/replay_into_file.py dashboard/public/week11_dashboard_trace.jsonl outputs/live.jsonl --delay 0.2`
2. Start the server (needs `pip install -e ".[server]"`):
   `python -m neuromorphic.server --trace outputs/live.jsonl`
3. Start the dashboard pointed at the socket:
   `VITE_WS_URL=ws://localhost:8000/stream npm run dev`

Frames stream in, the playhead follows the tail, and the TopBar LIVE badge shows
`live / reconnecting / ended / error`. Without `VITE_WS_URL`, the dashboard loads a static
trace file exactly as before.
