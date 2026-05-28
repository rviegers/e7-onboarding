import json
import queue
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, request, jsonify
import pandas as pd
from sklearn.metrics import root_mean_squared_error

import numpy as np
from sklearn.metrics import average_precision_score


def compute_map(targets: np.ndarray, preds: np.ndarray) -> float:
    """Mean Average Precision across classes (macro, label-wise)."""
    return average_precision_score(targets, preds, average="macro")
app = Flask(__name__)

_gt_df = pd.read_csv('answers.csv').sort_values('row_id')
SPECIES_COLS = [c for c in _gt_df.columns if c != 'row_id']
GROUND_TRUTH = _gt_df
SCORES_FILE = Path('scores.json')

scores_lock = threading.Lock()
sse_clients: list[queue.Queue] = []
sse_clients_lock = threading.Lock()


def load_scores() -> list[dict]:
    if SCORES_FILE.exists():
        return json.loads(SCORES_FILE.read_text())
    return []


def save_scores(scores: list[dict]) -> None:
    SCORES_FILE.write_text(json.dumps(scores, indent=2))


def push_sse_update(scores: list[dict]) -> None:
    data = f"data: {json.dumps(scores)}\n\n"
    with sse_clients_lock:
        for q in sse_clients:
            q.put(data)


@app.route('/submit', methods=['POST'])
def submit():
    file = request.files['file']
    user_name = request.form.get('name', 'Anonymous')

    user_preds = pd.read_csv(file).sort_values('row_id')
    score = compute_map(GROUND_TRUTH[SPECIES_COLS].values, user_preds[SPECIES_COLS].values)
    score = round(float(score), 4)

    entry = {
        "name": user_name,
        "score": score,
        "submitted_at": datetime.now().strftime("%H:%M:%S"),
    }

    with scores_lock:
        scores = load_scores()
        existing = next((s for s in scores if s['name'] == user_name), None)
        if existing is None or score > existing['score']:
            scores = [s for s in scores if s['name'] != user_name]
            scores.append(entry)
            save_scores(scores)
        sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)

    push_sse_update(sorted_scores)
    print(f"Submission from {user_name}: {score}")
    return jsonify({"score": score, "message": "Nice work!"})


@app.route('/api/scores')
def api_scores():
    with scores_lock:
        scores = load_scores()
    return jsonify(sorted(scores, key=lambda x: x['score'], reverse=True))


@app.route('/api/scores/stream')
def scores_stream():
    client_queue: queue.Queue = queue.Queue()

    with sse_clients_lock:
        sse_clients.append(client_queue)

    def generate():
        try:
            with scores_lock:
                scores = load_scores()
            yield f"data: {json.dumps(sorted(scores, key=lambda x: x['score'], reverse=True))}\n\n"
            while True:
                data = client_queue.get()
                yield data
        finally:
            with sse_clients_lock:
                sse_clients.remove(client_queue)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/leaderboard')
def leaderboard():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Leaderboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 48px 24px;
    }
    h1 {
      font-size: 2rem;
      font-weight: 700;
      margin-bottom: 8px;
      letter-spacing: -0.5px;
    }
    #status {
      font-size: 0.8rem;
      color: #64748b;
      margin-bottom: 36px;
    }
    #status.live::before {
      content: "";
      display: inline-block;
      width: 8px; height: 8px;
      background: #22c55e;
      border-radius: 50%;
      margin-right: 6px;
      animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.3; }
    }
    table {
      width: 100%;
      max-width: 640px;
      border-collapse: collapse;
    }
    thead th {
      text-align: left;
      padding: 10px 16px;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #64748b;
      border-bottom: 1px solid #1e293b;
    }
    tbody tr {
      border-bottom: 1px solid #1e293b;
      transition: background 0.15s;
    }
    tbody tr:hover { background: #1e293b; }
    tbody tr.highlight { animation: flash 0.8s ease-out; }
    @keyframes flash {
      0% { background: #1d4ed8; }
      100% { background: transparent; }
    }
    td {
      padding: 14px 16px;
      font-size: 0.95rem;
    }
    td.rank { color: #64748b; width: 48px; font-variant-numeric: tabular-nums; }
    td.name { font-weight: 600; }
    td.score { font-variant-numeric: tabular-nums; font-family: monospace; font-size: 1rem; }
    td.time { color: #64748b; font-size: 0.8rem; }
    .medal { margin-right: 6px; }
    #empty {
      color: #475569;
      margin-top: 48px;
      font-size: 0.95rem;
      display: none;
    }
  </style>
</head>
<body>
  <h1>Leaderboard</h1>
  <div id="status">Connecting...</div>
  <table id="table" style="display:none">
    <thead>
      <tr>
        <th>#</th>
        <th>Team</th>
        <th>Score (MAP)</th>
        <th>Time</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
  <p id="empty">No submissions yet.</p>

  <script>
    const medals = ["🥇","🥈","🥉"];
    let prevNames = [];

    function render(scores) {
      const tbody = document.getElementById("tbody");
      const table = document.getElementById("table");
      const empty = document.getElementById("empty");

      if (scores.length === 0) {
        table.style.display = "none";
        empty.style.display = "block";
        return;
      }
      table.style.display = "table";
      empty.style.display = "none";

      const newNames = scores.map(s => s.name + s.score);
      tbody.innerHTML = scores.map((s, i) => {
        const isNew = !prevNames.includes(s.name + s.score);
        const medal = medals[i] ? `<span class="medal">${medals[i]}</span>` : "";
        return `<tr class="${isNew ? "highlight" : ""}">
          <td class="rank">${i + 1}</td>
          <td class="name">${medal}${escHtml(s.name)}</td>
          <td class="score">${s.score.toFixed(4)}</td>
          <td class="time">${escHtml(s.submitted_at)}</td>
        </tr>`;
      }).join("");
      prevNames = newNames;
    }

    function escHtml(s) {
      return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    }

    function connect() {
      const es = new EventSource("/api/scores/stream");
      const status = document.getElementById("status");

      es.onopen = () => {
        status.textContent = "Live";
        status.className = "live";
      };
      es.onmessage = e => render(JSON.parse(e.data));
      es.onerror = () => {
        status.textContent = "Reconnecting...";
        status.className = "";
        es.close();
        setTimeout(connect, 3000);
      };
    }

    connect();
  </script>
</body>
</html>'''


if __name__ == '__main__':
    print("Starting server on http://localhost:5000 ...")
    app.run(port=5000, threaded=True)
