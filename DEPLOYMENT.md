# Cricket Fantasy AI — Deployment Guide

## Quick Start (Local)

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# App: http://localhost:3000
```

---

## Folder Structure

```
cricket-fantasy/
├── backend/
│   ├── api/
│   │   └── main.py              ← FastAPI routes
│   ├── core/
│   │   ├── scoring_engine.py    ← Dream11 scoring (strict rules)
│   │   ├── simulation_engine.py ← Monte Carlo simulator
│   │   ├── ml_model.py          ← ML baseline predictor
│   │   └── backtest_engine.py   ← Self-learning backtester
│   ├── data/
│   │   └── historical_data.py   ← Player + venue stats DB
│   ├── models/                  ← Saved ML model files (auto-created)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx              ← Full React dashboard
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
└── DEPLOYMENT.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Run ML + Monte Carlo prediction |
| GET | `/teams` | List all available teams |
| GET | `/players/{team}` | Get players for a team |
| GET | `/venues` | List all venues |
| GET | `/scenarios` | List all simulation scenarios |
| POST | `/score/manual` | Score a manually entered performance |
| GET | `/backtest/run` | Run self-learning backtest |
| GET | `/backtest/weights` | View learned player weight corrections |
| GET | `/health` | API health check |

---

## Sample API Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "team_a": "India",
    "team_b": "Australia",
    "batting_first_team": "India",
    "venue": "Wankhede Stadium",
    "pitch_type": "batting_friendly",
    "weather": "dew",
    "n_simulations": 500,
    "players": [
      {"name": "Rohit Sharma", "team": "India", "role": "bat", "batting_order": 1},
      {"name": "Virat Kohli", "team": "India", "role": "bat", "batting_order": 3},
      {"name": "Jasprit Bumrah", "team": "India", "role": "bowl", "batting_order": 11},
      {"name": "David Warner", "team": "Australia", "role": "bat", "batting_order": 1},
      {"name": "Adam Zampa", "team": "Australia", "role": "bowl", "batting_order": 11}
    ]
  }'
```

---

## GitHub Pages Deployment (Frontend)

```bash
cd frontend
npm run build
# Upload dist/ folder to GitHub Pages
# Or use: npx gh-pages -d dist
```

Set `base: "./"` in vite.config.js (already done).

---

## Production Deployment (Backend)

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/ .
RUN pip install -r requirements.txt
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Railway / Render
- Connect GitHub repo
- Set start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- Set root directory: `backend`

---

## Extending the System

### Add real player data
Replace `data/historical_data.py` with API calls to:
- ESPNCricinfo API
- CricSheet ball-by-ball data
- Rapid API Cricket endpoints

### Add more ML features
In `core/ml_model.py`, extend `FeatureEngineer` with:
- Head-to-head stats
- Last 5 match form
- Opponent bowling strength
- Fatigue (matches in last 7 days)

### Retrain the model
```python
from core.ml_model import CricketMLModel
import pandas as pd

model = CricketMLModel()
df = pd.read_csv("your_training_data.csv")
model.train(df)
model.save()
```
