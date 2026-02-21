# Sentidex V2 - Stock Forecasting App

Python + React app that:

1. Accepts a stock ticker.
2. Pulls the last month of price history.
3. Pulls financial news headlines.
4. Scores sentiment per outlet.
5. Generates next-week predicted prices per outlet + combined sentiment.
6. Writes 3 JSON files:
   - past month prices
   - per-outlet predictions
   - combined chart data

## Backend (FastAPI)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Optional env var:

- `NEWS_API_KEY` (if unset, sample fallback articles are used)

## Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## API response shape

`GET /api/forecast?ticker=AAPL` returns:

- file paths for generated JSON files in `backend/data/`
- full JSON payload for UI rendering
