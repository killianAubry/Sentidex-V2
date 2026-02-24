# Sentidex V2 - Minimalist Portfolio Forecast Dashboard

## What changed

- Modern minimalist dashboard redesign
- Uses **react-stockcharts** for the center charting experience
- Top lookup bar to add symbols to a locally stored portfolio
- Horizontal popular tickers row with live-ish quote cards
- Portfolio valuation chart (historical + one-week forecast) on startup
- Dedicated portfolio page with sortable forecast outcomes
- Click any portfolio stock to inspect an individual chart

## Backend features

- `/api/forecast` for single ticker enriched forecast data
- `/api/quotes/popular` for quick card quotes
- `/api/portfolio/forecast` to aggregate portfolio valuation and projected week-end value
- Multi-provider news + sentiment + transformer/lexicon options from prior iteration

## Run backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Run frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.
