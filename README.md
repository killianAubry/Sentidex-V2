# Sentidex V2 - Advanced Stock Intelligence Dashboard

This app now includes:

- **Multi-source news ingestion** (free-tier capable API providers):
  - NewsAPI (`NEWS_API_KEY`)
  - GNews (`GNEWS_API_KEY`)
  - Finnhub (`FINNHUB_API_KEY`)
  - Alpha Vantage (`ALPHAVANTAGE_API_KEY`)
- **Sentiment engine choices**:
  - Transformer sentiment (FinBERT, falls back if unavailable)
  - Lexicon sentiment
- **Forecasting tracks**:
  - Per-outlet sentiment 1-week projection
  - Combined sentiment 1-week projection
  - Transformer time-series 1-week projection
- **Additional datasets**:
  - Earnings data (via yfinance)
  - Macro data (FRED if `FRED_API_KEY` exists, fallback snapshot otherwise)
  - Options flow summary (put/call volume + open interest)
- **Comprehensive dashboard toggles** with center chart using **react-financial-charts** (React stock chart component family).

## Backend setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## API output files

`GET /api/forecast?ticker=AAPL` writes:

1. `*_past_month_prices.json`
2. `*_per_outlet_predictions.json`
3. `*_dashboard_chart_data.json`

All under `backend/data/`.
