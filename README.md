# Sentidex V2 - Multi-Source Portfolio Forecast Dashboard

## New focus areas

- **Polymarket signal integration** using keyword search (e.g. `oil`, `interest rates`, `AAPL`) to derive odds-weighted forecast influence.
- **CoinMarketCap signal integration** (or fallback) to incorporate crypto risk-on/risk-off pressure into valuation forecasts.
- **Source-combination overlays** on the main chart so users can estimate valuation under specific source subsets (e.g. AlphaVantage only, Polymarket+CMC, or all).
- **Local cache store** for external data calls to reduce API usage and avoid over-polling (`backend/data/external_cache.json`).

## Main UX

1. Top lookup bar for adding tickers and setting a keyword for Polymarket/CMC context.
2. Horizontal popular ticker cards with quote snapshots.
3. Main portfolio valuation chart with selectable source overlays via dropdown.
4. Portfolio page with sortable positions by forecast outcome and per-stock drilldown chart.

## API highlights

- `GET /api/forecast?ticker=AAPL&keyword=oil&selected_sources=AlphaVantage,Polymarket`
- `GET /api/quotes/popular`
- `POST /api/portfolio/forecast` with body:

```json
{
  "positions": [{ "ticker": "AAPL", "shares": 4 }],
  "sentiment_model": "transformer",
  "keyword": "interest rates"
}
```

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
