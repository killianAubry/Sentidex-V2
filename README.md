# Sentidex V2 - Multi-Source Portfolio Forecast Dashboard

## New focus areas

- **Polymarket signal integration** using keyword search (e.g. `oil`, `interest rates`, `AAPL`) to derive odds-weighted forecast influence.
- **CoinMarketCap signal integration** (or fallback) to incorporate crypto risk-on/risk-off pressure into valuation forecasts.
- **Source-combination overlays** on the main chart so users can estimate valuation under specific source subsets.
- **Local cache store** for external data calls to reduce API usage (`backend/data/external_cache.json`).
- **Global Market Intelligence Globe tab** with 3D-style interactive globe layers (implemented with an SVG globe in the current UI stack).

## Globe tab capabilities

- Company & market location markers with AI sentiment, forecast, confidence, and volatility.
- Shipping route overlays with disruption scoring.
- Weather risk overlays (storm/heat nodes).
- Regional macro overlays (rates, inflation, growth, unemployment).
- AI global signal panel combining Polymarket + CoinMarketCap effects.
- Layer toggles and time slider to animate day-by-day conditions.


## GMI processing pipeline (updated)

- **Three-agent orchestration per query**: `trade_route`, `country`, and `location` systems are processed independently and merged into a single response.
- **Provider fallback search stack**: Tavily (primary) → GNews (fallback) → SerpAPI (fallback) for event extraction.
- **Polymarket-weighted impact modeling**: detected event risk is calibrated with Polymarket probabilities for confidence weighting.
- **Supply-chain dependency graph output** includes tier-1/tier-2 suppliers, commodity dependencies, and production concentration heatmap points.

## API highlights

- `GET /api/forecast?ticker=AAPL&keyword=oil&selected_sources=AlphaVantage,Polymarket`
- `GET /api/quotes/popular`
- `POST /api/portfolio/forecast`
- `GET /api/global-intelligence?keyword=interest%20rates`

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


### Optional API keys for GMI

- `TAVILY_API_KEY`
- `GNEWS_API_KEY`
- `SERPAPI_API_KEY`
- `GROQ_API_KEY`
- `WEATHERAPI_KEY`
