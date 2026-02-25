from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import httpx
import numpy as np
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
try:
    from groq import Groq
except ImportError:
    Groq = None

from app.routes.globe import router as globe_router
from app.routes.insider import router as insider_router

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = DATA_DIR / "external_cache.json"

from app.routes.globe import router as globe_router

app = FastAPI(title="Sentidex Pro Forecast API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(globe_router)
app.include_router(insider_router)

POSITIVE_WORDS = {
    "beat",
    "growth",
    "upside",
    "bullish",
    "buy",
    "surge",
    "strong",
    "record",
    "expand",
    "optimistic",
    "gain",
    "upgrade",
    "outperform",
}
NEGATIVE_WORDS = {
    "miss",
    "downside",
    "bearish",
    "sell",
    "drop",
    "weak",
    "cut",
    "downgrade",
    "risk",
    "lawsuit",
    "decline",
    "loss",
    "underperform",
}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_cache() -> dict[str, Any]:
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_cache(cache: dict[str, Any]) -> None:
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def get_cached(key: str, ttl_seconds: int) -> Any | None:
    cache = load_cache()
    entry = cache.get(key)
    if not entry:
        return None
    if time.time() - entry.get("ts", 0) > ttl_seconds:
        return None
    return entry.get("payload")


def set_cached(key: str, payload: Any) -> None:
    cache = load_cache()
    cache[key] = {"ts": time.time(), "payload": payload}
    write_cache(cache)


def fetch_latest_quote(ticker: str) -> float:
    stock = yf.Ticker(ticker)
    fast_info = getattr(stock, "fast_info", {})
    last_price = fast_info.get("lastPrice") if isinstance(fast_info, dict) else None
    if last_price is None:
        hist = stock.history(period="5d", interval="1d")
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No quote found for {ticker}")
        last_price = float(hist["Close"].dropna().iloc[-1])
    return round(float(last_price), 2)


def fetch_month_prices(ticker: str) -> list[dict[str, Any]]:
    end = datetime.now(UTC)
    start = end - timedelta(days=35)
    df = yf.download(ticker, start=start.date(), end=end.date(), progress=False, auto_adjust=True)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No price data found for {ticker}")

    closes = df["Close"]
    if getattr(closes, "ndim", 1) > 1:
        closes = closes.iloc[:, 0]

    return [{"date": ts.strftime("%Y-%m-%d"), "close": round(float(close), 2)} for ts, close in closes.dropna().items()]


def lexicon_sentiment(text: str) -> float:
    tokens = [token.strip(".,:;!?()[]{}\"'").lower() for token in text.split()]
    if not tokens:
        return 0.0
    score = 0
    for token in tokens:
        if token in POSITIVE_WORDS:
            score += 1
        elif token in NEGATIVE_WORDS:
            score -= 1
    return score / max(len(tokens), 1)


def finbert_sentiment(text: str) -> float:
    try:
        from transformers import pipeline

        classifier = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
        output = classifier(text[:512])[0]
        weights = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        return float(sum(weights.get(item["label"].lower(), 0.0) * item["score"] for item in output))
    except Exception:
        return lexicon_sentiment(text)


async def fetch_news_from_newsapi(client: httpx.AsyncClient, ticker: str) -> list[dict[str, Any]]:
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return []
    params = {"q": f"{ticker} stock OR {ticker} earnings", "language": "en", "sortBy": "publishedAt", "pageSize": 20, "apiKey": api_key}
    resp = await client.get("https://newsapi.org/v2/everything", params=params)
    resp.raise_for_status()
    return [{"provider": "NewsAPI", "outlet": a.get("source", {}).get("name") or "Unknown", "title": a.get("title") or "", "description": a.get("description") or ""} for a in resp.json().get("articles", [])]


async def fetch_news_from_gnews(client: httpx.AsyncClient, ticker: str) -> list[dict[str, Any]]:
    api_key = os.getenv("GNEWS_API_KEY")
    if not api_key:
        return []
    params = {"q": f"{ticker} stock", "lang": "en", "max": 20, "token": api_key}
    resp = await client.get("https://gnews.io/api/v4/search", params=params)
    resp.raise_for_status()
    return [{"provider": "GNews", "outlet": a.get("source", {}).get("name") or "Unknown", "title": a.get("title") or "", "description": a.get("description") or ""} for a in resp.json().get("articles", [])]


async def fetch_news_from_finnhub(client: httpx.AsyncClient, ticker: str) -> list[dict[str, Any]]:
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return []
    today = datetime.now(UTC).date()
    week_ago = today - timedelta(days=7)
    params = {"symbol": ticker, "from": str(week_ago), "to": str(today), "token": api_key}
    resp = await client.get("https://finnhub.io/api/v1/company-news", params=params)
    resp.raise_for_status()
    return [{"provider": "Finnhub", "outlet": a.get("source") or "Unknown", "title": a.get("headline") or "", "description": a.get("summary") or ""} for a in resp.json()]


async def fetch_news_from_alphavantage(client: httpx.AsyncClient, ticker: str) -> list[dict[str, Any]]:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return []
    params = {"function": "NEWS_SENTIMENT", "tickers": ticker, "apikey": api_key, "limit": 40}
    resp = await client.get("https://www.alphavantage.co/query", params=params)
    resp.raise_for_status()
    return [{"provider": "AlphaVantage", "outlet": a.get("source") or "Unknown", "title": a.get("title") or "", "description": a.get("summary") or ""} for a in resp.json().get("feed", [])]


async def fetch_news(ticker: str) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=20) as client:
        articles: list[dict[str, Any]] = []
        for fn in [fetch_news_from_newsapi, fetch_news_from_gnews, fetch_news_from_finnhub, fetch_news_from_alphavantage]:
            try:
                articles.extend(await fn(client, ticker))
            except Exception:
                continue
    if articles:
        return articles
    return [
        {"provider": "Fallback", "outlet": "Reuters", "title": f"{ticker} shows strong growth outlook", "description": "Investors optimistic on expansion."},
        {"provider": "Fallback", "outlet": "Bloomberg", "title": f"{ticker} faces short-term risk", "description": "Funds warn about downside volatility."},
    ]


async def fetch_polymarket_signal(keyword: str) -> dict[str, Any]:
    cache_key = f"polymarket::{keyword.lower()}"
    cached = get_cached(cache_key, ttl_seconds=60 * 30)
    if cached is not None:
        return cached

    payload: dict[str, Any]
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get("https://gamma-api.polymarket.com/events", params={"query": keyword, "limit": 25})
            resp.raise_for_status()
            events = resp.json()
        probs = []
        labels = []
        for event in events[:20]:
            title = event.get("title") or ""
            labels.append(title)
            for market in event.get("markets", []):
                p = market.get("outcomePrices")
                if isinstance(p, list) and p:
                    try:
                        probs.append(float(p[0]))
                    except Exception:
                        pass
        avg_prob = round(sum(probs) / len(probs), 4) if probs else 0.5
        payload = {"source": "Polymarket", "keyword": keyword, "avgProbability": avg_prob, "sampleEvents": labels[:8]}
    except Exception:
        payload = {
            "source": "Polymarket",
            "keyword": keyword,
            "avgProbability": 0.53,
            "sampleEvents": [f"Will {keyword} rise this quarter?", f"Will macro conditions favor {keyword}?"],
        }

    set_cached(cache_key, payload)
    return payload


async def fetch_coinmarketcap_signal(keyword: str) -> dict[str, Any]:
    cache_key = f"cmc::{keyword.lower()}"
    cached = get_cached(cache_key, ttl_seconds=60 * 30)
    if cached is not None:
        return cached

    api_key = os.getenv("COINMARKETCAP_API_KEY")
    payload: dict[str, Any]
    try:
        if not api_key:
            raise ValueError("CMC key missing")
        async with httpx.AsyncClient(timeout=20) as client:
            headers = {"X-CMC_PRO_API_KEY": api_key}
            # Proxy macro sentiment through BTC, ETH, total market proxy assets
            resp = await client.get(
                "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest",
                params={"symbol": "BTC,ETH,SOL"},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
        changes = []
        for sym in ["BTC", "ETH", "SOL"]:
            q = data.get(sym, {}).get("quote", {}).get("USD", {})
            if q.get("percent_change_24h") is not None:
                changes.append(float(q["percent_change_24h"]))
        avg_change = round(sum(changes) / len(changes), 4) if changes else 0.0
        normalized = max(min(avg_change / 10, 1), -1)
        payload = {"source": "CoinMarketCap", "keyword": keyword, "avg24hChangePct": avg_change, "normalizedSignal": normalized}
    except Exception:
        payload = {"source": "CoinMarketCap", "keyword": keyword, "avg24hChangePct": 0.8, "normalizedSignal": 0.08}

    set_cached(cache_key, payload)
    return payload


def sentiment_by_outlet(articles: list[dict[str, Any]], mode: str = "transformer") -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"sentimentScores": [], "headlines": [], "providers": set()})
    for article in articles:
        outlet = article.get("outlet") or "Unknown"
        provider = article.get("provider") or "Unknown"
        text = f"{article.get('title', '')} {article.get('description', '')}".strip()
        score = finbert_sentiment(text) if mode == "transformer" else lexicon_sentiment(text)
        grouped[outlet]["sentimentScores"].append(score)
        grouped[outlet]["headlines"].append(article.get("title", ""))
        grouped[outlet]["providers"].add(provider)

    payload: dict[str, dict[str, Any]] = {}
    for outlet, values in grouped.items():
        scores = values["sentimentScores"]
        payload[outlet] = {
            "avgSentiment": round(sum(scores) / max(len(scores), 1), 4),
            "articleCount": len(scores),
            "providers": sorted(values["providers"]),
            "sampleHeadlines": values["headlines"][:5],
        }
    return payload


def project_week(history: list[dict[str, Any]], score: float) -> list[dict[str, Any]]:
    closes = [x["close"] for x in history[-10:]]
    curr = closes[-1]
    volatility = (max(closes) - min(closes)) / max(curr, 1)
    drift = score * 0.35
    start = datetime.strptime(history[-1]["date"], "%Y-%m-%d")
    out = []
    for i in range(1, 8):
        cyclical = math.sin(i / 2.1) * volatility * 0.2
        curr = curr * (1 + drift / 7 + cyclical)
        out.append({"date": (start + timedelta(days=i)).strftime("%Y-%m-%d"), "predictedClose": round(curr, 2)})
    return out


def transformer_week_forecast(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    closes = np.array([x["close"] for x in history], dtype=np.float32)
    try:
        import torch
        import torch.nn as nn

        window = 14
        if len(closes) <= window:
            raise ValueError("insufficient history")
        mean, std = closes.mean(), closes.std() + 1e-6
        norm = (closes - mean) / std
        xs, ys = [], []
        for i in range(len(norm) - window):
            xs.append(norm[i : i + window])
            ys.append(norm[i + window])
        x = torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(-1)

        class TinyTransformer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_proj = nn.Linear(1, 16)
                enc_layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=32, dropout=0.1)
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
                self.out = nn.Linear(16, 1)

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                return self.out(self.encoder(self.in_proj(z))[:, -1, :])

        model = TinyTransformer()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        for _ in range(30):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

        seq = norm[-window:].tolist()
        start = datetime.strptime(history[-1]["date"], "%Y-%m-%d")
        result = []
        for i in range(1, 8):
            inp = torch.tensor(seq[-window:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            nxt = float(model(inp).detach().numpy().squeeze())
            seq.append(nxt)
            result.append({"date": (start + timedelta(days=i)).strftime("%Y-%m-%d"), "predictedClose": round(float(nxt * std + mean), 2)})
        return result
    except Exception:
        slope = (closes[-1] - closes[0]) / max(len(closes) - 1, 1)
        start = datetime.strptime(history[-1]["date"], "%Y-%m-%d")
        return [{"date": (start + timedelta(days=i)).strftime("%Y-%m-%d"), "predictedClose": round(float(closes[-1] + slope * i), 2)} for i in range(1, 8)]


def fetch_earnings_data(ticker: str) -> dict[str, Any]:
    stock = yf.Ticker(ticker)
    earnings_dates = stock.get_earnings_dates(limit=4)
    rows = []
    if earnings_dates is not None and not earnings_dates.empty:
        for idx, row in earnings_dates.iterrows():
            rows.append({
                "date": idx.strftime("%Y-%m-%d"),
                "epsEstimate": float(row.get("EPS Estimate", np.nan)) if not np.isnan(row.get("EPS Estimate", np.nan)) else None,
                "reportedEPS": float(row.get("Reported EPS", np.nan)) if not np.isnan(row.get("Reported EPS", np.nan)) else None,
                "surprisePct": float(row.get("Surprise(%)", np.nan)) if not np.isnan(row.get("Surprise(%)", np.nan)) else None,
            })
    return {"recentEarnings": rows}


async def fetch_macro_data() -> dict[str, Any]:
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        return {"source": "fallback", "latest": {"cpiYoY": 3.1, "fedFunds": 5.25, "unemployment": 3.9, "tenYearYield": 4.15}}
    series = {"cpiYoY": "CPIAUCSL", "fedFunds": "FEDFUNDS", "unemployment": "UNRATE", "tenYearYield": "DGS10"}
    async with httpx.AsyncClient(timeout=20) as client:
        latest: dict[str, float | None] = {}
        for label, sid in series.items():
            params = {"series_id": sid, "api_key": fred_key, "file_type": "json", "sort_order": "desc", "limit": 1}
            try:
                resp = await client.get("https://api.stlouisfed.org/fred/series/observations", params=params)
                resp.raise_for_status()
                latest[label] = float(resp.json().get("observations", [{}])[0].get("value"))
            except Exception:
                latest[label] = None
    return {"source": "FRED", "latest": latest}


def fetch_options_flow(ticker: str) -> dict[str, Any]:
    stock = yf.Ticker(ticker)
    expirations = stock.options[:3]
    out = []
    for exp in expirations:
        chain = stock.option_chain(exp)
        call_volume = float(chain.calls["volume"].fillna(0).sum())
        put_volume = float(chain.puts["volume"].fillna(0).sum())
        out.append({
            "expiration": exp,
            "callVolume": call_volume,
            "putVolume": put_volume,
            "putCallVolumeRatio": round(put_volume / call_volume, 4) if call_volume else None,
            "callOpenInterest": float(chain.calls["openInterest"].fillna(0).sum()),
            "putOpenInterest": float(chain.puts["openInterest"].fillna(0).sum()),
        })
    return {"expirations": out}


class PortfolioPosition(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)
    shares: float = Field(gt=0)


class PortfolioRequest(BaseModel):
    positions: list[PortfolioPosition]
    sentiment_model: str = "transformer"
    keyword: str | None = None


def source_weighted_curves(
    history: list[dict[str, Any]],
    provider_sentiment: dict[str, float],
    polymarket_prob: float,
    cmc_signal: float,
) -> dict[str, list[dict[str, Any]]]:
    curves: dict[str, list[dict[str, Any]]] = {}
    for provider, sentiment in provider_sentiment.items():
        curves[provider] = project_week(history, sentiment)

    # External sources as synthetic score curves
    pm_score = (polymarket_prob - 0.5) * 2
    curves["Polymarket"] = project_week(history, pm_score)
    curves["CoinMarketCap"] = project_week(history, cmc_signal)
    return curves


def combine_selected_sources(curves: dict[str, list[dict[str, Any]]], selected_sources: list[str] | None = None) -> list[dict[str, Any]]:
    keys = selected_sources if selected_sources else list(curves.keys())
    valid = [k for k in keys if k in curves]
    if not valid:
        return []

    by_date: dict[str, list[float]] = defaultdict(list)
    for key in valid:
        for row in curves[key]:
            by_date[row["date"]].append(row["predictedClose"])

    return [{"date": d, "predictedClose": round(sum(vals) / len(vals), 2)} for d, vals in sorted(by_date.items()) if vals]


async def build_forecast_payload(ticker: str, sentiment_model: str, history: list[dict[str, Any]], articles: list[dict[str, Any]], keyword: str | None = None) -> dict[str, Any]:
    sentiments = sentiment_by_outlet(articles, mode=sentiment_model)
    if not sentiments:
        raise HTTPException(status_code=404, detail="No outlet sentiment data generated")

    per_outlet: dict[str, Any] = {}
    provider_scores: dict[str, list[float]] = defaultdict(list)
    for outlet, details in sentiments.items():
        curve = project_week(history, details["avgSentiment"])
        per_outlet[outlet] = {**details, "nextWeekForecast": curve, "predictedWeekEndPrice": curve[-1]["predictedClose"]}
        for provider in details.get("providers", []):
            provider_scores[provider].append(details["avgSentiment"])

    provider_sentiment = {k: sum(v) / len(v) for k, v in provider_scores.items() if v}
    combined_score = sum(p["avgSentiment"] for p in per_outlet.values()) / len(per_outlet)

    q = keyword or ticker
    polymarket = await fetch_polymarket_signal(q)
    cmc = await fetch_coinmarketcap_signal(q)

    source_curves = source_weighted_curves(history, provider_sentiment, polymarket.get("avgProbability", 0.5), cmc.get("normalizedSignal", 0.0))
    source_curves["CombinedSentiment"] = project_week(history, combined_score)
    source_curves["TransformerTS"] = transformer_week_forecast(history)

    combined_forecast = combine_selected_sources(source_curves)

    return {
        "ticker": ticker,
        "generatedAt": datetime.now(UTC).isoformat(),
        "historicalPrices": history,
        "perOutletForecast": per_outlet,
        "combinedSentimentScore": round(combined_score, 4),
        "combinedForecast": combined_forecast,
        "sourceForecasts": source_curves,
        "sourceScores": {
            **{k: round(v, 4) for k, v in provider_sentiment.items()},
            "Polymarket": round((polymarket.get("avgProbability", 0.5) - 0.5) * 2, 4),
            "CoinMarketCap": round(cmc.get("normalizedSignal", 0.0), 4),
        },
        "externalSignals": {"polymarket": polymarket, "coinmarketcap": cmc},
        "earningsData": fetch_earnings_data(ticker),
        "optionsFlow": fetch_options_flow(ticker),
        "newsProvidersUsed": sorted(list({a.get("provider", "Unknown") for a in articles})),
    }


@app.get("/api/forecast")
async def forecast(
    ticker: str = Query(..., min_length=1, max_length=10),
    sentiment_model: str = Query("transformer"),
    keyword: str | None = Query(None),
    selected_sources: str | None = Query(None),
) -> dict[str, Any]:
    ticker = ticker.upper().strip()
    history = fetch_month_prices(ticker)
    articles = await fetch_news(ticker)
    payload = await build_forecast_payload(ticker, sentiment_model, history, articles, keyword=keyword)
    payload["macroData"] = await fetch_macro_data()

    if selected_sources:
        keys = [x.strip() for x in selected_sources.split(",") if x.strip()]
        custom = combine_selected_sources(payload["sourceForecasts"], keys)
        if custom:
            payload["combinedForecast"] = custom
            payload["selectedSources"] = keys

    now = datetime.now(UTC)
    base = f"{ticker}_{now.strftime('%Y%m%d_%H%M%S')}"
    prices_path = DATA_DIR / f"{base}_past_month_prices.json"
    per_outlet_path = DATA_DIR / f"{base}_per_outlet_predictions.json"
    dashboard_path = DATA_DIR / f"{base}_dashboard_chart_data.json"
    save_json(prices_path, {"ticker": ticker, "window": "1m", "prices": history})
    save_json(per_outlet_path, {"ticker": ticker, "generatedAt": now.isoformat(), "perOutletForecast": payload["perOutletForecast"]})
    save_json(dashboard_path, payload)

    return {
        "ticker": ticker,
        "files": {
            "pastMonthPrices": str(prices_path.relative_to(BASE_DIR)),
            "perOutletPredictions": str(per_outlet_path.relative_to(BASE_DIR)),
            "dashboardChartData": str(dashboard_path.relative_to(BASE_DIR)),
            "cacheStore": str(CACHE_FILE.relative_to(BASE_DIR)),
        },
        "data": payload,
    }


@app.get("/api/quotes/popular")
def popular_quotes() -> dict[str, Any]:
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM"]
    quotes = []
    for ticker in tickers:
        try:
            quotes.append({"ticker": ticker, "price": fetch_latest_quote(ticker)})
        except Exception:
            continue
    return {"quotes": quotes}


@app.post("/api/portfolio/forecast")
async def portfolio_forecast(req: PortfolioRequest) -> dict[str, Any]:
    if not req.positions:
        raise HTTPException(status_code=400, detail="Portfolio is empty")

    positions = []
    for pos in req.positions:
        ticker = pos.ticker.upper().strip()
        history = fetch_month_prices(ticker)
        articles = await fetch_news(ticker)
        payload = await build_forecast_payload(ticker, req.sentiment_model, history, articles, keyword=req.keyword or ticker)
        latest = history[-1]["close"]
        next_week = payload["combinedForecast"][-1]["predictedClose"]
        positions.append(
            {
                "ticker": ticker,
                "shares": pos.shares,
                "latestPrice": latest,
                "forecastWeekEndPrice": next_week,
                "forecastDeltaPct": round((next_week - latest) / latest * 100, 3) if latest else 0,
                "historicalPrices": history,
                "combinedForecast": payload["combinedForecast"],
                "sourceForecasts": payload["sourceForecasts"],
            }
        )

    hist_dates = {row["date"] for p in positions for row in p["historicalPrices"]}
    forecast_dates = {row["date"] for p in positions for row in p["combinedForecast"]}
    hist_agg = {d: 0.0 for d in hist_dates}
    forecast_agg = {d: 0.0 for d in forecast_dates}

    for p in positions:
        hist_map = {r["date"]: r["close"] for r in p["historicalPrices"]}
        fore_map = {r["date"]: r["predictedClose"] for r in p["combinedForecast"]}
        for d in hist_agg:
            hist_agg[d] += hist_map.get(d, 0.0) * p["shares"]
        for d in forecast_agg:
            forecast_agg[d] += fore_map.get(d, 0.0) * p["shares"]

    history = [{"date": d, "value": round(v, 2)} for d, v in sorted(hist_agg.items())]
    forecast = [{"date": d, "value": round(v, 2)} for d, v in sorted(forecast_agg.items())]

    return {
        "positions": positions,
        "portfolioValuation": {
            "history": history,
            "forecast": forecast,
            "currentValue": history[-1]["value"] if history else 0,
            "forecastWeekEndValue": forecast[-1]["value"] if forecast else 0,
        },
    }


def project_coord(lat: float, lon: float) -> dict[str, float]:
    # Simple equirectangular projection for frontend SVG globe fallback.
    x = (lon + 180) / 360
    y = (90 - lat) / 180
    return {"x": round(x, 6), "y": round(y, 6)}


async def brainstorm_supply_chain(keyword: str) -> list[dict[str, Any]]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or Groq is None:
        # Detailed simulation if no API key
        kw = keyword.lower()
        if "apple" in kw or "aapl" in kw:
            return [
                {"name": "Apple Park (HQ)", "lat": 37.3349, "lon": -122.0090, "type": "HQ", "riskScore": 0.12},
                {"name": "Foxconn Zhengzhou", "lat": 34.7466, "lon": 113.6253, "type": "Factory", "riskScore": 0.58},
                {"name": "Port of Long Beach", "lat": 33.7701, "lon": -118.1937, "type": "Logistics", "riskScore": 0.34},
                {"name": "Cork Operations", "lat": 51.8985, "lon": -8.4756, "type": "Regional HQ", "riskScore": 0.15},
            ]
        return [
            {"name": f"{keyword} Global HQ", "lat": 40.7128, "lon": -74.0060, "type": "HQ", "riskScore": 0.1},
            {"name": f"{keyword} Manufacturing Unit", "lat": 22.3193, "lon": 114.1694, "type": "Factory", "riskScore": 0.45},
            {"name": f"{keyword} Key Distribution Point", "lat": 52.3676, "lon": 4.9041, "type": "Logistics", "riskScore": 0.25}
        ]

    try:
        client = Groq(api_key=api_key)
        prompt = (
            f"Identify 5 key supply chain nodes for '{keyword}'. "
            "Return ONLY a JSON object with a 'nodes' key containing a list of objects. "
            "Each object must have 'name', 'lat', 'lon', 'type', and 'riskScore' (0.0 to 1.0)."
        )
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            response_format={"type": "json_object"}
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("nodes", [])
    except Exception:
        return [{"name": f"{keyword} Ops", "lat": 0, "lon": 0, "type": "Unknown", "riskScore": 0.5}]


async def build_global_intelligence(keyword: str = "global markets") -> dict[str, Any]:
    cache_key = f"global-intel::{keyword.lower()}"
    cached = get_cached(cache_key, ttl_seconds=60 * 20)
    if cached is not None:
        return cached

    kw = keyword.lower()

    # Query AI for supply chain nodes
    ai_nodes = await brainstorm_supply_chain(keyword)

    companies = [
        {"ticker": "AAPL", "name": "Apple", "lat": 37.3349, "lon": -122.0090, "region": "North America"},
        {"ticker": "MSFT", "name": "Microsoft", "lat": 47.6426, "lon": -122.1396, "region": "North America"},
        {"ticker": "NVDA", "name": "NVIDIA", "lat": 37.3875, "lon": -121.9631, "region": "North America"},
        {"ticker": "TSM", "name": "TSMC", "lat": 24.8138, "lon": 120.9675, "region": "Asia"},
        {"ticker": "SHEL", "name": "Shell", "lat": 51.5072, "lon": -0.1276, "region": "Europe"},
        {"ticker": "BABA", "name": "Alibaba", "lat": 30.2741, "lon": 120.1551, "region": "Asia"},
        {"ticker": "XOM", "name": "ExxonMobil", "lat": 32.8925, "lon": -96.9419, "region": "North America"},
        {"ticker": "TTE", "name": "TotalEnergies", "lat": 48.8566, "lon": 2.3522, "region": "Europe"},
    ]

    if "oil" in kw or "energy" in kw or "gas" in kw:
        companies = [c for c in companies if c["ticker"] in ["SHEL", "XOM", "TTE"]]
    elif "tech" in kw or "chip" in kw:
        companies = [c for c in companies if c["ticker"] in ["AAPL", "MSFT", "NVDA", "TSM"]]

    polymarket = await fetch_polymarket_signal(keyword)
    cmc = await fetch_coinmarketcap_signal(keyword)

    company_markers: list[dict[str, Any]] = []

    # Combine static and AI nodes
    combined_nodes = companies + ai_nodes

    for i, c in enumerate(combined_nodes):
        ticker = c.get("ticker")
        try:
            if not ticker:
                raise ValueError("No ticker")
            history = fetch_month_prices(ticker)
            articles = await fetch_news(ticker)
            forecast_payload = await build_forecast_payload(ticker, "transformer", history, articles, keyword)
            pred = forecast_payload["combinedForecast"][-1]["predictedClose"]
            last = history[-1]["close"]
            direction = "up" if pred >= last else "down"
            confidence = min(0.95, max(0.35, abs(pred - last) / max(last, 1)))
            sentiment = forecast_payload["combinedSentimentScore"]
            options_flow = forecast_payload["optionsFlow"]
        except Exception:
            last = fetch_latest_quote(ticker) if ticker else 150.0
            pred = round(last * 1.01, 2)
            direction = "up"
            confidence = 0.5
            sentiment = 0.08
            options_flow = {"expirations": []}

        company_markers.append(
            {
                **c,
                **project_coord(c["lat"], c["lon"]),
                "dayOffset": i % 8,
                "priceNow": last,
                "forecastPrice": pred,
                "direction": direction,
                "confidence": round(confidence, 4),
                "volatility": round(abs(pred - last) / max(last, 1), 4),
                "sentiment": sentiment,
                "optionsFlow": options_flow,
                "riskScore": c.get("riskScore", 0.2)
            }
        )

    if "oil" in kw or "energy" in kw or "gas" in kw:
        shipping_routes = [
            {
                "name": "Middle East -> Europe",
                "from": {"lat": 25.2048, "lon": 55.2708, **project_coord(25.2048, 55.2708)},
                "to": {"lat": 51.5072, "lon": -0.1276, **project_coord(51.5072, -0.1276)},
                "type": "oil-tanker",
                "disruptionScore": 0.55,
                "dayOffset": 1,
            },
            {
                "name": "US Gulf -> Europe",
                "from": {"lat": 29.7604, "lon": -95.3698, **project_coord(29.7604, -95.3698)},
                "to": {"lat": 51.9225, "lon": 4.4791, **project_coord(51.9225, 4.4791)},
                "type": "oil-tanker",
                "disruptionScore": 0.25,
                "dayOffset": 3,
            }
        ]
    elif "tech" in kw or "chip" in kw:
        shipping_routes = [
            {
                "name": "Taiwan -> US West Coast",
                "from": {"lat": 24.8138, "lon": 120.9675, **project_coord(24.8138, 120.9675)},
                "to": {"lat": 34.0522, "lon": -118.2437, **project_coord(34.0522, -118.2437)},
                "type": "container",
                "disruptionScore": 0.34,
                "dayOffset": 2,
            },
            {
                "name": "Shanghai -> Silicon Valley",
                "from": {"lat": 31.2304, "lon": 121.4737, **project_coord(31.2304, 121.4737)},
                "to": {"lat": 37.3382, "lon": -121.8863, **project_coord(37.3382, -121.8863)},
                "type": "container",
                "disruptionScore": 0.45,
                "dayOffset": 4,
            }
        ]
    else:
        shipping_routes = [
            {
                "name": "Asia -> US West Coast",
                "from": {"lat": 31.2304, "lon": 121.4737, **project_coord(31.2304, 121.4737)},
                "to": {"lat": 34.0522, "lon": -118.2437, **project_coord(34.0522, -118.2437)},
                "type": "container",
                "disruptionScore": 0.34,
                "dayOffset": 2,
            },
            {
                "name": "Middle East -> Europe",
                "from": {"lat": 25.2048, "lon": 55.2708, **project_coord(25.2048, 55.2708)},
                "to": {"lat": 51.5072, "lon": -0.1276, **project_coord(51.5072, -0.1276)},
                "type": "oil-tanker",
                "disruptionScore": 0.49,
                "dayOffset": 5,
            },
        ]

    # Add transit routes between AI nodes
    for i in range(len(ai_nodes) - 1):
        n1 = ai_nodes[i]
        n2 = ai_nodes[i+1]
        shipping_routes.append({
            "name": f"{n1['name']} -> {n2['name']}",
            "from": {"lat": n1["lat"], "lon": n1["lon"], **project_coord(n1["lat"], n1["lon"])},
            "to": {"lat": n2["lat"], "lon": n2["lon"], **project_coord(n2["lat"], n2["lon"])},
            "type": "transit",
            "disruptionScore": (n1.get("riskScore", 0) + n2.get("riskScore", 0)) / 2,
            "dayOffset": (i + 1) * 2
        })

    weather_nodes = [
        {"name": "North Atlantic Storm", "lat": 46.0, "lon": -35.0, **project_coord(46.0, -35.0), "severity": 0.78, "precipitationMm": 34, "dayOffset": 3},
        {"name": "Pacific Typhoon Band", "lat": 20.0, "lon": 140.0, **project_coord(20.0, 140.0), "severity": 0.69, "precipitationMm": 52, "dayOffset": 6},
        {"name": "Gulf Heat Stress", "lat": 27.0, "lon": 48.0, **project_coord(27.0, 48.0), "severity": 0.56, "tempC": 39, "dayOffset": 7},
    ]

    macro = {
        "North America": {"interestRate": 5.25, "inflation": 3.1, "gdpGrowth": 2.2, "unemployment": 3.9, "lat": 39.0, "lon": -98.0, **project_coord(39.0, -98.0), "dayOffset": 0},
        "Europe": {"interestRate": 4.0, "inflation": 2.8, "gdpGrowth": 1.1, "unemployment": 6.1, "lat": 50.0, "lon": 10.0, **project_coord(50.0, 10.0), "dayOffset": 0},
        "Asia": {"interestRate": 3.2, "inflation": 2.4, "gdpGrowth": 4.3, "unemployment": 4.6, "lat": 23.0, "lon": 105.0, **project_coord(23.0, 105.0), "dayOffset": 0},
    }

    ai_signal = {
        "keyword": keyword,
        "polymarketProbability": polymarket.get("avgProbability", 0.5),
        "coinMarketRegime": cmc.get("normalizedSignal", 0.0),
        "globalDirection": round(((polymarket.get("avgProbability", 0.5) - 0.5) * 2 + cmc.get("normalizedSignal", 0.0)) / 2, 4),
    }

    payload = {
        "generatedAt": datetime.now(UTC).isoformat(),
        "keyword": keyword,
        "layers": {
            "companies": company_markers,
            "shippingRoutes": shipping_routes,
            "weather": weather_nodes,
            "macro": macro,
            "aiSignal": ai_signal,
        },
        "timeWindowDays": 14,
    }
    set_cached(cache_key, payload)
    return payload


@app.get("/api/global-intelligence")
async def global_intelligence(keyword: str = Query("global markets")) -> dict[str, Any]:
    payload = await build_global_intelligence(keyword)
    return {"data": payload, "cache": str(CACHE_FILE.relative_to(BASE_DIR))}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
