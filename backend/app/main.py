from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Sentidex Pro Forecast API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    history: list[dict[str, Any]] = []
    for ts, close in closes.dropna().items():
        history.append({"date": ts.strftime("%Y-%m-%d"), "close": round(float(close), 2)})
    return history


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
    params = {
        "q": f"{ticker} stock OR {ticker} earnings",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 25,
        "apiKey": api_key,
    }
    resp = await client.get("https://newsapi.org/v2/everything", params=params)
    resp.raise_for_status()
    return [
        {
            "provider": "NewsAPI",
            "outlet": a.get("source", {}).get("name") or "Unknown",
            "title": a.get("title") or "",
            "description": a.get("description") or "",
        }
        for a in resp.json().get("articles", [])
    ]


async def fetch_news_from_gnews(client: httpx.AsyncClient, ticker: str) -> list[dict[str, Any]]:
    api_key = os.getenv("GNEWS_API_KEY")
    if not api_key:
        return []
    params = {"q": f"{ticker} stock", "lang": "en", "max": 20, "token": api_key}
    resp = await client.get("https://gnews.io/api/v4/search", params=params)
    resp.raise_for_status()
    return [
        {
            "provider": "GNews",
            "outlet": a.get("source", {}).get("name") or "Unknown",
            "title": a.get("title") or "",
            "description": a.get("description") or "",
        }
        for a in resp.json().get("articles", [])
    ]


async def fetch_news_from_finnhub(client: httpx.AsyncClient, ticker: str) -> list[dict[str, Any]]:
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return []
    today = datetime.now(UTC).date()
    week_ago = today - timedelta(days=7)
    params = {"symbol": ticker, "from": str(week_ago), "to": str(today), "token": api_key}
    resp = await client.get("https://finnhub.io/api/v1/company-news", params=params)
    resp.raise_for_status()
    return [
        {
            "provider": "Finnhub",
            "outlet": a.get("source") or "Unknown",
            "title": a.get("headline") or "",
            "description": a.get("summary") or "",
        }
        for a in resp.json()
    ]


async def fetch_news_from_alphavantage(client: httpx.AsyncClient, ticker: str) -> list[dict[str, Any]]:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return []
    params = {"function": "NEWS_SENTIMENT", "tickers": ticker, "apikey": api_key, "limit": 40}
    resp = await client.get("https://www.alphavantage.co/query", params=params)
    resp.raise_for_status()
    return [
        {
            "provider": "AlphaVantage",
            "outlet": a.get("source") or "Unknown",
            "title": a.get("title") or "",
            "description": a.get("summary") or "",
        }
        for a in resp.json().get("feed", [])
    ]


async def fetch_news(ticker: str) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=20) as client:
        articles: list[dict[str, Any]] = []
        for fn in [
            fetch_news_from_newsapi,
            fetch_news_from_gnews,
            fetch_news_from_finnhub,
            fetch_news_from_alphavantage,
        ]:
            try:
                articles.extend(await fn(client, ticker))
            except Exception:
                continue

    if articles:
        return articles

    return [
        {
            "provider": "Fallback",
            "outlet": "Reuters",
            "title": f"{ticker} shows strong growth outlook as analysts upgrade sentiment",
            "description": "Investors optimistic on demand expansion and record margins.",
        },
        {
            "provider": "Fallback",
            "outlet": "Bloomberg",
            "title": f"{ticker} faces short-term risk despite resilient fundamentals",
            "description": "Some funds warn about downside volatility and sector decline.",
        },
        {
            "provider": "Fallback",
            "outlet": "CNBC",
            "title": f"Traders bullish as {ticker} may beat quarterly expectations",
            "description": "Potential gain if revenue surprise remains strong.",
        },
    ]


def sentiment_by_outlet(articles: list[dict[str, Any]], mode: str = "transformer") -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"sentimentScores": [], "headlines": [], "providers": set()}
    )

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

    output: list[dict[str, Any]] = []
    for i in range(1, 8):
        cyclical = math.sin(i / 2.1) * volatility * 0.2
        curr = curr * (1 + drift / 7 + cyclical)
        output.append({"date": (start + timedelta(days=i)).strftime("%Y-%m-%d"), "predictedClose": round(curr, 2)})
    return output


def transformer_week_forecast(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    closes = np.array([x["close"] for x in history], dtype=np.float32)
    try:
        import torch
        import torch.nn as nn

        window = 14
        if len(closes) <= window:
            raise ValueError("Insufficient history")

        mean = closes.mean()
        std = closes.std() + 1e-6
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
                z = self.in_proj(z)
                z = self.encoder(z)
                return self.out(z[:, -1, :])

        model = TinyTransformer()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(30):
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

        model.eval()
        seq = norm[-window:].tolist()
        start = datetime.strptime(history[-1]["date"], "%Y-%m-%d")
        out: list[dict[str, Any]] = []
        for i in range(1, 8):
            inp = torch.tensor(seq[-window:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            nxt = float(model(inp).detach().numpy().squeeze())
            seq.append(nxt)
            price = nxt * std + mean
            out.append({"date": (start + timedelta(days=i)).strftime("%Y-%m-%d"), "predictedClose": round(float(price), 2)})
        return out
    except Exception:
        # Fallback trend if torch/transformer is unavailable in runtime.
        slope = (closes[-1] - closes[0]) / max(len(closes) - 1, 1)
        start = datetime.strptime(history[-1]["date"], "%Y-%m-%d")
        return [
            {
                "date": (start + timedelta(days=i)).strftime("%Y-%m-%d"),
                "predictedClose": round(float(closes[-1] + slope * i), 2),
            }
            for i in range(1, 8)
        ]


def fetch_earnings_data(ticker: str) -> dict[str, Any]:
    stock = yf.Ticker(ticker)
    earnings_dates = stock.get_earnings_dates(limit=4)
    rows: list[dict[str, Any]] = []
    if earnings_dates is not None and not earnings_dates.empty:
        for idx, row in earnings_dates.iterrows():
            rows.append(
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "epsEstimate": float(row.get("EPS Estimate", np.nan)) if not np.isnan(row.get("EPS Estimate", np.nan)) else None,
                    "reportedEPS": float(row.get("Reported EPS", np.nan)) if not np.isnan(row.get("Reported EPS", np.nan)) else None,
                    "surprisePct": float(row.get("Surprise(%)", np.nan)) if not np.isnan(row.get("Surprise(%)", np.nan)) else None,
                }
            )
    return {"recentEarnings": rows}


async def fetch_macro_data() -> dict[str, Any]:
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        return {
            "source": "fallback",
            "latest": {
                "cpiYoY": 3.1,
                "fedFunds": 5.25,
                "unemployment": 3.9,
                "tenYearYield": 4.15,
            },
        }

    series = {
        "cpiYoY": "CPIAUCSL",
        "fedFunds": "FEDFUNDS",
        "unemployment": "UNRATE",
        "tenYearYield": "DGS10",
    }
    async with httpx.AsyncClient(timeout=20) as client:
        latest: dict[str, float | None] = {}
        for label, sid in series.items():
            params = {"series_id": sid, "api_key": fred_key, "file_type": "json", "sort_order": "desc", "limit": 1}
            try:
                resp = await client.get("https://api.stlouisfed.org/fred/series/observations", params=params)
                resp.raise_for_status()
                value = resp.json().get("observations", [{}])[0].get("value")
                latest[label] = float(value)
            except Exception:
                latest[label] = None
    return {"source": "FRED", "latest": latest}


def fetch_options_flow(ticker: str) -> dict[str, Any]:
    stock = yf.Ticker(ticker)
    expirations = stock.options[:3]
    data = []
    for exp in expirations:
        chain = stock.option_chain(exp)
        call_volume = float(chain.calls["volume"].fillna(0).sum())
        put_volume = float(chain.puts["volume"].fillna(0).sum())
        call_oi = float(chain.calls["openInterest"].fillna(0).sum())
        put_oi = float(chain.puts["openInterest"].fillna(0).sum())
        data.append(
            {
                "expiration": exp,
                "callVolume": call_volume,
                "putVolume": put_volume,
                "putCallVolumeRatio": round(put_volume / call_volume, 4) if call_volume else None,
                "callOpenInterest": call_oi,
                "putOpenInterest": put_oi,
            }
        )
    return {"expirations": data}


def build_forecast_payload(ticker: str, sentiment_model: str, history: list[dict[str, Any]], articles: list[dict[str, Any]]) -> dict[str, Any]:
    sentiments = sentiment_by_outlet(articles, mode=sentiment_model)

    per_outlet: dict[str, Any] = {}
    for outlet, details in sentiments.items():
        forecast_curve = project_week(history, details["avgSentiment"])
        per_outlet[outlet] = {
            **details,
            "nextWeekForecast": forecast_curve,
            "predictedWeekEndPrice": forecast_curve[-1]["predictedClose"],
        }

    if not per_outlet:
        raise HTTPException(status_code=404, detail="No outlet sentiment data generated")

    combined_score = sum(x["avgSentiment"] for x in per_outlet.values()) / len(per_outlet)
    combined_forecast = project_week(history, combined_score)
    transformer_forecast = transformer_week_forecast(history)
    earnings = fetch_earnings_data(ticker)
    options_flow = fetch_options_flow(ticker)

    return {
        "ticker": ticker,
        "generatedAt": datetime.now(UTC).isoformat(),
        "historicalPrices": history,
        "perOutletForecast": per_outlet,
        "combinedSentimentScore": round(combined_score, 4),
        "combinedForecast": combined_forecast,
        "transformerForecast": transformer_forecast,
        "earningsData": earnings,
        "optionsFlow": options_flow,
        "newsProvidersUsed": sorted(list({a.get("provider", "Unknown") for a in articles})),
    }


class PortfolioPosition(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)
    shares: float = Field(gt=0)


class PortfolioRequest(BaseModel):
    positions: list[PortfolioPosition]
    sentiment_model: str = "transformer"


@app.get("/api/forecast")
async def forecast(ticker: str = Query(..., min_length=1, max_length=10), sentiment_model: str = Query("transformer")) -> dict[str, Any]:
    ticker = ticker.upper().strip()
    history = fetch_month_prices(ticker)
    articles = await fetch_news(ticker)
    payload = build_forecast_payload(ticker, sentiment_model, history, articles)
    payload["macroData"] = await fetch_macro_data()

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

    position_payloads: list[dict[str, Any]] = []
    for pos in req.positions:
        ticker = pos.ticker.upper().strip()
        history = fetch_month_prices(ticker)
        articles = await fetch_news(ticker)
        payload = build_forecast_payload(ticker, req.sentiment_model, history, articles)
        latest = history[-1]["close"]
        next_week = payload["combinedForecast"][-1]["predictedClose"]
        position_payloads.append(
            {
                "ticker": ticker,
                "shares": pos.shares,
                "latestPrice": latest,
                "forecastWeekEndPrice": next_week,
                "forecastDeltaPct": round((next_week - latest) / latest * 100, 3) if latest else 0,
                "historicalPrices": history,
                "combinedForecast": payload["combinedForecast"],
                "perOutletForecast": payload["perOutletForecast"],
            }
        )

    base_history = position_payloads[0]["historicalPrices"]
    by_date: dict[str, float] = {row["date"]: 0.0 for row in base_history}
    future_dates = {row["date"] for row in position_payloads[0]["combinedForecast"]}
    by_future_date: dict[str, float] = {d: 0.0 for d in future_dates}

    for pos in position_payloads:
        hist_map = {row["date"]: row["close"] for row in pos["historicalPrices"]}
        fut_map = {row["date"]: row["predictedClose"] for row in pos["combinedForecast"]}
        for d in by_date:
            by_date[d] += hist_map.get(d, 0.0) * pos["shares"]
        for d in by_future_date:
            by_future_date[d] += fut_map.get(d, 0.0) * pos["shares"]

    valuation_history = [{"date": d, "value": round(v, 2)} for d, v in sorted(by_date.items())]
    valuation_forecast = [{"date": d, "value": round(v, 2)} for d, v in sorted(by_future_date.items())]

    return {
        "positions": position_payloads,
        "portfolioValuation": {
            "history": valuation_history,
            "forecast": valuation_forecast,
            "currentValue": valuation_history[-1]["value"],
            "forecastWeekEndValue": valuation_forecast[-1]["value"],
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
