from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Sentiment Forecast API")
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
}


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
        history.append(
            {
                "date": ts.strftime("%Y-%m-%d"),
                "close": round(float(close), 2),
            }
        )
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


async def fetch_news(ticker: str) -> list[dict[str, Any]]:
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return [
            {
                "source": {"name": "Reuters"},
                "title": f"{ticker} shows strong growth outlook as analysts upgrade sentiment",
                "description": "Investors optimistic on demand expansion and record margins.",
            },
            {
                "source": {"name": "Bloomberg"},
                "title": f"{ticker} faces short-term risk despite resilient fundamentals",
                "description": "Some funds warn about downside volatility and sector decline.",
            },
            {
                "source": {"name": "CNBC"},
                "title": f"Traders bullish as {ticker} may beat quarterly expectations",
                "description": "Potential gain if revenue surprise remains strong.",
            },
        ]

    params = {
        "q": f"{ticker} stock",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 80,
        "apiKey": api_key,
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get("https://newsapi.org/v2/everything", params=params)
        resp.raise_for_status()
        payload = resp.json()
    return payload.get("articles", [])


def sentiment_by_outlet(articles: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"sentimentScores": [], "headlines": []})

    for article in articles:
        outlet = article.get("source", {}).get("name") or "Unknown"
        title = article.get("title") or ""
        desc = article.get("description") or ""
        text = f"{title} {desc}".strip()
        score = lexicon_sentiment(text)
        grouped[outlet]["sentimentScores"].append(score)
        grouped[outlet]["headlines"].append(title)

    response: dict[str, dict[str, Any]] = {}
    for outlet, values in grouped.items():
        scores = values["sentimentScores"]
        avg = sum(scores) / max(len(scores), 1)
        response[outlet] = {
            "articleCount": len(scores),
            "avgSentiment": round(avg, 4),
            "sampleHeadlines": values["headlines"][:5],
        }
    return response


def project_week(history: list[dict[str, Any]], score: float) -> list[dict[str, Any]]:
    closes = [row["close"] for row in history[-10:]]
    last_close = closes[-1]
    volatility = (max(closes) - min(closes)) / max(last_close, 1)
    sentiment_drift = score * 0.35

    curve: list[dict[str, Any]] = []
    curr = last_close
    start_day = datetime.strptime(history[-1]["date"], "%Y-%m-%d")
    for i in range(1, 8):
        cyclical = math.sin(i / 2.3) * volatility * 0.2
        daily_return = sentiment_drift / 7 + cyclical
        curr = curr * (1 + daily_return)
        curve.append(
            {
                "date": (start_day + timedelta(days=i)).strftime("%Y-%m-%d"),
                "predictedClose": round(curr, 2),
            }
        )
    return curve


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@app.get("/api/forecast")
async def forecast(ticker: str = Query(..., min_length=1, max_length=10)) -> dict[str, Any]:
    ticker = ticker.upper().strip()
    history = fetch_month_prices(ticker)
    articles = await fetch_news(ticker)
    outlet_sentiments = sentiment_by_outlet(articles)

    predictions: dict[str, Any] = {}
    for outlet, details in outlet_sentiments.items():
        curve = project_week(history, details["avgSentiment"])
        predictions[outlet] = {
            **details,
            "nextWeekForecast": curve,
            "predictedWeekEndPrice": curve[-1]["predictedClose"],
        }

    if not predictions:
        raise HTTPException(status_code=404, detail="No articles found to compute sentiment")

    combined_score = sum(p["avgSentiment"] for p in predictions.values()) / len(predictions)
    combined_curve = project_week(history, combined_score)

    past_month_json = {
        "ticker": ticker,
        "window": "1m",
        "prices": history,
    }
    predictions_json = {
        "ticker": ticker,
        "generatedAt": datetime.now(UTC).isoformat(),
        "perOutletForecast": predictions,
    }
    combined_json = {
        "ticker": ticker,
        "generatedAt": datetime.now(UTC).isoformat(),
        "combinedSentimentScore": round(combined_score, 4),
        "historicalPrices": history,
        "combinedForecast": combined_curve,
        "perOutletForecast": predictions,
    }

    base_name = f"{ticker}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    prices_path = DATA_DIR / f"{base_name}_past_month_prices.json"
    outlets_path = DATA_DIR / f"{base_name}_per_outlet_predictions.json"
    combined_path = DATA_DIR / f"{base_name}_combined_chart_data.json"
    save_json(prices_path, past_month_json)
    save_json(outlets_path, predictions_json)
    save_json(combined_path, combined_json)

    return {
        "ticker": ticker,
        "files": {
            "pastMonthPrices": str(prices_path.relative_to(BASE_DIR)),
            "perOutletPredictions": str(outlets_path.relative_to(BASE_DIR)),
            "combinedChartData": str(combined_path.relative_to(BASE_DIR)),
        },
        "data": {
            "pastMonthPrices": past_month_json,
            "perOutletPredictions": predictions_json,
            "combinedChartData": combined_json,
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
