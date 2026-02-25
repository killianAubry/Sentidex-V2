from __future__ import annotations

import httpx
from fastapi import APIRouter, Query
from datetime import datetime, timedelta

router = APIRouter()

HOUSE_DISCLOSURE_URL = "https://house-stock-watcher-data.s3-us-east-2.amazonaws.com/data/all_transactions.json"
SENATE_DISCLOSURE_URL = "https://senate-stock-watcher-data.s3-us-east-2.amazonaws.com/aggregate/all_transactions.json"

_cache: dict = {"house": None, "senate": None, "fetched_at": None}

async def _fetch_disclosures():
    now = datetime.utcnow()
    if _cache["fetched_at"] and (now - _cache["fetched_at"]) < timedelta(hours=6):
        return _cache["house"], _cache["senate"]

    async with httpx.AsyncClient(timeout=20) as client:
        try:
            h = await client.get(HOUSE_DISCLOSURE_URL)
            _cache["house"] = h.json()
        except Exception:
            _cache["house"] = []
        try:
            s = await client.get(SENATE_DISCLOSURE_URL)
            _cache["senate"] = s.json()
        except Exception:
            _cache["senate"] = []

    _cache["fetched_at"] = now
    return _cache["house"], _cache["senate"]


def _normalize_house(t: dict) -> dict:
    return {
        "source": "House",
        "member": t.get("representative", ""),
        "ticker": t.get("ticker", "").strip("$").upper(),
        "asset": t.get("asset_description", ""),
        "type": t.get("type", ""),
        "amount": t.get("amount", ""),
        "transaction_date": t.get("transaction_date", ""),
        "disclosure_date": t.get("disclosure_date", ""),
        "district": t.get("district", ""),
        "party": t.get("party", ""),
        "state": t.get("state", ""),
    }


def _normalize_senate(t: dict) -> dict:
    return {
        "source": "Senate",
        "member": t.get("senator", ""),
        "ticker": t.get("ticker", "").strip("$").upper(),
        "asset": t.get("asset_description", ""),
        "type": t.get("type", ""),
        "amount": t.get("amount", ""),
        "transaction_date": t.get("transaction_date", ""),
        "disclosure_date": t.get("disclosure_date", ""),
        "district": "",
        "party": t.get("party", ""),
        "state": t.get("state", ""),
    }


@router.get("/api/insider/trades")
async def get_trades(
    chamber: str = Query("all", description="house | senate | all"),
    ticker: str = Query("", description="Filter by ticker symbol"),
    member: str = Query("", description="Filter by member name"),
    trade_type: str = Query("", description="purchase | sale | all"),
    party: str = Query("", description="Democrat | Republican | all"),
    days: int = Query(90, description="Only show trades from last N days"),
    limit: int = Query(200),
    offset: int = Query(0),
):
    house_raw, senate_raw = await _fetch_disclosures()

    trades = []
    if chamber in ("house", "all"):
        trades += [_normalize_house(t) for t in (house_raw or []) if t.get("ticker")]
    if chamber in ("senate", "all"):
        trades += [_normalize_senate(t) for t in (senate_raw or []) if t.get("ticker")]

    # Date filter
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    trades = [t for t in trades if t["transaction_date"] >= cutoff]

    # Filters
    if ticker:
        trades = [t for t in trades if ticker.upper() in t["ticker"].upper()]
    if member:
        trades = [t for t in trades if member.lower() in t["member"].lower()]
    if trade_type and trade_type != "all":
        trades = [t for t in trades if trade_type.lower() in t["type"].lower()]
    if party and party != "all":
        trades = [t for t in trades if party.lower() in t["party"].lower()]

    # Sort newest first
    trades.sort(key=lambda t: t["transaction_date"], reverse=True)

    # Top movers — tickers bought/sold most
    ticker_counts: dict[str, dict] = {}
    for t in trades:
        tk = t["ticker"]
        if not tk or tk in ("--", "N/A"):
            continue
        if tk not in ticker_counts:
            ticker_counts[tk] = {"ticker": tk, "buys": 0, "sells": 0, "members": set()}
        if "purchase" in t["type"].lower():
            ticker_counts[tk]["buys"] += 1
        elif "sale" in t["type"].lower():
            ticker_counts[tk]["sells"] += 1
        ticker_counts[tk]["members"].add(t["member"])

    top_movers = sorted(
        [{"ticker": v["ticker"], "buys": v["buys"], "sells": v["sells"],
          "total": v["buys"] + v["sells"], "unique_members": len(v["members"])}
         for v in ticker_counts.values()],
        key=lambda x: x["total"], reverse=True
    )[:20]

    return {
        "trades": trades[offset: offset + limit],
        "total": len(trades),
        "top_movers": top_movers,
        "fetched_at": _cache["fetched_at"].isoformat() if _cache["fetched_at"] else None,
    }