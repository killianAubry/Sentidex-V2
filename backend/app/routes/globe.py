import os, json, asyncio, httpx
from pathlib import Path
from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv
from groq import AsyncGroq
from app.logger import logger

# parents[0] = routes/, parents[1] = app/, parents[2] = backend/ (where .env lives)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

router = APIRouter()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY", "")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")
OPENBB_BASE = os.getenv("OPENBB_BASE", "http://127.0.0.1:6900/api/v1")

def _get_groq_client() -> AsyncGroq:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not set in .env")
    return AsyncGroq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# Step 1 – Ask Groq to map the supply chain
# ---------------------------------------------------------------------------
async def get_supply_chain_nodes(query: str, node_count: int = 12) -> list[dict]:
    groq_client = _get_groq_client()
    prompt = f"""
You are a supply chain intelligence analyst.
For the query "{query}", return a JSON array of the most important supply chain nodes.
Each node must have:
  - "name": place or facility name (string)
  - "type": one of [headquarters, manufacturing, port, warehouse, supplier, distribution, mine, farm]
  - "role": short description of role in this supply chain (string)
  - "country": country name (string)
  - "city": city name (string)
  - "connections": list of other node names this node ships to/from (array of strings)
  - "risk_keywords": list of risk keywords to search news for (array of strings)

Return ONLY the raw JSON array, no markdown, no explanation.
Limit to {node_count} nodes maximum.
"""
    response = await groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # replaces decommissioned llama3-70b-8192
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000,
    )
    raw = (response.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Step 2 – Geocode each node via Nominatim (free, no API key needed)
# ---------------------------------------------------------------------------
async def geocode_node(node: dict, client: httpx.AsyncClient) -> dict:
    queries = [
        f"{node.get('name', '')}, {node.get('city', '')}, {node.get('country', '')}",
        f"{node.get('city', '')}, {node.get('country', '')}",
    ]
    for query in queries:
        if not query.strip(', '):
            continue
        try:
            r = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "json", "limit": 1},
                headers={"User-Agent": "sentidex/2.0 (contact@sentidex.io)"},
                timeout=10,
            )
            results = r.json()
            if results:
                node["lat"] = float(results[0]["lat"])
                node["lon"] = float(results[0]["lon"])
                return node
        except Exception:
            continue

    logger.warning(f"Geocoding failed for node: {node.get('name')}, {node.get('city')}. Using fallback (0,0).")
    node["lat"] = 0.0
    node["lon"] = 0.0
    return node


async def geocode_all_nodes(nodes: list[dict], client: httpx.AsyncClient) -> list[dict]:
    """Geocode nodes sequentially with a 1.1s delay to respect Nominatim rate limit."""
    results = []
    for node in nodes:
        result = await geocode_node(node, client)
        results.append(result)
        await asyncio.sleep(1.1)  # Nominatim hard limit: 1 req/sec
    return results


# ---------------------------------------------------------------------------
# Step 3 – Weather for each node
# ---------------------------------------------------------------------------
async def get_weather(lat: float, lon: float, client: httpx.AsyncClient) -> dict:
    if not WEATHERAPI_KEY:
        logger.warning(f"WEATHERAPI_KEY missing. Returning fallback weather for {lat},{lon}")
        return {"condition": "Unknown", "temp_c": None, "icon": "🌐", "wind_kph": None}
    try:
        r = await client.get(
            "http://api.weatherapi.com/v1/current.json",
            params={"key": WEATHERAPI_KEY, "q": f"{lat},{lon}"},
            timeout=5,
        )
        d = r.json()["current"]
        cond = d["condition"]["text"].lower()
        icon = (
            "🌩️" if "thunder" in cond else
            "🌧️" if "rain" in cond else
            "❄️" if "snow" in cond else
            "🌫️" if "fog" in cond or "mist" in cond else
            "⛅" if "cloud" in cond else
            "☀️"
        )
        return {
            "condition": d["condition"]["text"],
            "temp_c": d["temp_c"],
            "wind_kph": d["wind_kph"],
            "icon": icon,
        }
    except Exception as e:
        logger.warning(f"Weather API call failed for {lat},{lon}: {e}")
        return {"condition": "Unknown", "temp_c": None, "icon": "🌐", "wind_kph": None}


# ---------------------------------------------------------------------------
# Step 4 – Risk analysis via Groq + news search
# ---------------------------------------------------------------------------
async def get_node_risk(node: dict, client: httpx.AsyncClient) -> dict:
    keywords = node.get("risk_keywords", [node.get("name", ""), node.get("country", "")])
    keyword_str = " OR ".join(keywords[:3])

    # Try GNews
    gnews_key = os.getenv("GNEWS_API_KEY", "")
    headlines = []
    if gnews_key:
        try:
            r = await client.get(
                "https://gnews.io/api/v4/search",
                params={"q": keyword_str, "lang": "en", "max": 5,
                        "apikey": gnews_key},
                timeout=6,
            )
            articles = r.json().get("articles", [])
            headlines = [a["title"] for a in articles]
        except Exception:
            pass

    if not headlines:
        logger.warning(f"No headlines found for risk analysis of node: {node.get('name')}")
        return {"score": 0.3, "level": "low", "summary": "No recent news found.", "headlines": []}

    prompt = f"""
You are a geopolitical risk analyst.
Node: {node['name']} ({node['type']}) in {node['city']}, {node['country']}
Role: {node['role']}
Recent headlines:
{chr(10).join(f'- {h}' for h in headlines)}

Return a JSON object with:
  - "score": float 0.0 (no risk) to 1.0 (critical risk)
  - "level": one of [low, medium, high, critical]
  - "summary": one sentence summary of main risk (string)

Return ONLY raw JSON, no markdown.
"""
    try:
        groq_client = _get_groq_client()
        resp = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  # replaces decommissioned llama3-8b-8192
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        risk = json.loads(raw)
        risk["headlines"] = headlines
        return risk
    except Exception as e:
        logger.warning(f"Risk analysis failed for node {node.get('name')}: {e}")
        return {"score": 0.3, "level": "low", "summary": "Analysis unavailable.", "headlines": headlines}


def _keyword_score(text: str, keywords: list[str]) -> float:
    lowered = text.lower()
    hits = sum(1 for k in keywords if k in lowered)
    return min(1.0, hits / max(len(keywords), 1))


async def get_trade_route_impacts(routes: list[dict], query: str, client: httpx.AsyncClient) -> list[dict]:
    if not routes:
        return []

    disruption_terms = ["disruption", "closure", "strike", "typhoon", "storm", "price", "delay", "blockade", "conflict"]
    top_routes = sorted(routes, key=lambda r: r.get("risk", 0), reverse=True)[:6]
    route_impacts = []
    for route in top_routes:
        route_name = f"{route['from']['name']} ↔ {route['to']['name']}"
        headline_pool = []
        if GNEWS_API_KEY:
            try:
                resp = await client.get(
                    "https://gnews.io/api/v4/search",
                    params={
                        "q": f"{route['from']['name']} {route['to']['name']} shipping route disruption OR canal OR port",
                        "lang": "en",
                        "max": 4,
                        "apikey": GNEWS_API_KEY,
                    },
                    timeout=6,
                )
                headline_pool = [a.get("title", "") for a in resp.json().get("articles", [])]
            except Exception:
                headline_pool = []

        if not headline_pool:
            logger.warning(f"No headlines found for trade route: {route_name}")
            headline_pool = [
                f"Monitoring route volatility for {route_name} in {query} supply chain",
                f"No major confirmed closure yet for {route_name}",
            ]

        text_blob = " ".join(headline_pool)
        disruption_score = max(route.get("risk", 0), _keyword_score(text_blob, disruption_terms))
        route_impacts.append({
            "route": route_name,
            "status": "impacted" if disruption_score >= 0.55 else "watch",
            "impact_score": round(disruption_score, 2),
            "reason": "Price, weather, and shipping-risk signals synthesized from route risk + headlines.",
            "headlines": headline_pool[:3],
        })

    return route_impacts


async def get_country_policy_impacts(nodes: list[dict], query: str, client: httpx.AsyncClient) -> list[dict]:
    countries = sorted({n.get("country", "").strip() for n in nodes if n.get("country")})[:8]
    results = []
    policy_terms = ["policy", "tariff", "export", "sanction", "regulation", "subsidy", "trade law"]
    for country in countries:
        headlines = []
        if GNEWS_API_KEY:
            try:
                r = await client.get(
                    "https://gnews.io/api/v4/search",
                    params={
                        "q": f"{country} trade policy supply chain {query}",
                        "lang": "en",
                        "max": 4,
                        "apikey": GNEWS_API_KEY,
                    },
                    timeout=6,
                )
                headlines = [a.get("title", "") for a in r.json().get("articles", [])]
            except Exception:
                headlines = []

        if not headlines:
            logger.warning(f"No headlines found for country policy: {country}")
            headlines = [f"No major confirmed policy shifts detected yet in {country} for {query}."]

        policy_score = _keyword_score(" ".join(headlines), policy_terms)
        results.append({
            "country": country,
            "status": "policy-change-risk" if policy_score >= 0.4 else "stable-watch",
            "policy_score": round(policy_score, 2),
            "summary": f"Country policy agent scanned new and draft policy signals for {country}.",
            "headlines": headlines[:3],
        })
    return results


def get_location_impacts(nodes: list[dict]) -> list[dict]:
    location_signals = []
    for node in nodes[:12]:
        risk = node.get("risk", {})
        weather = node.get("weather", {})
        category = "weather" if weather.get("wind_kph") and weather.get("wind_kph") > 40 else "news"
        if risk.get("score", 0) > 0.68:
            category = "security"
        location_signals.append({
            "location": f"{node.get('city', 'Unknown')}, {node.get('country', 'Unknown')}",
            "node": node.get("name", "Unknown"),
            "event_category": category,
            "impact_score": round(max(risk.get("score", 0), 0.2), 2),
            "summary": risk.get("summary", "Location signal detected."),
            "headlines": (risk.get("headlines") or [])[:2],
        })
    return sorted(location_signals, key=lambda x: x["impact_score"], reverse=True)


async def get_polymarket_signals(query: str, countries: list[str], client: httpx.AsyncClient) -> list[dict]:
    try:
        search = f"{query} {' '.join(countries[:3])}".strip()
        r = await client.get(
            "https://gamma-api.polymarket.com/events",
            params={"limit": 10, "active": "true", "closed": "false", "tag": "Politics", "query": search},
            timeout=8,
        )
        events = r.json() if isinstance(r.json(), list) else []
        signals = []
        for evt in events[:5]:
            title = evt.get("title") or evt.get("question") or "Polymarket event"
            slug = evt.get("slug", "")
            signals.append({
                "market": title,
                "probability": round(float(evt.get("volume24hr", 0) or 0) / 1_000_000, 2),
                "liquidity": evt.get("liquidity", 0),
                "status": "implicated" if any(c.lower() in title.lower() for c in countries[:3]) else "related",
                "url": f"https://polymarket.com/event/{slug}" if slug else "https://polymarket.com",
            })
        return signals
    except Exception as e:
        logger.warning(f"Polymarket API call failed for {query}: {e}")
        return [{
            "market": f"No live Polymarket data accessible for {query}; fallback probability weighting applied.",
            "probability": 0.5,
            "liquidity": 0,
            "status": "fallback",
            "url": "https://polymarket.com",
        }]


def build_dependency_graph(query: str, nodes: list[dict]) -> dict:
    critical_nodes = sorted(nodes, key=lambda n: n.get("risk", {}).get("score", 0), reverse=True)
    tier1 = [n["name"] for n in critical_nodes[:4]]
    tier2 = [n["name"] for n in critical_nodes[4:10]]
    commodities = [
        {"name": "Lithium", "dependency_score": 0.74},
        {"name": "Semiconductors", "dependency_score": 0.88},
        {"name": "Oil", "dependency_score": 0.66},
    ]

    heatmap = []
    for node in nodes[:12]:
        heatmap.append({
            "location": f"{node.get('city', 'Unknown')}, {node.get('country', 'Unknown')}",
            "lat": node.get("lat", 0),
            "lon": node.get("lon", 0),
            "concentration": round(0.4 + (node.get("risk", {}).get("score", 0) * 0.6), 2),
        })

    return {
        "query": query,
        "tier_1_suppliers": tier1,
        "tier_2_suppliers": tier2,
        "commodity_dependencies": commodities,
        "production_concentration_heatmap": heatmap,
    }


# ---------------------------------------------------------------------------
# Step 5 – OpenBB enrichment (price/fundamentals if ticker available)
# ---------------------------------------------------------------------------
async def get_openbb_data(query: str, client: httpx.AsyncClient) -> dict:
    try:
        r = await client.get(
            f"{OPENBB_BASE}/equity/search", # type: ignore
            params={"query": query, "limit": 1},
            timeout=5,
        )
        results = r.json().get("results", [])
        if not results:
            return {}
        ticker = results[0].get("symbol", "")
        if not ticker:
            return {}
        price_r = await client.get(
            f"{OPENBB_BASE}/equity/price/quote", # type: ignore
            params={"symbol": ticker},
            timeout=5,
        )
        price_data = price_r.json().get("results", [{}])[0]
        return {
            "ticker": ticker,
            "price": price_data.get("last_price"),
            "change_pct": price_data.get("change_percent"),
            "market_cap": price_data.get("market_cap"),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------
@router.get("/api/globe-supply-chain")
async def globe_supply_chain(query: str = "Apple", node_count: int = 12):
    logger.info(f"Starting supply chain analysis for query: {query}, nodes: {node_count}")
    async with httpx.AsyncClient() as client:
        nodes = await get_supply_chain_nodes(query, node_count)

        # 2. Geocode sequentially (rate limited), weather + risk in parallel
        geocoded = await geocode_all_nodes(nodes, client)

        weather_tasks = [get_weather(n["lat"], n["lon"], client) for n in geocoded]
        risk_tasks = [get_node_risk(n, client) for n in geocoded]
        openbb_task = get_openbb_data(query, client)

        weather_results, risk_results, openbb_data = await asyncio.gather(
            asyncio.gather(*weather_tasks),
            asyncio.gather(*risk_tasks),
            openbb_task,
        )

        # 3. Merge all data
        enriched = []
        for i, node in enumerate(geocoded):
            enriched.append({
                **node,
                "weather": weather_results[i],
                "risk": risk_results[i],
            })

        # 4. Build routes from connections
        name_to_idx = {n["name"]: i for i, n in enumerate(enriched)}
        routes = []
        seen = set()
        for node in enriched:
            for conn in node.get("connections", []):
                key = tuple(sorted([node["name"], conn]))
                if key not in seen and conn in name_to_idx:
                    seen.add(key)
                    target = enriched[name_to_idx[conn]]
                    routes.append({
                        "from": {"name": node["name"], "lat": node["lat"], "lon": node["lon"]},
                        "to": {"name": target["name"], "lat": target["lat"], "lon": target["lon"]},
                        "risk": max(node["risk"]["score"], target["risk"]["score"]),
                    })

        country_list = sorted({n.get("country", "") for n in enriched if n.get("country")})
        trade_route_system, country_system, polymarket_signals = await asyncio.gather(
            get_trade_route_impacts(routes, query, client),
            get_country_policy_impacts(enriched, query, client),
            get_polymarket_signals(query, country_list, client),
        )

        location_system = get_location_impacts(enriched)
        dependency_graph = build_dependency_graph(query, enriched)

        return {
            "query": query,
            "nodes": enriched,
            "routes": routes,
            "openbb": openbb_data,
            "systems": {
                "trade_route": trade_route_system,
                "country": country_system,
                "location": location_system,
            },
            "polymarket": polymarket_signals,
            "dependency_graph": dependency_graph,
        }
