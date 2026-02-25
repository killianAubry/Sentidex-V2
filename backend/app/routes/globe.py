import os, json, asyncio, httpx
from pathlib import Path
from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv
from groq import AsyncGroq

# parents[0] = routes/, parents[1] = app/, parents[2] = backend/ (where .env lives)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

router = APIRouter()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY", "")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")

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
    except Exception:
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
    except Exception:
        return {"score": 0.3, "level": "low", "summary": "Analysis unavailable.", "headlines": headlines}


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

        return {
            "query": query,
            "nodes": enriched,
            "routes": routes,
            "openbb": openbb_data,
        }