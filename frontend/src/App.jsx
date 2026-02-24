import { useEffect, useMemo, useState } from 'react'
import { ChartCanvas, Chart } from 'react-stockcharts'
import { LineSeries } from 'react-stockcharts/lib/series'
import { XAxis, YAxis } from 'react-stockcharts/lib/axes'
import { CrossHairCursor, MouseCoordinateX, MouseCoordinateY } from 'react-stockcharts/lib/coordinates'
import { discontinuousTimeScaleProvider } from 'react-stockcharts/lib/scale'

const STORAGE_KEY = 'sentidex-portfolio-v3'
const defaultPortfolio = [{ ticker: 'AAPL', shares: 4 }, { ticker: 'MSFT', shares: 2 }, { ticker: 'NVDA', shares: 1 }]

function toSeries(history = [], forecast = [], key = 'close') {
  return [...history.map((r) => ({ date: new Date(r.date), value: r[key] })), ...forecast.map((r) => ({ date: new Date(r.date), value: r[key] }))]
}

function StockChart({ title, history, forecast, valueKey = 'close' }) {
  const data = useMemo(() => toSeries(history, forecast, valueKey), [history, forecast, valueKey])
  if (!data.length) return <section className="card">No chart data.</section>
  const xScaleProvider = discontinuousTimeScaleProvider.inputDateAccessor((d) => d.date)
  const { data: chartData, xScale, xAccessor, displayXAccessor } = xScaleProvider(data)

  return (
    <section className="card chart-card">
      <h2>{title}</h2>
      <ChartCanvas width={980} height={420} ratio={1} margin={{ left: 48, right: 48, top: 16, bottom: 28 }} data={chartData} seriesName={title} xScale={xScale} xAccessor={xAccessor} displayXAccessor={displayXAccessor}>
        <Chart id={1} yExtents={(d) => d.value}>
          <XAxis axisAt="bottom" orient="bottom" />
          <YAxis axisAt="left" orient="left" />
          <MouseCoordinateX />
          <MouseCoordinateY />
          <LineSeries yAccessor={(d) => d.value} stroke="#2563eb" />
        </Chart>
        <CrossHairCursor />
      </ChartCanvas>
    </section>
  )
}

function GlobeTab({ keyword }) {
  const [data, setData] = useState(null)
  const [time, setTime] = useState(7)
  const [selected, setSelected] = useState(null)
  const [layers, setLayers] = useState({ companies: true, shippingRoutes: true, weather: true, macro: true, aiSignal: true })

  useEffect(() => {
    fetch(`/api/global-intelligence?keyword=${encodeURIComponent(keyword)}`)
      .then((r) => r.json())
      .then((j) => setData(j.data))
      .catch(() => setData(null))
  }, [keyword])

  const filtered = useMemo(() => {
    if (!data) return null
    const l = data.layers
    return {
      companies: (l.companies || []).filter((x) => x.dayOffset <= time),
      shippingRoutes: (l.shippingRoutes || []).filter((x) => x.dayOffset <= time),
      weather: (l.weather || []).filter((x) => x.dayOffset <= time),
      macro: Object.entries(l.macro || {}).filter(([, v]) => (v.dayOffset || 0) <= time),
      aiSignal: l.aiSignal,
    }
  }, [data, time])

  function pxX(x) { return x * 860 + 20 }
  function pxY(y) { return y * 420 + 20 }

  function toggle(k) { setLayers((p) => ({ ...p, [k]: !p[k] })) }

  return (
    <section className="card globe-wrap">
      <div className="row-between">
        <h2>Global Market Intelligence Globe</h2>
        <div>Time +{time}d</div>
      </div>
      <input type="range" min="0" max={data?.timeWindowDays || 14} value={time} onChange={(e) => setTime(Number(e.target.value))} />
      <div className="layer-toggles">
        {Object.keys(layers).map((k) => (
          <label key={k}><input type="checkbox" checked={layers[k]} onChange={() => toggle(k)} /> {k}</label>
        ))}
      </div>

      <svg viewBox="0 0 900 460" className="globe-svg">
        <defs>
          <radialGradient id="g" cx="50%" cy="45%" r="62%">
            <stop offset="0%" stopColor="#e0f2fe" />
            <stop offset="100%" stopColor="#93c5fd" />
          </radialGradient>
          <clipPath id="clip"><circle cx="450" cy="230" r="210" /></clipPath>
        </defs>
        <rect x="0" y="0" width="900" height="460" fill="#f8fafc" />
        <circle cx="450" cy="230" r="210" fill="url(#g)" stroke="#cbd5e1" strokeWidth="2" />

        <g clipPath="url(#clip)">
          {[...Array(9)].map((_, i) => <line key={`lon-${i}`} x1={i * 100 + 50} y1="20" x2={i * 100 + 50} y2="440" stroke="#bfdbfe" strokeWidth="1" />)}
          {[...Array(7)].map((_, i) => <line key={`lat-${i}`} x1="20" y1={i * 70 + 20} x2="880" y2={i * 70 + 20} stroke="#bfdbfe" strokeWidth="1" />)}

          {layers.shippingRoutes && filtered?.shippingRoutes?.map((r) => (
            <g key={r.name}>
              <line x1={pxX(r.from.x)} y1={pxY(r.from.y)} x2={pxX(r.to.x)} y2={pxY(r.to.y)} stroke={r.disruptionScore > 0.4 ? '#ef4444' : '#f59e0b'} strokeWidth="2" strokeDasharray="6 4" />
            </g>
          ))}

          {layers.weather && filtered?.weather?.map((w) => (
            <circle key={w.name} cx={pxX(w.x)} cy={pxY(w.y)} r={6 + w.severity * 8} fill="rgba(59,130,246,0.35)" stroke="#1d4ed8" />
          ))}

          {layers.companies && filtered?.companies?.map((c) => (
            <circle key={c.ticker} cx={pxX(c.x)} cy={pxY(c.y)} r={6 + c.confidence * 8} fill={c.direction === 'up' ? '#16a34a' : '#dc2626'} onClick={() => setSelected(c)} />
          ))}

          {layers.macro && filtered?.macro?.map(([k, m]) => (
            <rect key={k} x={pxX(m.x) - 8} y={pxY(m.y) - 8} width="16" height="16" fill="#0f172a" opacity="0.75" onClick={() => setSelected({ name: k, ...m, type: 'macro' })} />
          ))}
        </g>
      </svg>

      {layers.aiSignal && filtered?.aiSignal && (
        <div className="card ai-signal">
          <strong>AI Global Signal</strong>
          <div>Keyword: {filtered.aiSignal.keyword}</div>
          <div>Polymarket Probability: {filtered.aiSignal.polymarketProbability}</div>
          <div>CoinMarket Regime: {filtered.aiSignal.coinMarketRegime}</div>
          <div>Direction Score: {filtered.aiSignal.globalDirection}</div>
        </div>
      )}

      {selected && (
        <div className="card marker-detail">
          <h4>{selected.name || selected.ticker}</h4>
          {'ticker' in selected ? (
            <>
              <div>Ticker: {selected.ticker}</div>
              <div>Now: ${selected.priceNow}</div>
              <div>Forecast: ${selected.forecastPrice}</div>
              <div>Sentiment: {selected.sentiment}</div>
              <div>Confidence: {selected.confidence}</div>
              <div>Volatility: {selected.volatility}</div>
            </>
          ) : (
            <pre>{JSON.stringify(selected, null, 2)}</pre>
          )}
        </div>
      )}
    </section>
  )
}

function aggregateForecastFromSources(positions = [], selectedSources = []) {
  const selected = selectedSources.length ? selectedSources : ['CombinedSentiment']
  const byDate = {}
  for (const pos of positions) {
    const curves = pos.sourceForecasts || {}
    const activeCurves = selected.map((s) => curves[s]).filter(Boolean)
    if (!activeCurves.length) continue
    const perDate = {}
    for (const curve of activeCurves) {
      for (const row of curve) {
        if (!perDate[row.date]) perDate[row.date] = []
        perDate[row.date].push(row.predictedClose)
      }
    }
    for (const [date, vals] of Object.entries(perDate)) {
      const avg = vals.reduce((a, b) => a + b, 0) / vals.length
      byDate[date] = (byDate[date] || 0) + avg * pos.shares
    }
  }
  return Object.entries(byDate).sort((a, b) => a[0].localeCompare(b[0])).map(([date, value]) => ({ date, value: Number(value.toFixed(2)) }))
}

export function App() {
  const [page, setPage] = useState('dashboard')
  const [portfolio, setPortfolio] = useState(defaultPortfolio)
  const [portfolioData, setPortfolioData] = useState(null)
  const [popular, setPopular] = useState([])
  const [selectedTicker, setSelectedTicker] = useState('AAPL')
  const [lookup, setLookup] = useState('AAPL')
  const [keyword, setKeyword] = useState('interest rates')
  const [sortMode, setSortMode] = useState('best')
  const [selectedSources, setSelectedSources] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      try {
        const parsed = JSON.parse(raw)
        if (Array.isArray(parsed) && parsed.length) setPortfolio(parsed)
      } catch { }
    }
  }, [])
  useEffect(() => { localStorage.setItem(STORAGE_KEY, JSON.stringify(portfolio)) }, [portfolio])
  useEffect(() => { fetch('/api/quotes/popular').then((r) => r.json()).then((j) => setPopular(j.quotes || [])).catch(() => setPopular([])) }, [])

  async function refresh() {
    setLoading(true)
    try {
      const res = await fetch('/api/portfolio/forecast', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ positions: portfolio, sentiment_model: 'transformer', keyword }) })
      const json = await res.json()
      setPortfolioData(json)
      if (json?.positions?.length && !selectedTicker) setSelectedTicker(json.positions[0].ticker)
    } finally { setLoading(false) }
  }

  useEffect(() => { if (portfolio.length) refresh() }, [portfolio])

  const sourceOptions = useMemo(() => {
    const found = new Set()
    for (const p of portfolioData?.positions || []) Object.keys(p.sourceForecasts || {}).forEach((k) => found.add(k))
    return Array.from(found).sort()
  }, [portfolioData])
  useEffect(() => { if (sourceOptions.length && !selectedSources.length) setSelectedSources(['CombinedSentiment']) }, [sourceOptions])

  const forecastBySelectedSources = useMemo(() => aggregateForecastFromSources(portfolioData?.positions || [], selectedSources), [portfolioData, selectedSources])
  const sortedPositions = useMemo(() => {
    const list = [...(portfolioData?.positions || [])]
    if (sortMode === 'best') list.sort((a, b) => b.forecastDeltaPct - a.forecastDeltaPct)
    else if (sortMode === 'worst') list.sort((a, b) => a.forecastDeltaPct - b.forecastDeltaPct)
    else list.sort((a, b) => a.ticker.localeCompare(b.ticker))
    return list
  }, [portfolioData, sortMode])
  const selectedPosition = sortedPositions.find((x) => x.ticker === selectedTicker)

  function toggleSource(source) { setSelectedSources((p) => (p.includes(source) ? p.filter((x) => x !== source) : [...p, source])) }
  function addTicker() {
    const ticker = lookup.trim().toUpperCase()
    if (!ticker || portfolio.some((p) => p.ticker === ticker)) return
    setPortfolio((p) => [...p, { ticker, shares: 1 }])
  }

  return (
    <main className="minimal-dashboard">
      <header className="topbar">
        <div className="brand">Sentidex</div>
        <div className="lookup-row">
          <input value={lookup} onChange={(e) => setLookup(e.target.value.toUpperCase())} placeholder="Lookup ticker" />
          <input value={keyword} onChange={(e) => setKeyword(e.target.value)} placeholder="Global keyword (oil, rates, AAPL)" />
          <button onClick={addTicker}>Add</button>
          <button onClick={refresh} disabled={loading}>{loading ? 'Refreshing...' : 'Forecast'}</button>
        </div>
        <nav>
          <button className={page === 'dashboard' ? 'active' : ''} onClick={() => setPage('dashboard')}>Dashboard</button>
          <button className={page === 'portfolio' ? 'active' : ''} onClick={() => setPage('portfolio')}>Portfolio</button>
          <button className={page === 'globe' ? 'active' : ''} onClick={() => setPage('globe')}>Global Intel Globe</button>
        </nav>
      </header>

      <section className="popular-row">
        {popular.map((q) => <article key={q.ticker} className="ticker-card" onClick={() => setLookup(q.ticker)}><h4>{q.ticker}</h4><p>${q.price}</p></article>)}
      </section>

      {page !== 'globe' && (
        <section className="card source-dropdown">
          <h3>Data source overlays</h3>
          <details>
            <summary>Select one or many sources</summary>
            <div className="source-grid">
              {sourceOptions.map((source) => <label key={source}><input type="checkbox" checked={selectedSources.includes(source)} onChange={() => toggleSource(source)} /> {source}</label>)}
            </div>
          </details>
        </section>
      )}

      {page === 'dashboard' ? (
        <>
          <StockChart title={`Portfolio valuation weighted by: ${selectedSources.join(', ') || 'CombinedSentiment'}`} history={portfolioData?.portfolioValuation?.history || []} forecast={forecastBySelectedSources} valueKey="value" />
          <section className="card stats-grid">
            <div><h4>Current Value</h4><p>${portfolioData?.portfolioValuation?.currentValue ?? '-'}</p></div>
            <div><h4>Selected-Source Week End</h4><p>${forecastBySelectedSources?.[forecastBySelectedSources.length - 1]?.value ?? '-'}</p></div>
            <div><h4>Portfolio Size</h4><p>{portfolio.length}</p></div>
          </section>
        </>
      ) : page === 'portfolio' ? (
        <section className="portfolio-layout">
          <aside className="card portfolio-list">
            <div className="row-between">
              <h3>Positions</h3>
              <select value={sortMode} onChange={(e) => setSortMode(e.target.value)}>
                <option value="best">Best forecast</option>
                <option value="worst">Worst forecast</option>
                <option value="alpha">A-Z</option>
              </select>
            </div>
            {sortedPositions.map((p) => <button key={p.ticker} className="position-btn" onClick={() => setSelectedTicker(p.ticker)}><span>{p.ticker} ({p.shares} sh)</span><strong>{p.forecastDeltaPct}%</strong></button>)}
          </aside>
          <StockChart title={selectedPosition ? `${selectedPosition.ticker} chart` : 'Select a stock'} history={selectedPosition?.historicalPrices || []} forecast={(selectedPosition?.combinedForecast || []).map((r) => ({ date: r.date, close: r.predictedClose }))} valueKey="close" />
        </section>
      ) : (
        <GlobeTab keyword={keyword} />
      )}
    </main>
  )
}
