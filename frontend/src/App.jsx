import { useEffect, useMemo, useState, useRef } from 'react'
import * as d3 from 'd3'
import * as d3geo from 'd3-geo'
import * as topojson from 'topojson-client'
import Highcharts from 'highcharts'
import HighchartsReact from 'highcharts-react-official'
import HighchartsMore from 'highcharts/highcharts-more'

// Fix: call the default export correctly
if (typeof Highcharts === 'object') {
  // eslint-disable-next-line no-undef
  import('highcharts/highcharts-more').then(m => {
    const fn = m.default || m
    if (typeof fn === 'function') fn(Highcharts)
  })
}


const STORAGE_KEY = 'sentidex-portfolio-v3'
const defaultPortfolio = [{ ticker: 'AAPL', shares: 4 }, { ticker: 'MSFT', shares: 2 }, { ticker: 'NVDA', shares: 1 }]

// --- Helper Functions ---
function toSeries(history = [], forecast = [], key = 'close') {
  return [
    ...history.map((r) => ({ date: new Date(r.date), value: r[key], type: 'history' })),
    ...forecast.map((r) => ({ date: new Date(r.date), value: r[key], type: 'forecast' }))
  ]
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

// --- Components ---

function StockChart({ title, history, forecast, valueKey = 'value', separateLines = [], showConfidence = false }) {
  const options = useMemo(() => {
    const histData = history.map(d => [new Date(d.date).getTime(), d[valueKey] || d.close])
    const foreData = forecast.map(d => [new Date(d.date).getTime(), d.predictedClose || d.value])

    // Create area range data (forecast +/- 5% for demo)
    const rangeData = forecast.map(d => {
      const val = d.predictedClose || d.value
      return [new Date(d.date).getTime(), val * 0.95, val * 1.05]
    })

    const series = [
      {
        name: 'History',
        data: histData,
        color: '#3b82f6',
        zIndex: 2
      },
      {
        name: 'Aggregate Forecast',
        data: foreData,
        color: '#60a5fa',
        dashStyle: 'ShortDash',
        zIndex: 3
      }
    ]

    if (showConfidence) {
      series.push({
        name: 'Confidence Range',
        data: rangeData,
        type: 'arearange',
        lineWidth: 0,
        linkedTo: ':previous',
        color: '#3b82f6',
        fillOpacity: 0.1,
        zIndex: 1,
        marker: { enabled: false }
      })
    }

    separateLines.forEach((line, i) => {
      series.push({
        name: `Source: ${line.name}`,
        data: line.data.map(d => [new Date(d.date).getTime(), d.value]),
        color: Highcharts.getOptions().colors[i + 2],
        dashStyle: 'Dot',
        zIndex: 2,
        opacity: 0.7
      })
    })

    return {
      chart: {
        backgroundColor: '#1e293b',
        style: { fontFamily: 'Inter, sans-serif' },
        height: 400
      },
      title: { text: null },
      xAxis: {
        type: 'datetime',
        gridLineColor: '#334155',
        labels: { style: { color: '#94a3b8' } }
      },
      yAxis: {
        title: { text: null },
        gridLineColor: '#334155',
        labels: { style: { color: '#94a3b8' } }
      },
      legend: { itemStyle: { color: '#94a3b8' } },
      tooltip: { shared: true },
      series
    }
  }, [history, forecast, valueKey, separateLines, showConfidence])

  if (!history.length && !forecast.length) return <div className="no-data">No chart data available.</div>

  return (
    <div className="chart-container">
      <HighchartsReact highcharts={Highcharts} options={options} />
    </div>
  )
}

const RiskMeter = ({ score }) => {
  const color = score > 0.7 ? '#ef4444' : score > 0.4 ? '#f59e0b' : '#22c55e'
  return (
    <div className="risk-meter-container">
      <div className="risk-meter-label">RISK LEVEL: {(score * 100).toFixed(0)}%</div>
      <div className="risk-meter-track">
        <div className="risk-meter-bar" style={{ width: `${score * 100}%`, backgroundColor: color }} />
      </div>
    </div>
  )
}

function PortfolioDashboard({
  portfolio,
  portfolioData,
  refresh,
  loading,
  lookup,
  setLookup,
  addTicker,
  selectedSources,
  toggleSource,
  sourceOptions,
  forecastBySelectedSources,
  viewStates,
  toggleViewState
}) {
  const sortedPositions = useMemo(() => {
    return [...(portfolioData?.positions || [])].sort((a, b) => b.forecastDeltaPct - a.forecastDeltaPct)
  }, [portfolioData])

  // Mock separate forecasts for visualization if enabled
  const separateLines = useMemo(() => {
    if (!viewStates.separateForecasts) return []
    return selectedSources.map(source => ({
      name: source,
      data: aggregateForecastFromSources(portfolioData?.positions || [], [source])
    }))
  }, [portfolioData, selectedSources, viewStates.separateForecasts])

  return (
    <div className="portfolio-dashboard">
      <div className="dashboard-main">
        <div className="dashboard-header">
          <div className="ticker-lookup">
            <input
              value={lookup}
              onChange={(e) => setLookup(e.target.value.toUpperCase())}
              placeholder="Enter Ticker (e.g. AAPL)"
            />
            <button onClick={addTicker}>Add to Portfolio</button>
            <button className="refresh-btn" onClick={refresh} disabled={loading}>
              {loading ? 'Updating...' : 'Refresh Forecasts'}
            </button>
          </div>
        </div>

        <div className="dashboard-content">
          <div className="center-screen">
            <div className="chart-card card">
              <div className="card-header">
                <h3>Portfolio Valuation Forecast</h3>
                <span className="source-label">Sources: {selectedSources.join(', ')}</span>
              </div>
              <StockChart
                title="Portfolio Valuation"
                history={portfolioData?.portfolioValuation?.history || []}
                forecast={forecastBySelectedSources}
                valueKey="value"
                separateLines={separateLines}
                showConfidence={viewStates.showConfidence}
              />
            </div>

            <div className="command-center card">
              <h4>Command Center</h4>
              <div className="controls-grid">
                <div className="control-group">
                  <h5>Data Source Overlays</h5>
                  <div className="source-toggles">
                    {sourceOptions.map((source) => (
                      <label key={source} className="toggle-label">
                        <input
                          type="checkbox"
                          checked={selectedSources.includes(source)}
                          onChange={() => toggleSource(source)}
                        />
                        {source}
                      </label>
                    ))}
                  </div>
                </div>
                <div className="control-group">
                  <h5>Views & Analysis</h5>
                  <div className="view-buttons">
                    <button
                      className={`small-btn ${viewStates.showConfidence ? 'active' : ''}`}
                      onClick={() => toggleViewState('showConfidence')}
                    >
                      Show Confidence Levels
                    </button>
                    <button
                      className={`small-btn ${viewStates.separateForecasts ? 'active' : ''}`}
                      onClick={() => toggleViewState('separateForecasts')}
                    >
                      Separate Forecasts
                    </button>
                    <button
                      className={`small-btn ${viewStates.macroImpact ? 'active' : ''}`}
                      onClick={() => toggleViewState('macroImpact')}
                    >
                      Macro Impact
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <aside className="portfolio-sidebar card">
            <h3>Current Holdings</h3>
            <div className="holdings-list">
              {sortedPositions.length > 0 ? (
                sortedPositions.map((p) => (
                  <div key={p.ticker} className="holding-item">
                    <div className="holding-info">
                      <span className="ticker">{p.ticker}</span>
                      <span className="shares">{p.shares} shares</span>
                    </div>
                    <div className={`forecast-delta ${p.forecastDeltaPct >= 0 ? 'pos' : 'neg'}`}>
                      {p.forecastDeltaPct >= 0 ? '▲' : '▼'} {Math.abs(p.forecastDeltaPct)}%
                    </div>
                  </div>
                ))
              ) : (
                <div className="empty-portfolio">No stocks in portfolio.</div>
              )}
            </div>
          </aside>
        </div>
      </div>
    </div>
  )
}

const NODE_ICONS = {
  headquarters: '🏢',
  manufacturing: '🏭',
  port: '⚓',
  warehouse: '📦',
  supplier: '🔩',
  distribution: '🚚',
  mine: '⛏️',
  farm: '🌾',
  default: '📍',
}

const RISK_COLORS = {
  low: '#22c55e',
  medium: '#f59e0b',
  high: '#ef4444',
  critical: '#7c3aed',
}

function GlobalMarketIntelligence({ keyword, setKeyword }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState(null)
  const [layers, setLayers] = useState({ nodes: true, routes: true, weather: true, risk: true })
  const svgRef = useRef(null)
  const [rotation, setRotation] = useState([0, -30])
  const [scale, setScale] = useState(250)
  const [land, setLand] = useState(null)
  const [inputVal, setInputVal] = useState(keyword)

  // Load world topology
  useEffect(() => {
    fetch('https://cdn.jsdelivr.net/npm/world-atlas@2/land-110m.json')
      .then(r => r.json())
      .then(world => setLand(topojson.feature(world, world.objects.land)))
  }, [])

  // Drag + zoom on globe
  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.call(
      d3.drag().on('drag', (e) =>
        setRotation(([r0, r1]) => [r0 + e.dx / 2, r1 - e.dy / 2])
      )
    )
    svg.call(
      d3.zoom().scaleExtent([0.5, 4]).on('zoom', (e) =>
        setScale(250 * e.transform.k)
      )
    )
  }, [])

  async function runSearch(q) {
    setLoading(true)
    setSelected(null)
    setData(null)
    try {
      const r = await fetch(`/api/globe-supply-chain?query=${encodeURIComponent(q)}`)
      const j = await r.json()
      setData(j)
      setKeyword(q)
      // Auto-rotate to first node
      if (j.nodes?.length) {
        setRotation([-j.nodes[0].lon, -j.nodes[0].lat + 20])
      }
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  const width = 800
  const height = 580

  const projection = useMemo(() =>
    d3geo.geoOrthographic()
      .scale(scale)
      .translate([width / 2, height / 2])
      .rotate(rotation)
      .clipAngle(90),
    [scale, rotation]
  )

  const path = useMemo(() => d3geo.geoPath(projection), [projection])

  function isVisible(lon, lat) {
    try {
      const center = projection.invert([width / 2, height / 2])
      return d3geo.geoDistance(center, [lon, lat]) < Math.PI / 2
    } catch { return false }
  }

  function getRoutePath(from, to) {
    try {
      return path({ type: 'LineString', coordinates: [[from.lon, from.lat], [to.lon, to.lat]] })
    } catch { return null }
  }

  return (
    <div className="gmi-page">
      {/* Header */}
      <div className="gmi-header">
        <div className="search-bar" style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <input
            value={inputVal}
            onChange={e => setInputVal(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') runSearch(inputVal) }}
            placeholder="Enter company, commodity or sector... (Press Enter)"
            style={{ flex: 1 }}
          />
          <button onClick={() => runSearch(inputVal)} disabled={loading} className="refresh-btn">
            {loading ? '🔄 Analyzing...' : '🔍 Analyze Supply Chain'}
          </button>
        </div>

        {data?.openbb?.ticker && (
          <div className="openbb-bar">
            <span>{data.openbb.ticker}</span>
            <span>${data.openbb.price?.toFixed(2)}</span>
            <span style={{ color: data.openbb.change_pct >= 0 ? '#22c55e' : '#ef4444' }}>
              {data.openbb.change_pct >= 0 ? '▲' : '▼'} {Math.abs(data.openbb.change_pct ?? 0).toFixed(2)}%
            </span>
          </div>
        )}
      </div>

      <div className="gmi-layout">
        {/* Globe */}
        <div className="globe-container card" style={{ position: 'relative' }}>
          {loading && (
            <div style={{
              position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center', background: 'rgba(15,23,42,0.85)',
              zIndex: 10, borderRadius: '1rem', gap: '1rem'
            }}>
              <div style={{ fontSize: '2rem' }}>🌐</div>
              <div style={{ color: '#60a5fa', fontSize: '1rem' }}>AI is mapping the supply chain...</div>
              <div style={{ color: '#94a3b8', fontSize: '0.8rem' }}>Querying Groq → Geocoding → Risk Analysis</div>
            </div>
          )}

          <svg ref={svgRef} viewBox={`0 0 ${width} ${height}`} className="globe-3d">
            <defs>
              <radialGradient id="globe-grad" cx="45%" cy="35%" r="55%">
                <stop offset="0%" stopColor="#1e3a5f" />
                <stop offset="100%" stopColor="#0a0f1e" />
              </radialGradient>
            </defs>

            {/* Ocean */}
            <circle cx={width / 2} cy={height / 2} r={scale} fill="url(#globe-grad)" />

            {/* Graticule */}
            <path d={path(d3geo.geoGraticule()())} fill="none" stroke="#1e3a5f" strokeWidth="0.5" opacity="0.8" />

            {/* Land */}
            {land && <path d={path(land)} fill="#1e293b" stroke="#334155" strokeWidth="0.8" />}

            {/* Routes */}
            {layers.routes && data?.routes?.map((r, i) => {
              const d = getRoutePath(r.from, r.to)
              if (!d) return null
              const color = r.risk > 0.6 ? '#ef4444' : r.risk > 0.3 ? '#f59e0b' : '#22c55e'
              return (
                <path key={`route-${i}`} d={d} fill="none"
                  stroke={color} strokeWidth="1.5"
                  strokeDasharray="5 4" opacity="0.75"
                />
              )
            })}

            {/* Nodes */}
            {layers.nodes && data?.nodes?.map((node, i) => {
              if (!isVisible(node.lon, node.lat)) return null
              const coords = projection([node.lon, node.lat])
              if (!coords) return null
              const riskColor = RISK_COLORS[node.risk?.level] || '#94a3b8'
              const icon = NODE_ICONS[node.type] || NODE_ICONS.default

              return (
                <g key={`node-${i}`} transform={`translate(${coords[0]},${coords[1]})`}
                  onClick={() => setSelected(node)} className="node-marker" style={{ cursor: 'pointer' }}>

                  {/* Risk pulse ring */}
                  {layers.risk && (
                    <circle r="18" fill="none" stroke={riskColor} strokeWidth="1.5" opacity="0.4" />
                  )}

                  {/* Node background */}
                  <circle r="12" fill="#0f172a" stroke={riskColor} strokeWidth="2" />

                  {/* Icon */}
                  <text textAnchor="middle" dominantBaseline="central" fontSize="12">{icon}</text>

                  {/* Weather icon */}
                  {layers.weather && node.weather?.icon && (
                    <text x="14" y="-14" fontSize="11" textAnchor="middle">{node.weather.icon}</text>
                  )}

                  {/* Risk score badge */}
                  {layers.risk && node.risk?.score > 0.4 && (
                    <g transform="translate(10,-10)">
                      <circle r="6" fill={riskColor} />
                      <text textAnchor="middle" dominantBaseline="central" fontSize="7" fill="white" fontWeight="bold">
                        {Math.round(node.risk.score * 10)}
                      </text>
                    </g>
                  )}

                  {/* Name label */}
                  <text y="24" textAnchor="middle" fontSize="9" fill="#94a3b8" style={{ pointerEvents: 'none' }}>
                    {node.city || node.name}
                  </text>
                </g>
              )
            })}
          </svg>

          {/* Layer toggles overlay */}
          <div style={{
            position: 'absolute', bottom: '1rem', left: '1rem',
            display: 'flex', gap: '0.5rem', flexWrap: 'wrap'
          }}>
            {Object.keys(layers).map(k => (
              <button key={k}
                onClick={() => setLayers(p => ({ ...p, [k]: !p[k] }))}
                style={{
                  padding: '0.25rem 0.6rem', fontSize: '0.7rem', borderRadius: '999px',
                  background: layers[k] ? '#3b82f6' : '#1e293b',
                  border: '1px solid #334155', color: 'white', cursor: 'pointer'
                }}>
                {k.charAt(0).toUpperCase() + k.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Sidebar */}
        <aside className="gmi-sidebar">
          {/* Node detail panel */}
          {selected && (
            <div className="detail-panel card" style={{ marginBottom: '1rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <h4 style={{ margin: 0 }}>
                    {NODE_ICONS[selected.type] || '📍'} {selected.name}
                  </h4>
                  <div style={{ color: '#60a5fa', fontSize: '0.8rem' }}>{selected.type} · {selected.city}, {selected.country}</div>
                </div>
                <button className="close-btn" onClick={() => setSelected(null)}>✕</button>
              </div>

              <p style={{ color: '#94a3b8', fontSize: '0.85rem', margin: '0.5rem 0' }}>{selected.role}</p>

              {/* Risk */}
              <div style={{ marginBottom: '0.75rem' }}>
                <div style={{ color: RISK_COLORS[selected.risk?.level], fontWeight: 'bold', fontSize: '0.85rem', marginBottom: '0.25rem' }}>
                  Risk: {selected.risk?.level?.toUpperCase()} ({Math.round((selected.risk?.score || 0) * 100)}%)
                </div>
                <RiskMeter score={selected.risk?.score || 0} />
                <p style={{ color: '#94a3b8', fontSize: '0.8rem', marginTop: '0.4rem' }}>{selected.risk?.summary}</p>
              </div>

              {/* Weather */}
              {selected.weather && (
                <div style={{ background: '#0f172a', borderRadius: '0.5rem', padding: '0.5rem', marginBottom: '0.75rem' }}>
                  <div style={{ color: '#94a3b8', fontSize: '0.8rem' }}>
                    {selected.weather.icon} {selected.weather.condition}
                    {selected.weather.temp_c != null && ` · ${selected.weather.temp_c}°C`}
                    {selected.weather.wind_kph != null && ` · 💨 ${selected.weather.wind_kph} km/h`}
                  </div>
                </div>
              )}

              {/* Headlines */}
              {selected.risk?.headlines?.length > 0 && (
                <div>
                  <div style={{ color: '#64748b', fontSize: '0.75rem', marginBottom: '0.4rem' }}>RECENT NEWS</div>
                  {selected.risk.headlines.slice(0, 3).map((h, i) => (
                    <div key={i} style={{
                      fontSize: '0.75rem', color: '#94a3b8', padding: '0.3rem 0',
                      borderBottom: '1px solid #1e293b'
                    }}>• {h}</div>
                  ))}
                </div>
              )}

              {/* Connections */}
              {selected.connections?.length > 0 && (
                <div style={{ marginTop: '0.75rem' }}>
                  <div style={{ color: '#64748b', fontSize: '0.75rem', marginBottom: '0.3rem' }}>CONNECTS TO</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem' }}>
                    {selected.connections.map(c => (
                      <span key={c} style={{
                        fontSize: '0.7rem', padding: '0.15rem 0.4rem',
                        background: '#1e293b', borderRadius: '999px', color: '#60a5fa'
                      }}>{c}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Node list */}
          {data?.nodes && (
            <div className="card" style={{ padding: '1rem' }}>
              <h4 style={{ margin: '0 0 0.75rem' }}>Supply Chain Nodes ({data.nodes.length})</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', maxHeight: '350px', overflowY: 'auto' }}>
                {[...data.nodes]
                  .sort((a, b) => (b.risk?.score || 0) - (a.risk?.score || 0))
                  .map((node, i) => (
                    <div key={i}
                      onClick={() => {
                        setSelected(node)
                        setRotation([-node.lon, -node.lat + 20])
                      }}
                      style={{
                        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                        padding: '0.4rem 0.6rem', background: '#0f172a', borderRadius: '0.4rem',
                        cursor: 'pointer', border: `1px solid ${RISK_COLORS[node.risk?.level] || '#334155'}22`
                      }}>
                      <div>
                        <span style={{ marginRight: '0.4rem' }}>{NODE_ICONS[node.type] || '📍'}</span>
                        <span style={{ fontSize: '0.8rem', color: '#e2e8f0' }}>{node.name}</span>
                        <span style={{ fontSize: '0.7rem', color: '#64748b', marginLeft: '0.4rem' }}>{node.city}</span>
                      </div>
                      <span style={{
                        fontSize: '0.7rem', padding: '0.1rem 0.4rem', borderRadius: '999px',
                        background: RISK_COLORS[node.risk?.level] + '33',
                        color: RISK_COLORS[node.risk?.level]
                      }}>{node.risk?.level}</span>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {!data && !loading && (
            <div className="card" style={{ padding: '2rem', textAlign: 'center', color: '#64748b' }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🌐</div>
              <div>Enter a company or commodity above and press Enter to map its global supply chain using AI.</div>
            </div>
          )}
        </aside>
      </div>
    </div>
  )
}

export function App() {
  const [page, setPage] = useState('portfolio')
  const [portfolio, setPortfolio] = useState(defaultPortfolio)
  const [portfolioData, setPortfolioData] = useState(null)
  const [lookup, setLookup] = useState('AAPL')
  const [keyword, setKeyword] = useState('interest rates')
  const [selectedSources, setSelectedSources] = useState(['CombinedSentiment'])
  const [loading, setLoading] = useState(false)
  const [viewStates, setViewStates] = useState({
    showConfidence: false,
    separateForecasts: false,
    macroImpact: false
  })

  function toggleViewState(key) {
    setViewStates(prev => ({ ...prev, [key]: !prev[key] }))
  }

  useEffect(() => {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      try {
        const parsed = JSON.parse(raw)
        if (Array.isArray(parsed) && parsed.length) setPortfolio(parsed)
      } catch { }
    }
  }, [])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(portfolio))
  }, [portfolio])

  async function refresh() {
    setLoading(true)
    try {
      const res = await fetch('/api/portfolio/forecast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ positions: portfolio, sentiment_model: 'transformer', keyword })
      })
      const json = await res.json()
      setPortfolioData(json)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (portfolio.length) refresh()
  }, [portfolio])

  const sourceOptions = useMemo(() => {
    const found = new Set()
    for (const p of portfolioData?.positions || []) {
      Object.keys(p.sourceForecasts || {}).forEach((k) => found.add(k))
    }
    return Array.from(found).sort()
  }, [portfolioData])

  const forecastBySelectedSources = useMemo(() =>
    aggregateForecastFromSources(portfolioData?.positions || [], selectedSources),
  [portfolioData, selectedSources])

  function toggleSource(source) {
    setSelectedSources((p) => (p.includes(source) ? p.filter((x) => x !== source) : [...p, source]))
  }

  function addTicker() {
    const ticker = lookup.trim().toUpperCase()
    if (!ticker || portfolio.some((p) => p.ticker === ticker)) return
    setPortfolio((p) => [...p, { ticker, shares: 1 }])
  }

  return (
    <div className="app-container">
      <nav className="navbar">
        <div className="nav-brand">SENTIDEX <span className="pro-tag">PRO</span></div>
        <div className="nav-links">
          <button
            className={page === 'portfolio' ? 'active' : ''}
            onClick={() => setPage('portfolio')}
          >
            Portfolio
          </button>
          <button
            className={page === 'gmi' ? 'active' : ''}
            onClick={() => setPage('gmi')}
          >
            GMI
          </button>
        </div>
      </nav>

      <main className="main-content">
        {page === 'portfolio' ? (
          <PortfolioDashboard
            portfolio={portfolio}
            portfolioData={portfolioData}
            refresh={refresh}
            loading={loading}
            lookup={lookup}
            setLookup={setLookup}
            addTicker={addTicker}
            selectedSources={selectedSources}
            toggleSource={toggleSource}
            sourceOptions={sourceOptions}
            forecastBySelectedSources={forecastBySelectedSources}
            viewStates={viewStates}
            toggleViewState={toggleViewState}
          />
        ) : (
          <GlobalMarketIntelligence
            keyword={keyword}
            setKeyword={setKeyword}
          />
        )}
      </main>
    </div>
  )
}
