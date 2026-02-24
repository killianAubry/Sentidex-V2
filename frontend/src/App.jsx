import { useEffect, useMemo, useState, useRef } from 'react'
import * as d3 from 'd3'
import * as d3geo from 'd3-geo'
import * as topojson from 'topojson-client'
import Highcharts from 'highcharts'
import HighchartsReact from 'highcharts-react-official'
import HighchartsMore from 'highcharts/highcharts-more'

if (typeof Highcharts === 'object') {
  HighchartsMore(Highcharts)
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

function GlobalMarketIntelligence({ keyword, setKeyword }) {
  const [data, setData] = useState(null)
  const [time, setTime] = useState(7)
  const [selected, setSelected] = useState(null)
  const [layers, setLayers] = useState({ companies: true, shippingRoutes: true, weather: true, macro: true, aiSignal: true })
  const svgRef = useRef(null)
  const [rotation, setRotation] = useState([45, -30])
  const [scale, setScale] = useState(250)
  const [land, setLand] = useState(null)

  useEffect(() => {
    // Load initial data on mount
    fetch(`/api/global-intelligence?keyword=${encodeURIComponent(keyword)}`)
      .then((r) => r.json())
      .then((j) => setData(j.data))
      .catch(() => setData(null))
  }, [])

  useEffect(() => {
    fetch('https://cdn.jsdelivr.net/npm/world-atlas@2/land-110m.json')
      .then(r => r.json())
      .then(world => {
        setLand(topojson.feature(world, world.objects.land))
      })
  }, [])

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

  // D3 Projection for 3D Globe
  const width = 800
  const height = 600
  const projection = d3geo.geoOrthographic()
    .scale(scale)
    .translate([width / 2, height / 2])
    .rotate(rotation)
    .clipAngle(90)

  const path = d3geo.geoPath(projection)

  // Drag and Zoom behavior
  useEffect(() => {
    const svg = d3.select(svgRef.current)

    const drag = d3.drag().on('drag', (event) => {
      setRotation(([r0, r1]) => [r0 + event.dx / 2, r1 - event.dy / 2])
    })

    const zoom = d3.zoom().on('zoom', (event) => {
      setScale(250 * event.transform.k)
    })

    svg.call(drag)
    svg.call(zoom)
  }, [])

  function toggle(k) { setLayers((p) => ({ ...p, [k]: !p[k] })) }

  const getLinePath = (from, to) => {
    const start = [from.lon, from.lat]
    const end = [to.lon, to.lat]
    return path({ type: 'LineString', coordinates: [start, end] })
  }

  return (
    <div className="gmi-page">
      <div className="gmi-header">
        <div className="search-bar">
          <input
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                // Trigger refresh or specific supply chain search if needed
                fetch(`/api/global-intelligence?keyword=${encodeURIComponent(keyword)}`)
                  .then((r) => r.json())
                  .then((j) => setData(j.data))
                  .catch(() => setData(null))
              }
            }}
            placeholder="Search commodities, companies, sectors... (Press Enter)"
          />
        </div>
        <div className="time-control">
          <span>Time: +{time}d</span>
          <input
            type="range"
            min="0"
            max={data?.timeWindowDays || 14}
            value={time}
            onChange={(e) => setTime(Number(e.target.value))}
          />
        </div>
      </div>

      <div className="gmi-layout">
        <div className="globe-container card">
          <svg ref={svgRef} viewBox={`0 0 ${width} ${height}`} className="globe-3d">
            <defs>
              <radialGradient id="globe-grad" cx="50%" cy="40%" r="50%">
                <stop offset="0%" stopColor="#1e293b" />
                <stop offset="100%" stopColor="#0f172a" />
              </radialGradient>
            </defs>
            <circle cx={width/2} cy={height/2} r={projection.scale()} fill="url(#globe-grad)" />

            {/* Graticule */}
            <path d={path(d3geo.geoGraticule()())} fill="none" stroke="#475569" strokeWidth="0.5" />

            {/* Land */}
            {land && (
              <path d={path(land)} fill="rgba(255,255,255,0.05)" stroke="#475569" strokeWidth="0.5" />
            )}

            {/* Supply Chain Routes */}
            {layers.shippingRoutes && filtered?.shippingRoutes?.map((r, i) => {
              const d = getLinePath(r.from, r.to)
              if (!d) return null
              return (
                <path
                  key={`route-${i}`}
                  d={d}
                  fill="none"
                    stroke={r.disruptionScore > 0.4 ? '#ef4444' : '#94a3b8'}
                    strokeWidth="1.5"
                    strokeDasharray="3 3"
                    opacity="0.6"
                />
              )
            })}

            {/* Nodes / Markers */}
            <g className="markers">
              {layers.companies && filtered?.companies?.map((c) => {
                const coords = projection([c.lon, c.lat])
                const isVisible = d3geo.geoDistance(projection.invert([width/2, height/2]), [c.lon, c.lat]) < Math.PI / 2
                if (!coords || !isVisible) return null
                return (
                  <g key={c.ticker} transform={`translate(${coords[0]-8},${coords[1]-8})`} onClick={() => setSelected(c)} className="node-marker">
                    <rect width="16" height="16" rx="2" fill={c.direction === 'up' ? '#22c55e' : '#ef4444'} />
                    <text x="8" y="12" fontSize="10" textAnchor="middle" fill="white" fontWeight="bold">B</text>
                  </g>
                )
              })}

              {layers.weather && filtered?.weather?.map((w, i) => {
                const coords = projection([w.lon, w.lat])
                const isVisible = d3geo.geoDistance(projection.invert([width/2, height/2]), [w.lon, w.lat]) < Math.PI / 2
                if (!coords || !isVisible) return null
                return (
                  <g key={`weather-${i}`} transform={`translate(${coords[0]-10},${coords[1]-10})`} onClick={() => setSelected({ ...w, type: 'weather' })} className="node-marker">
                    <circle r="10" cx="10" cy="10" fill="rgba(96, 165, 250, 0.2)" stroke="#60a5fa" strokeWidth="1" />
                    <text x="10" y="14" fontSize="12" textAnchor="middle">☁️</text>
                  </g>
                )
              })}

              {layers.macro && filtered?.macro?.map(([k, m], i) => {
                const coords = projection([m.lon, m.lat])
                const isVisible = d3geo.geoDistance(projection.invert([width/2, height/2]), [m.lon, m.lat]) < Math.PI / 2
                if (!coords || !isVisible) return null
                return (
                  <g key={`macro-${i}`} transform={`translate(${coords[0]-8},${coords[1]-8})`} onClick={() => setSelected({ name: k, ...m, type: 'macro' })} className="node-marker">
                    <circle r="8" cx="8" cy="8" fill="#f59e0b" stroke="#fff" strokeWidth="1" />
                    <text x="8" y="11" fontSize="10" textAnchor="middle" fill="white">M</text>
                  </g>
                )
              })}
            </g>
          </svg>
        </div>

        <aside className="gmi-sidebar">
          <div className="layer-controls card">
            <h4>Intelligence Layers</h4>
            <div className="layer-toggles">
              {Object.keys(layers).map((k) => (
                <label key={k} className="toggle-label">
                  <input type="checkbox" checked={layers[k]} onChange={() => toggle(k)} />
                  {k.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                </label>
              ))}
            </div>
          </div>

          {selected && (
            <div className="detail-panel card">
              <h4>{selected.name || selected.ticker}</h4>
              <div className="detail-content">
                <RiskMeter score={selected.riskScore || (selected.disruptionScore || 0.2)} />
                <div style={{ marginBottom: '1rem' }} />

                {selected.type === 'weather' ? (
                  <>
                    <p>Severity: {(selected.severity * 100).toFixed(0)}%</p>
                    {selected.tempC && <p>Temp: {selected.tempC}°C</p>}
                    {selected.precipitationMm && <p>Precipitation: {selected.precipitationMm}mm</p>}
                  </>
                ) : selected.type === 'macro' ? (
                  <>
                    <p>Interest Rate: {selected.interestRate}%</p>
                    <p>Inflation: {selected.inflation}%</p>
                    <p>GDP Growth: {selected.gdpGrowth}%</p>
                  </>
                ) : (
                  <>
                    <p>Price: ${selected.priceNow}</p>
                    <p>Forecast: ${selected.forecastPrice}</p>
                    <p>Sentiment: {selected.sentiment}</p>
                    <p>Confidence: {(selected.confidence * 100).toFixed(0)}%</p>
                  </>
                )}
              </div>
              <button className="close-btn" onClick={() => setSelected(null)}>Close</button>
            </div>
          )}

          {layers.aiSignal && filtered?.aiSignal && (
            <div className="ai-signal-panel card">
              <h4>AI Global Signal</h4>
              <div className="signal-data">
                <div className="signal-item">
                  <span className="label">Keyword</span>
                  <span className="value">{filtered.aiSignal.keyword}</span>
                </div>
                <div className="signal-item">
                  <span className="label">Polymarket Prob</span>
                  <span className="value">{(filtered.aiSignal.polymarketProbability * 100).toFixed(1)}%</span>
                </div>
                <div className="signal-item">
                  <span className="label">Global Direction</span>
                  <span className="value">{(filtered.aiSignal.globalDirection * 100).toFixed(1)}</span>
                </div>
              </div>
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
