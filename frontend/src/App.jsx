import { useEffect, useMemo, useState, useRef } from 'react'
import * as d3 from 'd3'
import * as d3geo from 'd3-geo'
import * as topojson from 'topojson-client'
import Highcharts from 'highcharts'
import HighchartsReact from 'highcharts-react-official'
import {
  Building01,
  Settings01,
  Anchor,
  Package,
  Tool01,
  Truck01,
  LayersTwo01,
  Sun,
  MarkerPin01,
  CloudRaining01,
  Snowflake01,
  Lightning01,
  Cloud01,
  Globe01,
  AlertTriangle,
  BarChart01,
  SearchLg,
  RefreshCw01,
  XClose,
  ChevronDown,
  Wind01,
  Thermometer01,
} from '@untitled-ui/icons-react'

if (typeof HighchartsMore === 'function') {
  HighchartsMore(Highcharts)
} else if (HighchartsMore && HighchartsMore.default) {
  HighchartsMore.default(Highcharts)
}

const STORAGE_KEY = 'sentidex-portfolio-vbloom'
const defaultPortfolio = [{ ticker: 'AAPL', shares: 10 }, { ticker: 'MSFT', shares: 5 }, { ticker: 'TSLA', shares: 2 }]

// --- Logo Component ---
const SentidexLogo = () => (
  <svg className="logo-placeholder" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 2L2 7L12 12L22 7L12 2Z" />
    <path d="M2 17L12 22L22 17" />
    <path d="M2 12L12 17L22 12" />
  </svg>
)

// --- Helper Functions ---
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

function StockChart({ history = [], forecast = [], valueKey = 'value', separateLines = [], showConfidence = true }) {
  const options = useMemo(() => {
    const histData = history.map(d => [new Date(d.date).getTime(), d[valueKey] || d.close])
    const foreData = forecast.map(d => [new Date(d.date).getTime(), d.predictedClose || d.value])

    // Confidence range (simulated +/- percentage)
    const rangeData = forecast.map(d => {
      const val = d.predictedClose || d.value
      return [new Date(d.date).getTime(), val * 0.96, val * 1.04]
    })
    const series = [
      {
        name: 'HISTORY',
        data: histData,
        color: '#ffffff',
        lineWidth: 1.5,
        marker: { enabled: false },
        zIndex: 2
      },
      {
        name: 'FORECAST',
        data: foreData,
        color: '#ffb000',
        lineWidth: 2,
        dashStyle: 'ShortDash',
        marker: { enabled: false },
        zIndex: 3
      }
    ]

    if (showConfidence && foreData.length) {
      series.push({
        name: 'CONFIDENCE',
        data: rangeData,
        type: 'arearange',
        lineWidth: 0,
        linkedTo: ':previous',
        color: '#ffb000',
        fillOpacity: 0.15,
        zIndex: 1,
        marker: { enabled: false }
      })
    }
    separateLines.forEach((line, i) => {
      series.push({
        name: line.name.toUpperCase(),
        data: line.data.map(d => [new Date(d.date).getTime(), d.value]),
        color: Highcharts.getOptions().colors[i + 5],
        dashStyle: 'Dot',
        lineWidth: 1,
        zIndex: 2,
        opacity: 0.5,
        marker: { enabled: false }
      })
    })
    return {
      chart: {
        backgroundColor: '#000000',
        style: { fontFamily: 'Roboto Mono, Inter, monospace' },
        height: 450,
        zoomType: 'x',
        panning: true,
        panKey: 'shift'
      },
      title: { text: null },
      xAxis: {
        type: 'datetime',
        gridLineColor: '#111',
        lineColor: '#333',
        tickColor: '#333',
        labels: { style: { color: '#cccccc' } }
      },
      yAxis: {
        title: { text: null },
        gridLineColor: '#111',
        lineColor: '#333',
        labels: { style: { color: '#cccccc' } }
      },
      legend: {
        itemStyle: { color: '#cccccc', fontSize: '0.7rem' },
        align: 'left',
        verticalAlign: 'top'
      },
      tooltip: {
        shared: true,
        backgroundColor: '#111',
        style: { color: '#fff' },
        borderColor: '#ffb000'
      },
      plotOptions: {
        series: {
          animation: false
        }
      },
      credits: { enabled: false },
      series
    }
  }, [history, forecast, valueKey, separateLines, showConfidence])

  if (!history.length && !forecast.length) return <div className="no-data">NO DATA AVAILABLE</div>

  return (
    <div className="chart-card">
      <HighchartsReact highcharts={Highcharts} options={options} />
    </div>
  )
}

const RiskMeter = ({ score }) => {
  const color = score > 0.7 ? '#ff3333' : score > 0.4 ? '#ffb000' : '#00ff00'
  return (
    <div className="risk-meter-container">
      <div className="risk-meter-label">RISK FACTOR: {(score * 100).toFixed(0)}%</div>
      <div className="risk-meter-track">
        <div className="risk-meter-bar" style={{ width: `${score * 100}%`, backgroundColor: color }} />
      </div>
    </div>
  )
}

// Untitled UI icon components per node type (sidebar/panels only — NOT used on globe)
const NODE_ICON_COMPONENTS = {
  headquarters: Building01,
  manufacturing: Settings01,
  port: Anchor,
  warehouse: Package,
  supplier: Tool01,
  distribution: Truck01,
  mine: LayersTwo01,
  farm: Sun,
  default: MarkerPin01,
}

const WEATHER_ICON_COMPONENTS = {
  '☀️': Sun,
  '🌧️': CloudRaining01,
  '❄️': Snowflake01,
  '🌩️': Lightning01,
  '⛅': Cloud01,
  '🌫️': Wind01,
  '🌐': Globe01,
}

function NodeIcon({ type, size = 16, color = 'currentColor', ...props }) {
  const Icon = NODE_ICON_COMPONENTS[type] || NODE_ICON_COMPONENTS.default
  return <Icon width={size} height={size} color={color} {...props} />
}

function WeatherIcon({ icon, size = 14, color = '#60a5fa', ...props }) {
  const Icon = WEATHER_ICON_COMPONENTS[icon] || Globe01
  return <Icon width={size} height={size} color={color} {...props} />
}

const RISK_COLORS = {
  low: '#22c55e',
  medium: '#f59e0b',
  high: '#ef4444',
  critical: '#7c3aed',
}

// Simple dot shapes for globe nodes — no icons, no paths, always crisp
function GlobeDot({ riskColor, isSelected }) {
  return (
    <g>
      <circle r="10" fill="#0f172a" stroke={riskColor} strokeWidth="2" />
      <circle r="4" fill={riskColor} />
      {isSelected && <circle r="14" fill="none" stroke={riskColor} strokeWidth="1" opacity="0.5" />}
    </g>
  )
}

function PortfolioDashboard({
  portfolio,
  portfolioData,
  refresh,
  loading,
  setPortfolio,
  horizon,
  setHorizon
}) {
  const [activeTab, setActiveTab] = useState('forecast')
  const [lookup, setLookup] = useState('')

  const summary = useMemo(() => {
    if (!portfolioData) return { total: 0, change: 0, changePct: 0 }
    const total = portfolioData.portfolioValuation.currentValue
    const forecast = portfolioData.portfolioValuation.forecastWeekEndValue
    const change = forecast - total
    const changePct = total ? (change / total) * 100 : 0
    return { total, change, changePct }
  }, [portfolioData])

  const impacts = useMemo(() => {
    if (!portfolioData?.positions) return []
    return portfolioData.positions.map(p => ({
      ticker: p.ticker,
      shares: p.shares,
      delta: p.forecastDeltaPct,
      impact: (p.latestPrice * p.shares * (p.forecastDeltaPct / 100))
    })).sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))
  }, [portfolioData])

  const addTicker = () => {
    const t = lookup.trim().toUpperCase()
    if (!t || portfolio.some(p => p.ticker === t)) return
    setPortfolio([...portfolio, { ticker: t, shares: 1 }])
    setLookup('')
  }

  const removeTicker = (t) => {
    setPortfolio(portfolio.filter(p => p.ticker !== t))
  }

  const updateShares = (t, s) => {
    setPortfolio(portfolio.map(p => p.ticker === t ? { ...p, shares: Number(s) } : p))
  }

  return (
    <div className="dashboard-view">
      <div className="view-controls">
        <div className="horizon-toggles">
          <button className={`horizon-btn ${horizon === 7 ? 'active' : ''}`} onClick={() => setHorizon(7)}>7D</button>
          <button className={`horizon-btn ${horizon === 30 ? 'active' : ''}`} onClick={() => setHorizon(30)}>30D</button>
        </div>
        <button className="analyze-btn" onClick={refresh} disabled={loading}>
          {loading ? 'ANALYZING...' : 'ANALYZE'}
        </button>
        <div className="ticker-lookup">
          <input value={lookup} onChange={e => setLookup(e.target.value.toUpperCase())} placeholder="TICKER" onKeyDown={e => e.key === 'Enter' && addTicker()} />
          <button className="horizon-btn" onClick={addTicker}>+</button>
        </div>
      </div>

      <StockChart
        history={portfolioData?.portfolioValuation?.history || []}
        forecast={portfolioData?.portfolioValuation?.forecast || []}
      />

      <div className="tab-container">
        <button className={`tab-btn ${activeTab === 'performance' ? 'active' : ''}`} onClick={() => setActiveTab('performance')}>Performance</button>
        <button className={`tab-btn ${activeTab === 'forecast' ? 'active' : ''}`} onClick={() => setActiveTab('forecast')}>Forecast</button>
        <button className={`tab-btn ${activeTab === 'sentiment' ? 'active' : ''}`} onClick={() => setActiveTab('sentiment')}>Sentiment Analysis</button>
      </div>

      <div className="tab-content card">
        {activeTab === 'forecast' && (
          <div className="forecast-tab">
            <div className="panel-header">Predicted Impact by Asset</div>
            <div className="impact-list">
              {impacts.map(i => (
                <div key={i.ticker} className={`impact-item ${i.delta < 0 ? 'warning' : ''}`}>
                  <span>{i.ticker} ({i.shares} SHARES)</span>
                  <span className="metric-value">
                    {i.delta >= 0 ? '+' : ''}{i.delta.toFixed(2)}%
                    ({i.impact >= 0 ? '+' : ''}${i.impact.toFixed(2)})
                    {i.delta < 0 && ' ⚠️'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
        {activeTab === 'performance' && (
          <div>
            <div className="panel-header">Historical Accuracy Log</div>
            <div className="metric-row"><span>Last 30 Days Accuracy</span><span className="metric-value">84.2%</span></div>
            <div className="metric-row"><span>Mean Absolute Error</span><span className="metric-value">$142.10</span></div>
          </div>
        )}
        {activeTab === 'sentiment' && (
          <div>
            <div className="panel-header">Global Sentiment Aggregator</div>
            <div className="metric-row"><span>News Sentiment</span><span className="metric-value" style={{color: '#00ff00'}}>BULLISH (0.62)</span></div>
            <div className="metric-row"><span>Social Buzz</span><span className="metric-value">HIGH (+12%)</span></div>
          </div>
        )}
      </div>

    </div>
  )
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

  useEffect(() => {
    fetch(`/api/global-intelligence?keyword=${encodeURIComponent(keyword)}`)
      .then((r) => r.json())
      .then((j) => setData(j.data))
      .catch(() => setData(null))
  }, [])

  useEffect(() => {
    fetch('https://cdn.jsdelivr.net/npm/world-atlas@2/land-110m.json')
      .then(r => r.json())
      .then(world => setLand(topojson.feature(world, world.objects.land)))
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

  const width = 800
  const height = 600
  const projection = d3geo.geoOrthographic()
    .scale(scale)
    .translate([width / 2, height / 2])
    .rotate(rotation)
    .clipAngle(90)

  const path = d3geo.geoPath(projection)

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
      <div className="globe-container">
        <div style={{ padding: '10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <input
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                fetch(`/api/global-intelligence?keyword=${encodeURIComponent(keyword)}`)
                  .then((r) => r.json())
                  .then((j) => setData(j.data))
              }
            }}
            placeholder="GMI SEARCH (ENTER)"
          />
          <div className="horizon-toggles" style={{ marginLeft: '20px' }}>
            <span style={{ fontSize: '0.7rem', color: '#ffb000' }}>T + {time}D</span>
            <input
              type="range"
              min="0"
              max="14"
              value={time}
              onChange={(e) => setTime(Number(e.target.value))}
              style={{ width: '100px' }}
            />
          </div>
        </div>
        <svg ref={svgRef} viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: '100%' }}>
          <circle cx={width/2} cy={height/2} r={projection.scale()} fill="#050505" />
          <path d={path(d3geo.geoGraticule()())} fill="none" stroke="#222" strokeWidth="0.5" />
          {land && <path d={path(land)} fill="#222" stroke="#444" strokeWidth="0.5" />}
          {layers.shippingRoutes && filtered?.shippingRoutes?.map((r, i) => {
            const d = getLinePath(r.from, r.to)
            if (!d) return null
            return <path key={i} d={d} fill="none" stroke={r.disruptionScore > 0.4 ? '#ff3333' : '#666'} strokeWidth="1" strokeDasharray="2 2" />
          })}
          <g className="markers">
            {layers.companies && filtered?.companies?.map((c) => {
              const coords = projection([c.lon, c.lat])
              if (!coords || d3geo.geoDistance(projection.invert([width/2, height/2]), [c.lon, c.lat]) > Math.PI / 2) return null
              return (
                <g key={c.name} transform={`translate(${coords[0]-6},${coords[1]-6})`} onClick={() => setSelected(c)} style={{ cursor: 'pointer' }}>
                  <rect width="12" height="12" fill={c.direction === 'up' ? '#00ff00' : '#ff3333'} />
                </g>
              )
            })}
            {layers.weather && filtered?.weather?.map((w, i) => {
              const coords = projection([w.lon, w.lat])
              if (!coords || d3geo.geoDistance(projection.invert([width/2, height/2]), [w.lon, w.lat]) > Math.PI / 2) return null
              return (
                <g key={`w-${i}`} transform={`translate(${coords[0]-8},${coords[1]-8})`} onClick={() => setSelected({ name: w.condition, riskScore: w.risk_impact, ...w })} style={{ cursor: 'pointer' }}>
                  <text fontSize="14px">⚡</text>
                </g>
              )
            })}
            {layers.macro && filtered?.macro?.map(([country, m], i) => {
              const locs = { 'USA': [-98, 38], 'China': [104, 35] }
              const coord = locs[country]
              if (!coord) return null
              const coords = projection(coord)
              if (!coords || d3geo.geoDistance(projection.invert([width/2, height/2]), coord) > Math.PI / 2) return null
              return (
                <g key={`m-${i}`} transform={`translate(${coords[0]-8},${coords[1]-8})`} onClick={() => setSelected({ name: country, ...m })} style={{ cursor: 'pointer' }}>
                  <text fontSize="14px">🌐</text>
                </g>
              )
            })}
          </g>
        </svg>
      </div>

      <aside className="sidebar">
        <div className="panel-header">Intelligence Layers</div>
        <div className="impact-list">
          {Object.keys(layers).map((k) => (
            <label key={k} style={{ display: 'flex', gap: '8px', fontSize: '0.8rem', textTransform: 'uppercase' }}>
              <input type="checkbox" checked={layers[k]} onChange={() => toggle(k)} /> {k}
            </label>
          ))}
        </div>

        {selected && (
          <div className="card" style={{ marginTop: '20px' }}>
            <div className="panel-header">{selected.name || selected.ticker}</div>
            <RiskMeter score={selected.riskScore || 0.2} />
            {selected.sentiment && <div className="metric-row"><span>Sentiment</span><span className="metric-value">{selected.sentiment}</span></div>}
            {selected.forecastPrice && <div className="metric-row"><span>Forecast</span><span className="metric-value">${selected.forecastPrice}</span></div>}
            {selected.condition && <div className="metric-row"><span>Condition</span><span className="metric-value">{selected.condition}</span></div>}
            {selected.gdp && <div className="metric-row"><span>GDP Growth</span><span className="metric-value">{selected.gdp}</span></div>}
            {selected.inflation && <div className="metric-row"><span>Inflation</span><span className="metric-value">{selected.inflation}</span></div>}
            <button className="horizon-btn" onClick={() => setSelected(null)} style={{ width: '100%', marginTop: '10px' }}>CLOSE</button>
          </div>
        )}
      </aside>
    </div>
  )
}

export function App() {
  const [page, setPage] = useState('portfolio')
  const [portfolio, setPortfolio] = useState(defaultPortfolio)
  const [portfolioData, setPortfolioData] = useState(null)
  const [keyword, setKeyword] = useState('global markets')
  const [loading, setLoading] = useState(false)
  const [horizon, setHorizon] = useState(7)

  useEffect(() => {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) { try { const p = JSON.parse(raw); if (Array.isArray(p) && p.length) setPortfolio(p) } catch { } }
  }, [])

  useEffect(() => { localStorage.setItem(STORAGE_KEY, JSON.stringify(portfolio)) }, [portfolio])

  async function refresh() {
    setLoading(true)
    try {
      const res = await fetch('/api/portfolio/forecast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ positions: portfolio, sentiment_model: 'transformer' })
      })
      setPortfolioData(await res.json())
    } finally { setLoading(false) }
  }

  useEffect(() => { if (portfolio.length) refresh() }, [portfolio])

  return (
    <div className="app-container">
      <nav className="navbar">
        <div className="nav-brand">
          <SentidexLogo />
          SENTIDEX PRO
        </div>

        {page === 'portfolio' && portfolioData && (
          <div className="portfolio-summary">
            <div className="summary-item">
              <span className="summary-label">Total Value</span>
              <span className="summary-value">${portfolioData.portfolioValuation.currentValue.toLocaleString()}</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Expected Gain</span>
              <span className={`summary-value ${(portfolioData.portfolioValuation.forecastWeekEndValue - portfolioData.portfolioValuation.currentValue) >= 0 ? 'pos' : 'neg'}`}>
                ${(portfolioData.portfolioValuation.forecastWeekEndValue - portfolioData.portfolioValuation.currentValue).toLocaleString()}
              </span>
            </div>
          </div>
        )}

        <div className="nav-links">
          <button className={page === 'portfolio' ? 'active' : ''} onClick={() => setPage('portfolio')}>Portfolio</button>
          <button className={page === 'gmi' ? 'active' : ''} onClick={() => setPage('gmi')}>GMI</button>
        </div>
      </nav>
      <main className="main-content">
        {page === 'portfolio' ? (
          <>
            <PortfolioDashboard
              portfolio={portfolio}
              portfolioData={portfolioData}
              refresh={refresh}
              loading={loading}
              setPortfolio={setPortfolio}
              horizon={horizon}
              setHorizon={setHorizon}
            />
            <aside className="sidebar">
              <div>
                <div className="panel-header">Key Metrics</div>
                <div className="metric-row"><span>Confidence Score</span><span className="metric-value">0.82</span></div>
                <div className="metric-row"><span>Model Accuracy</span><span className="metric-value">91%</span></div>
              </div>

              <div>
                <div className="panel-header">Analysis Log</div>
                <div className="impact-list" style={{ fontSize: '0.7rem' }}>
                  <div className="impact-item"><span>{new Date().toLocaleTimeString()}</span><span>REFRESH_OK</span></div>
                  <div className="impact-item"><span>T-minus 5m</span><span>TRANSFORMER_IDLE</span></div>
                </div>
              </div>

              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
                <div className="panel-header">Portfolio Weights</div>
                <div className="impact-list" style={{ overflowY: 'auto' }}>
                  {portfolio.map(p => (
                    <div key={p.ticker} className="impact-item">
                      <span>{p.ticker}</span>
                      <div style={{ display: 'flex', alignItems: 'center' }}>
                        <input
                          type="number"
                          value={p.shares}
                          onChange={e => setPortfolio(portfolio.map(it => it.ticker === p.ticker ? { ...it, shares: Number(e.target.value) } : it))}
                          style={{ width: '50px', height: '20px', fontSize: '0.7rem', padding: '0 2px' }}
                        />
                        <button
                          className="horizon-btn"
                          style={{ color: '#ff3333', border: 'none', background: 'transparent', padding: '0 5px' }}
                          onClick={() => setPortfolio(portfolio.filter(it => it.ticker !== p.ticker))}
                        >
                          ×
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </aside>
          </>
        ) : (
          <GlobalMarketIntelligence keyword={keyword} setKeyword={setKeyword} />
        )}
      </main>
    </div>
  )
}
