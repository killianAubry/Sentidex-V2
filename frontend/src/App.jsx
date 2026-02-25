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

if (typeof Highcharts === 'object') {
  import('highcharts/highcharts-more').then(m => {
    const fn = m.default || m
    if (typeof fn === 'function') fn(Highcharts)
  })
}

const STORAGE_KEY = 'sentidex-portfolio-v3'
const defaultPortfolio = [{ ticker: 'AAPL', shares: 4 }, { ticker: 'MSFT', shares: 2 }, { ticker: 'NVDA', shares: 1 }]

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

function StockChart({ title, history, forecast, valueKey = 'value', separateLines = [], showConfidence = false }) {
  const options = useMemo(() => {
    const histData = history.map(d => [new Date(d.date).getTime(), d[valueKey] || d.close])
    const foreData = forecast.map(d => [new Date(d.date).getTime(), d.predictedClose || d.value])
    const rangeData = forecast.map(d => {
      const val = d.predictedClose || d.value
      return [new Date(d.date).getTime(), val * 0.95, val * 1.05]
    })
    const series = [
      { name: 'History', data: histData, color: '#3b82f6', zIndex: 2 },
      { name: 'Aggregate Forecast', data: foreData, color: '#60a5fa', dashStyle: 'ShortDash', zIndex: 3 }
    ]
    if (showConfidence) {
      series.push({
        name: 'Confidence Range', data: rangeData, type: 'arearange',
        lineWidth: 0, linkedTo: ':previous', color: '#3b82f6',
        fillOpacity: 0.1, zIndex: 1, marker: { enabled: false }
      })
    }
    separateLines.forEach((line, i) => {
      series.push({
        name: `Source: ${line.name}`,
        data: line.data.map(d => [new Date(d.date).getTime(), d.value]),
        color: Highcharts.getOptions().colors[i + 2],
        dashStyle: 'Dot', zIndex: 2, opacity: 0.7
      })
    })
    return {
      chart: { backgroundColor: '#1e293b', style: { fontFamily: 'Inter, sans-serif' }, height: 400 },
      title: { text: null },
      xAxis: { type: 'datetime', gridLineColor: '#334155', labels: { style: { color: '#94a3b8' } } },
      yAxis: { title: { text: null }, gridLineColor: '#334155', labels: { style: { color: '#94a3b8' } } },
      legend: { itemStyle: { color: '#94a3b8' } },
      tooltip: { shared: true },
      series
    }
  }, [history, forecast, valueKey, separateLines, showConfidence])

  if (!history.length && !forecast.length) return <div className="no-data">No chart data available.</div>
  return <div className="chart-container"><HighchartsReact highcharts={Highcharts} options={options} /></div>
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
  portfolio, portfolioData, refresh, loading, lookup, setLookup,
  addTicker, selectedSources, toggleSource, sourceOptions,
  forecastBySelectedSources, viewStates, toggleViewState
}) {
  const sortedPositions = useMemo(() => {
    return [...(portfolioData?.positions || [])].sort((a, b) => b.forecastDeltaPct - a.forecastDeltaPct)
  }, [portfolioData])

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
            <input value={lookup} onChange={(e) => setLookup(e.target.value.toUpperCase())} placeholder="Enter Ticker (e.g. AAPL)" />
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
                        <input type="checkbox" checked={selectedSources.includes(source)} onChange={() => toggleSource(source)} />
                        {source}
                      </label>
                    ))}
                  </div>
                </div>
                <div className="control-group">
                  <h5>Views & Analysis</h5>
                  <div className="view-buttons">
                    <button className={`small-btn ${viewStates.showConfidence ? 'active' : ''}`} onClick={() => toggleViewState('showConfidence')}>Show Confidence Levels</button>
                    <button className={`small-btn ${viewStates.separateForecasts ? 'active' : ''}`} onClick={() => toggleViewState('separateForecasts')}>Separate Forecasts</button>
                    <button className={`small-btn ${viewStates.macroImpact ? 'active' : ''}`} onClick={() => toggleViewState('macroImpact')}>Macro Impact</button>
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
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState(null)
  const [layers, setLayers] = useState({ nodes: true, routes: true, weather: true, risk: true })
  const svgRef = useRef(null)
  const [rotation, setRotation] = useState([0, -30])
  const [scale, setScale] = useState(250)
  const [land, setLand] = useState(null)
  const [inputVal, setInputVal] = useState(keyword)
  const [nodeCount, setNodeCount] = useState(12)  // user-configurable node count

  useEffect(() => {
    fetch('https://cdn.jsdelivr.net/npm/world-atlas@2/land-110m.json')
      .then(r => r.json())
      .then(world => setLand(topojson.feature(world, world.objects.land)))
  }, [])

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.call(d3.drag().on('drag', (e) =>
      setRotation(([r0, r1]) => [r0 + e.dx / 2, r1 - e.dy / 2])
    ))
    svg.call(d3.zoom().scaleExtent([0.3, 20]).on('zoom', (e) =>
      setScale(250 * e.transform.k)
    ))
  }, [])

  async function runSearch(q) {
    setLoading(true)
    setSelected(null)
    setData(null)
    try {
      const r = await fetch(`/api/globe-supply-chain?query=${encodeURIComponent(q)}&node_count=${nodeCount}`)
      const j = await r.json()
      setData(j)
      setKeyword(q)
      if (j.nodes?.length) setRotation([-j.nodes[0].lon, -j.nodes[0].lat + 20])
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
          {/* Node count control */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#94a3b8', fontSize: '0.8rem', whiteSpace: 'nowrap' }}>
            Nodes:
            <input
              type="number"
              min={3}
              max={20}
              value={nodeCount}
              onChange={e => setNodeCount(Math.min(20, Math.max(3, parseInt(e.target.value) || 12)))}
              style={{ width: '3.5rem', padding: '0.3rem', background: '#1e293b', border: '1px solid #334155', borderRadius: '0.4rem', color: 'white', textAlign: 'center' }}
            />
          </div>
          <button onClick={() => runSearch(inputVal)} disabled={loading} className="refresh-btn"
            style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            {loading
              ? <><RefreshCw01 width={14} height={14} /> Analyzing...</>
              : <><SearchLg width={14} height={14} /> Analyze Supply Chain</>
            }
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
              <Globe01 width={48} height={48} color="#60a5fa" />
              <div style={{ color: '#60a5fa', fontSize: '1rem' }}>AI is mapping the supply chain...</div>
              <div style={{ color: '#94a3b8', fontSize: '0.8rem' }}>Querying Groq → Geocoding {nodeCount} nodes → Risk Analysis</div>
            </div>
          )}

          <svg ref={svgRef} viewBox={`0 0 ${width} ${height}`} className="globe-3d">
            {/* Flat dark ocean — no gradient, no shading */}
            <circle cx={width / 2} cy={height / 2} r={scale} fill="#0a0f1e" />

            {/* Graticule — subtle grid lines */}
            <path d={path(d3geo.geoGraticule()())} fill="none" stroke="#1e293b" strokeWidth="0.4" opacity="0.6" />

            {/* Land — white outline only, flat fill */}
            {land && <path d={path(land)} fill="#111827" stroke="rgba(255,255,255,0.35)" strokeWidth="0.6" />}

            {/* Routes */}
            {layers.routes && data?.routes?.map((r, i) => {
              const d = getRoutePath(r.from, r.to)
              if (!d) return null
              const color = r.risk > 0.6 ? '#ef4444' : r.risk > 0.3 ? '#f59e0b' : '#22c55e'
              return (
                <path key={`route-${i}`} d={d} fill="none"
                  stroke={color} strokeWidth="1.5"
                  strokeDasharray="5 4" opacity="0.8"
                />
              )
            })}

            {/* Nodes — simple dots only, no icons */}
            {layers.nodes && data?.nodes?.map((node, i) => {
              if (!isVisible(node.lon, node.lat)) return null
              const coords = projection([node.lon, node.lat])
              if (!coords) return null
              const riskColor = RISK_COLORS[node.risk?.level] || '#94a3b8'

              return (
                <g key={`node-${i}`}
                  transform={`translate(${coords[0]},${coords[1]})`}
                  onClick={() => setSelected(node)}
                  style={{ cursor: 'pointer' }}>

                  {/* Outer risk ring */}
                  {layers.risk && (
                    <circle r="16" fill="none" stroke={riskColor} strokeWidth="1" opacity="0.3" />
                  )}

                  {/* Main dot */}
                  <GlobeDot riskColor={riskColor} isSelected={selected?.name === node.name} />

                  {/* Risk score badge */}
                  {layers.risk && node.risk?.score > 0.4 && (
                    <g transform="translate(10,-10)">
                      <circle r="6" fill={riskColor} />
                      <text textAnchor="middle" dominantBaseline="central" fontSize="7" fill="white" fontWeight="bold">
                        {Math.round(node.risk.score * 10)}
                      </text>
                    </g>
                  )}

                  {/* City label */}
                  <text y="22" textAnchor="middle" fontSize="8" fill="#94a3b8" style={{ pointerEvents: 'none' }}>
                    {node.city || node.name}
                  </text>
                </g>
              )
            })}
          </svg>

          {/* Layer toggles */}
          <div style={{ position: 'absolute', bottom: '1rem', left: '1rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
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
          {selected && (
            <div className="detail-panel card" style={{ marginBottom: '1rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <h4 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <NodeIcon type={selected.type} size={18} color={RISK_COLORS[selected.risk?.level] || '#60a5fa'} />
                    {selected.name}
                  </h4>
                  <div style={{ color: '#60a5fa', fontSize: '0.8rem' }}>{selected.type} · {selected.city}, {selected.country}</div>
                </div>
                <button className="close-btn" onClick={() => setSelected(null)}><XClose width={14} height={14} /></button>
              </div>

              <p style={{ color: '#94a3b8', fontSize: '0.85rem', margin: '0.5rem 0' }}>{selected.role}</p>

              <div style={{ marginBottom: '0.75rem' }}>
                <div style={{ color: RISK_COLORS[selected.risk?.level], fontWeight: 'bold', fontSize: '0.85rem', marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                  <AlertTriangle width={14} height={14} />
                  Risk: {selected.risk?.level?.toUpperCase()} ({Math.round((selected.risk?.score || 0) * 100)}%)
                </div>
                <RiskMeter score={selected.risk?.score || 0} />
                <p style={{ color: '#94a3b8', fontSize: '0.8rem', marginTop: '0.4rem' }}>{selected.risk?.summary}</p>
              </div>

              {selected.weather && (
                <div style={{ background: '#0f172a', borderRadius: '0.5rem', padding: '0.5rem', marginBottom: '0.75rem' }}>
                  <div style={{ color: '#94a3b8', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
                    <WeatherIcon icon={selected.weather.icon} size={14} />
                    {selected.weather.condition}
                    {selected.weather.temp_c != null && <span style={{ display: 'flex', alignItems: 'center', gap: '0.2rem' }}>· <Thermometer01 width={12} height={12} /> {selected.weather.temp_c}°C</span>}
                    {selected.weather.wind_kph != null && <span style={{ display: 'flex', alignItems: 'center', gap: '0.2rem' }}>· <Wind01 width={12} height={12} /> {selected.weather.wind_kph} km/h</span>}
                  </div>
                </div>
              )}

              {selected.risk?.headlines?.length > 0 && (
                <div>
                  <div style={{ color: '#64748b', fontSize: '0.75rem', marginBottom: '0.4rem' }}>RECENT NEWS</div>
                  {selected.risk.headlines.slice(0, 3).map((h, i) => (
                    <div key={i} style={{ fontSize: '0.75rem', color: '#94a3b8', padding: '0.3rem 0', borderBottom: '1px solid #1e293b' }}>• {h}</div>
                  ))}
                </div>
              )}

              {selected.connections?.length > 0 && (
                <div style={{ marginTop: '0.75rem' }}>
                  <div style={{ color: '#64748b', fontSize: '0.75rem', marginBottom: '0.3rem' }}>CONNECTS TO</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem' }}>
                    {selected.connections.map(c => (
                      <span key={c} style={{ fontSize: '0.7rem', padding: '0.15rem 0.4rem', background: '#1e293b', borderRadius: '999px', color: '#60a5fa' }}>{c}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {data?.nodes && (
            <div className="card" style={{ padding: '1rem' }}>
              <h4 style={{ margin: '0 0 0.75rem' }}>Supply Chain Nodes ({data.nodes.length})</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', maxHeight: '350px', overflowY: 'auto' }}>
                {[...data.nodes]
                  .sort((a, b) => (b.risk?.score || 0) - (a.risk?.score || 0))
                  .map((node, i) => (
                    <div key={i}
                      onClick={() => { setSelected(node); setRotation([-node.lon, -node.lat + 20]) }}
                      style={{
                        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                        padding: '0.4rem 0.6rem', background: '#0f172a', borderRadius: '0.4rem',
                        cursor: 'pointer', border: `1px solid ${RISK_COLORS[node.risk?.level] || '#334155'}22`
                      }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <NodeIcon type={node.type} size={14} color={RISK_COLORS[node.risk?.level] || '#94a3b8'} />
                        <span style={{ fontSize: '0.8rem', color: '#e2e8f0' }}>{node.name}</span>
                        <span style={{ fontSize: '0.7rem', color: '#64748b' }}>{node.city}</span>
                      </div>
                      <span style={{
                        fontSize: '0.7rem', padding: '0.1rem 0.4rem', borderRadius: '999px',
                        background: (RISK_COLORS[node.risk?.level] || '#94a3b8') + '33',
                        color: RISK_COLORS[node.risk?.level] || '#94a3b8'
                      }}>{node.risk?.level}</span>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {!data && !loading && (
            <div className="card" style={{ padding: '2rem', textAlign: 'center', color: '#64748b' }}>
              <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1rem' }}>
                <Globe01 width={48} height={48} color="#334155" />
              </div>
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
  const [viewStates, setViewStates] = useState({ showConfidence: false, separateForecasts: false, macroImpact: false })

  function toggleViewState(key) { setViewStates(prev => ({ ...prev, [key]: !prev[key] })) }

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
        body: JSON.stringify({ positions: portfolio, sentiment_model: 'transformer', keyword })
      })
      setPortfolioData(await res.json())
    } finally { setLoading(false) }
  }

  useEffect(() => { if (portfolio.length) refresh() }, [portfolio])

  const sourceOptions = useMemo(() => {
    const found = new Set()
    for (const p of portfolioData?.positions || []) Object.keys(p.sourceForecasts || {}).forEach(k => found.add(k))
    return Array.from(found).sort()
  }, [portfolioData])

  const forecastBySelectedSources = useMemo(() =>
    aggregateForecastFromSources(portfolioData?.positions || [], selectedSources),
    [portfolioData, selectedSources])

  function toggleSource(source) {
    setSelectedSources(p => p.includes(source) ? p.filter(x => x !== source) : [...p, source])
  }

  function addTicker() {
    const ticker = lookup.trim().toUpperCase()
    if (!ticker || portfolio.some(p => p.ticker === ticker)) return
    setPortfolio(p => [...p, { ticker, shares: 1 }])
  }

  return (
    <div className="app-container">
      <nav className="navbar">
        <div className="nav-brand">SENTIDEX <span className="pro-tag">PRO</span></div>
        <div className="nav-links">
          <button className={page === 'portfolio' ? 'active' : ''} onClick={() => setPage('portfolio')}>Portfolio</button>
          <button className={page === 'gmi' ? 'active' : ''} onClick={() => setPage('gmi')}>GMI</button>
        </div>
      </nav>
      <main className="main-content">
        {page === 'portfolio' ? (
          <PortfolioDashboard
            portfolio={portfolio} portfolioData={portfolioData} refresh={refresh} loading={loading}
            lookup={lookup} setLookup={setLookup} addTicker={addTicker}
            selectedSources={selectedSources} toggleSource={toggleSource} sourceOptions={sourceOptions}
            forecastBySelectedSources={forecastBySelectedSources} viewStates={viewStates} toggleViewState={toggleViewState}
          />
        ) : (
          <GlobalMarketIntelligence keyword={keyword} setKeyword={setKeyword} />
        )}
      </main>
    </div>
  )
}
