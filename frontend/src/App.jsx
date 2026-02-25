import { useEffect, useMemo, useState, useRef, useCallback } from 'react'
import * as d3 from 'd3'
import * as d3geo from 'd3-geo'
import * as topojson from 'topojson-client'
import Highcharts from 'highcharts'
import HighchartsMore from 'highcharts/highcharts-more'
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
  SearchLg,
  RefreshCw01,
  XClose,
  Wind01,
  TrendUp01,
} from '@untitled-ui/icons-react'

const _more = HighchartsMore.default ?? HighchartsMore
if (typeof _more === 'function') _more(Highcharts)

const STORAGE_KEY = 'sentidex-portfolio-v3'
const GMI_CACHE_KEY = 'sentidex-gmi-cache-v1'
const defaultPortfolio = [{ ticker: 'AAPL', shares: 4 }, { ticker: 'MSFT', shares: 2 }, { ticker: 'NVDA', shares: 1 }]

// ---------------------------------------------------------------------------
// GMI Cache helpers (12-hour TTL)
// ---------------------------------------------------------------------------
function gmiCacheGet(query, nodeCount) {
  try {
    const raw = localStorage.getItem(GMI_CACHE_KEY)
    if (!raw) return null
    const cache = JSON.parse(raw)
    const key = `${query.toLowerCase()}__${nodeCount}`
    const entry = cache[key]
    if (!entry) return null
    const age = Date.now() - entry.ts
    if (age > 12 * 60 * 60 * 1000) return null
    return entry.data
  } catch { return null }
}

function gmiCacheSet(query, nodeCount, data) {
  try {
    const raw = localStorage.getItem(GMI_CACHE_KEY)
    const cache = raw ? JSON.parse(raw) : {}
    const key = `${query.toLowerCase()}__${nodeCount}`
    cache[key] = { ts: Date.now(), data }
    for (const k of Object.keys(cache)) {
      if (Date.now() - cache[k].ts > 12 * 60 * 60 * 1000) delete cache[k]
    }
    localStorage.setItem(GMI_CACHE_KEY, JSON.stringify(cache))
  } catch { }
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------
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
  '☀️': Sun, '🌧️': CloudRaining01, '❄️': Snowflake01,
  '🌩️': Lightning01, '⛅': Cloud01, '🌫️': Wind01, '🌐': Globe01,
}

function NodeIcon({ type, size = 16, color = 'currentColor', ...props }) {
  const Icon = NODE_ICON_COMPONENTS[type] || NODE_ICON_COMPONENTS.default
  return <Icon width={size} height={size} color={color} {...props} />
}

function WeatherIcon({ icon, size = 14, color = '#60a5fa', ...props }) {
  const Icon = WEATHER_ICON_COMPONENTS[icon] || Globe01
  return <Icon width={size} height={size} color={color} {...props} />
}

const RISK_COLORS = { low: '#22c55e', medium: '#f59e0b', high: '#ef4444', critical: '#7c3aed' }

function GlobeDot({ riskColor, isSelected, dotScale = 1 }) {
  const r1 = 10 * dotScale
  const r2 = 4 * dotScale
  const r3 = 12 * dotScale
  return (
    <g>
      <circle r={r1} fill="#0f172a" stroke={riskColor} strokeWidth={2 * dotScale} />
      <circle r={r2} fill={riskColor} />
      {isSelected && <circle r={r3} fill="none" stroke={riskColor} strokeWidth={dotScale} opacity="0.5" />}
    </g>
  )
}

// ---------------------------------------------------------------------------
// Global Market Intelligence
// ---------------------------------------------------------------------------
function GlobalMarketIntelligence({ data, loading, rotation, setRotation, selected, setSelected, scale, setScale, zoomRef, svgRef }) {
  const [layers, setLayers] = useState({ nodes: true, routes: true, weather: true, risk: true })
  const [land, setLand] = useState(null)

  useEffect(() => {
    let timer
    if (loading) {
      timer = d3.timer(() => {
        setRotation(([r0, r1]) => [r0 + 0.3, r1])
      })
    }
    return () => timer?.stop()
  }, [loading, setRotation])

  useEffect(() => {
    fetch('https://cdn.jsdelivr.net/npm/world-atlas@2/land-110m.json')
      .then(r => r.json())
      .then(world => setLand(topojson.feature(world, world.objects.land)))
  }, [])

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.call(d3.drag().on('drag', (e) =>
      setRotation(([r0, r1]) => [r0 + e.dx * 0.2, r1 - e.dy * 0.2])
    ))
    const zoom = d3.zoom().scaleExtent([0.8, 8]).on('zoom', (e) =>
      setScale(250 * e.transform.k)
    )
    zoomRef.current = zoom
    svg.call(zoom)
  }, [])

  // Auto-center on selected node
  useEffect(() => {
    if (!selected) return
    const target = [-selected.lon, -selected.lat + 20]
    const i = d3.interpolate(rotation, target)
    const timer = d3.timer((t) => {
      if (t >= 700) {
        setRotation(target)
        timer.stop()
      } else {
        setRotation(i(t / 700))
      }
    })
    return () => timer.stop()
  }, [selected])

  const handleResetView = () => {
    setRotation([0, -30])
    setScale(250)
    if (zoomRef.current && svgRef.current) {
      d3.select(svgRef.current).transition().duration(700).call(zoomRef.current.transform, d3.zoomIdentity)
    }
  }

  const width = 800, height = 580
  const projection = useMemo(() =>
    d3geo.geoOrthographic()
      .scale(scale).translate([width / 2, height / 2])
      .rotate(rotation).clipAngle(90),
    [scale, rotation]
  )
  const path = useMemo(() => d3geo.geoPath(projection), [projection])

  // Shrink dots as zoom increases so they don't overlap; clamp between 0.4 and 1.0
  const dotScale = useMemo(() => Math.min(1.0, Math.max(0.4, 250 / scale)), [scale])

  function isVisible(lon, lat) {
    try {
      const center = projection.invert([width / 2, height / 2])
      return d3geo.geoDistance(center, [lon, lat]) < Math.PI / 2
    } catch { return false }
  }

  return (
    <div className="gmi-page">
      <div className="gmi-main-section">
        <div className="globe-container floating">
          {loading && (
            <div className="loading-overlay">
              <Globe01 width={64} height={64} color="#60a5fa" className="animate-spin-slow" />
              <div className="loading-text">AI Analysis in Progress...</div>
            </div>
          )}

          <svg ref={svgRef} viewBox={`0 0 ${width} ${height}`} className="globe-3d">
            <circle cx={width / 2} cy={height / 2} r={scale} fill="#0a0f1e" />
            <path d={path(d3geo.geoGraticule()())} fill="none" stroke="#1e293b" strokeWidth="0.4" opacity="0.6" />
            {land && <path d={path(land)} fill="#111827" stroke="rgba(255,255,255,0.35)" strokeWidth="0.6" />}

            {layers.routes && data?.routes?.map((r, i) => {
              const d = path({ type: 'LineString', coordinates: [[r.from.lon, r.from.lat], [r.to.lon, r.to.lat]] })
              if (!d) return null
              const color = r.risk > 0.6 ? '#ef4444' : r.risk > 0.3 ? '#f59e0b' : '#22c55e'
              return <path key={`route-${i}`} d={d} fill="none" stroke={color} strokeWidth="1.5" strokeDasharray="5 4" opacity="0.8" />
            })}

            {layers.nodes && data?.nodes?.map((node, i) => {
              if (!isVisible(node.lon, node.lat)) return null
              const coords = projection([node.lon, node.lat])
              if (!coords) return null
              const riskColor = RISK_COLORS[node.risk?.level] || '#94a3b8'
              return (
                <g key={`node-${i}`} transform={`translate(${coords[0]},${coords[1]})`}
                  onClick={() => setSelected(node)} style={{ cursor: 'pointer' }}>
                  {layers.risk && <circle r={16 * dotScale} fill="none" stroke={riskColor} strokeWidth="1" opacity="0.3" />}
                  <GlobeDot riskColor={riskColor} isSelected={selected?.name === node.name} dotScale={dotScale} />
                  <text y={22 * dotScale} textAnchor="middle" fontSize={8 * dotScale} fill="#94a3b8" style={{ pointerEvents: 'none' }}>
                    {node.city || node.name}
                  </text>
                </g>
              )
            })}
          </svg>

          {selected && isVisible(selected.lon, selected.lat) && (
            <div className="node-callout" style={{
              position: 'absolute',
              left: projection([selected.lon, selected.lat])[0] + 15,
              top: projection([selected.lon, selected.lat])[1] - 15,
              transform: 'translate(0, -100%)',
              zIndex: 20
            }}>
              <div className="callout-content card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem' }}>
                  <div>
                    <h4 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <NodeIcon type={selected.type} size={16} color={RISK_COLORS[selected.risk?.level]} />
                      {selected.name}
                    </h4>
                    <div style={{ color: '#60a5fa', fontSize: '0.75rem' }}>{selected.city}, {selected.country}</div>
                  </div>
                  <button className="close-btn-small" onClick={() => setSelected(null)}><XClose width={12} height={12} /></button>
                </div>
                <p style={{ fontSize: '0.75rem', color: '#94a3b8', margin: '0.5rem 0' }}>{selected.role}</p>
                <RiskMeter score={selected.risk?.score} />
                {selected.weather && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', marginTop: '0.5rem', fontSize: '0.7rem', color: '#64748b' }}>
                    <WeatherIcon icon={selected.weather.icon} size={12} />
                    {selected.weather.condition} · {selected.weather.temp_c}°C
                  </div>
                )}
              </div>
            </div>
          )}

          <div style={{ position: 'absolute', bottom: '1.5rem', left: '1.5rem', display: 'flex', gap: '0.5rem' }}>
            <button onClick={handleResetView} className="globe-btn">Reset</button>
            {Object.keys(layers).map(k => (
              <button key={k} onClick={() => setLayers(p => ({ ...p, [k]: !p[k] }))}
                className={`globe-btn ${layers[k] ? 'active' : ''}`}>
                {k.charAt(0).toUpperCase() + k.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {data?.nodes && (
        <div className="breadcrumbs-container">
          <div className="breadcrumbs-flow">
            {data.nodes.map((node, i) => (
              <div key={i} className="breadcrumb-wrapper">
                <button 
                  className={`breadcrumb-node ${selected?.name === node.name ? 'active' : ''}`}
                  onClick={() => setSelected(node)}
                >
                  <NodeIcon type={node.type} size={12} color={RISK_COLORS[node.risk?.level] || '#94a3b8'} />
                  <span>{node.name}</span>
                </button>
                {i < data.nodes.length - 1 && (
                  <div className="breadcrumb-arrow"><TrendUp01 width={12} height={12} style={{ transform: 'rotate(90deg)' }} /></div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// App root
// ---------------------------------------------------------------------------
export function App() {
  const [portfolio, setPortfolio] = useState(defaultPortfolio)
  const [portfolioData, setPortfolioData] = useState(null)
  const [keyword, setKeyword] = useState('Apple')
  const [gmiData, setGmiData] = useState(null)
  const [gmiLoading, setGmiLoading] = useState(false)
  const [inputVal, setInputVal] = useState('Apple')
  const [nodeCount, setNodeCount] = useState(12)
  const [cacheInfo, setCacheInfo] = useState(null)
  const [selected, setSelected] = useState(null)
  const [rotation, setRotation] = useState([0, -30])
  const [scale, setScale] = useState(250)
  const svgRef = useRef(null)
  const zoomRef = useRef(null)

  const stats = useMemo(() => {
    const issues = (gmiData?.nodes || []).filter(n => (n.risk?.score || 0) > 0.4).length
    const pos = (portfolioData?.positions || []).find(p => 
      p.ticker.toLowerCase() === keyword.toLowerCase() || 
      (gmiData?.query || "").toLowerCase().includes(p.ticker.toLowerCase())
    )
    const totalVal = portfolioData?.portfolioValuation?.currentValue || 0
    const exposure = (pos && totalVal > 0) ? (pos.shares * pos.latestPrice / totalVal * 100).toFixed(1) : "0.0"
    const avgRisk = (gmiData?.nodes || []).reduce((acc, n) => acc + (n.risk?.score || 0), 0) / (gmiData?.nodes?.length || 1)
    const atRisk = pos ? (pos.shares * pos.latestPrice * avgRisk).toLocaleString(undefined, { style: 'currency', currency: 'USD' }) : "$0.00"
    return { issues, exposure, atRisk }
  }, [gmiData, portfolioData, keyword])

  async function refreshPortfolio() {
    try {
      const res = await fetch('/api/portfolio/forecast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ positions: portfolio, sentiment_model: 'transformer', keyword })
      })
      setPortfolioData(await res.json())
    } catch (e) {}
  }

  async function runSearch(q) {
    const cached = gmiCacheGet(q, nodeCount)
    if (cached) {
      setGmiData(cached)
      setKeyword(q)
      setCacheInfo('from cache')
      if (cached.nodes?.length) setSelected(cached.nodes[0])
      return
    }

    setGmiLoading(true)
    setSelected(null)
    setGmiData(null)
    setCacheInfo(null)
    try {
      const r = await fetch(`/api/globe-supply-chain?query=${encodeURIComponent(q)}&node_count=${nodeCount}`)
      const j = await r.json()
      setGmiData(j)
      setKeyword(q)
      gmiCacheSet(q, nodeCount, j)
      setCacheInfo('fresh')
      if (j.nodes?.length) setSelected(j.nodes[0])
    } catch (e) {
      console.error(e)
    } finally {
      setGmiLoading(false)
    }
  }

  useEffect(() => {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) { try { const p = JSON.parse(raw); if (Array.isArray(p) && p.length) setPortfolio(p) } catch { } }
    runSearch('Apple') // Initial search
  }, [])

  useEffect(() => { if (portfolio.length) refreshPortfolio() }, [portfolio])

  return (
    <div className="app-container">
      <nav className="navbar">
        <div className="nav-brand">SENTIDEX <span className="pro-tag">PRO</span></div>
        <div className="navbar-center">
          <div className="search-bar-container">
            <input
              value={inputVal}
              onChange={e => setInputVal(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') runSearch(inputVal) }}
              placeholder="Enter company, commodity or sector..."
              className="nav-search-input"
            />
            <div className="node-spec">
              <span>Nodes:</span>
              <input type="number" min={3} max={20} value={nodeCount}
                onChange={e => setNodeCount(Math.min(20, Math.max(3, parseInt(e.target.value) || 12)))}
              />
            </div>
            <button onClick={() => runSearch(inputVal)} disabled={gmiLoading} className="nav-search-btn">
              {gmiLoading ? <RefreshCw01 className="animate-spin" width={16} height={16} /> : <SearchLg width={16} height={16} />}
            </button>
          </div>
        </div>
        <div className="nav-right">{cacheInfo && <span className="cache-info-tag">{cacheInfo}</span>}</div>
      </nav>

      <main className="main-content">
        <div className="stats-bar">
          <div className="stat-card">
            <span className="stat-label">NUMBER OF ISSUES #</span>
            <span className="stat-value">{stats.issues}</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">PORTFOLIO EXPOSURE %</span>
            <span className="stat-value">{stats.exposure}%</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">ASSETS AT RISK $</span>
            <span className="stat-value">{stats.atRisk}</span>
          </div>
        </div>

        <GlobalMarketIntelligence 
          data={gmiData} loading={gmiLoading}
          rotation={rotation} setRotation={setRotation}
          selected={selected} setSelected={setSelected}
          scale={scale} setScale={setScale}
          zoomRef={zoomRef} svgRef={svgRef}
        />
      </main>
    </div>
  )
}
