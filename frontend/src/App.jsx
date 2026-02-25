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
  AlertTriangle,
  BarChart01,
  SearchLg,
  RefreshCw01,
  XClose,
  Wind01,
  Thermometer01,
  TrendUp01,
  TrendDown01,
  FilterLines,
  Users01,
  Building02,
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
    if (age > 12 * 60 * 60 * 1000) return null   // expired
    return entry.data
  } catch { return null }
}

function gmiCacheSet(query, nodeCount, data) {
  try {
    const raw = localStorage.getItem(GMI_CACHE_KEY)
    const cache = raw ? JSON.parse(raw) : {}
    const key = `${query.toLowerCase()}__${nodeCount}`
    cache[key] = { ts: Date.now(), data }
    // Prune entries older than 12h to keep storage clean
    for (const k of Object.keys(cache)) {
      if (Date.now() - cache[k].ts > 12 * 60 * 60 * 1000) delete cache[k]
    }
    localStorage.setItem(GMI_CACHE_KEY, JSON.stringify(cache))
  } catch { }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// StockChart — with zoom + mouseWheel
// ---------------------------------------------------------------------------
function StockChart({ title, history, forecast, valueKey = 'value', separateLines = [], showConfidence = false }) {
  const options = useMemo(() => {
    const histData = history.map(d => [new Date(d.date).getTime(), d[valueKey] || d.close || d.value])
    const foreData = forecast.map(d => [new Date(d.date).getTime(), d.predictedClose || d.value])
    const rangeData = forecast.map(d => {
      const val = d.predictedClose || d.value || 0
      return [new Date(d.date).getTime(), val * 0.95, val * 1.05]
    })

    const series = [
      {
        name: 'History', data: histData, color: '#3b82f6', zIndex: 2,
        marker: { enabled: false }, lineWidth: 2,
      },
      {
        name: 'Aggregate Forecast', data: foreData, color: '#60a5fa',
        dashStyle: 'ShortDash', zIndex: 3, marker: { enabled: false }, lineWidth: 2,
      },
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
        color: ['#f59e0b', '#a855f7', '#ec4899', '#14b8a6'][i % 4],
        dashStyle: 'Dot', zIndex: 2, opacity: 0.7, marker: { enabled: false },
      })
    })

    return {
      chart: {
        backgroundColor: '#1e293b',
        style: { fontFamily: 'Inter, sans-serif' },
        height: 400,
        // Enable zoom by dragging
        zooming: { type: 'x', mouseWheel: { enabled: true } },
        panning: { enabled: true, type: 'x' },
        panKey: 'shift',
        animation: { duration: 300 },
      },
      title: { text: null },
      credits: { enabled: false },
      xAxis: {
        type: 'datetime',
        gridLineColor: '#334155',
        labels: { style: { color: '#94a3b8' } },
        lineColor: '#334155',
      },
      yAxis: {
        title: { text: null },
        gridLineColor: '#334155',
        labels: { style: { color: '#94a3b8', formatter() { return '$' + this.value.toLocaleString() } } },
      },
      legend: {
        itemStyle: { color: '#94a3b8' },
        itemHoverStyle: { color: '#e2e8f0' },
      },
      tooltip: {
        shared: true,
        backgroundColor: '#1e293b',
        borderColor: '#334155',
        style: { color: '#e2e8f0' },
        xDateFormat: '%b %e, %Y',
      },
      plotOptions: {
        series: { animation: { duration: 400 } },
      },
      // Reset zoom button style
      resetZoomButton: {
        theme: {
          fill: '#1e293b',
          stroke: '#3b82f6',
          style: { color: '#60a5fa' },
          r: 4,
        }
      },
      series,
    }
  }, [history, forecast, valueKey, separateLines, showConfidence])

  if (!history.length && !forecast.length) return <div className="no-data">No chart data available.</div>
  return (
    <div className="chart-container">
      <div style={{ fontSize: '0.72rem', color: '#64748b', marginBottom: '0.25rem', textAlign: 'right' }}>
        Drag to zoom · Shift+drag to pan · Scroll to zoom
      </div>
      <HighchartsReact highcharts={Highcharts} options={options} />
    </div>
  )
}

// ---------------------------------------------------------------------------
// RiskMeter
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

// ---------------------------------------------------------------------------
// Icon maps
// ---------------------------------------------------------------------------
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

function GlobeDot({ riskColor, isSelected }) {
  return (
    <g>
      <circle r="10" fill="#0f172a" stroke={riskColor} strokeWidth="2" />
      <circle r="4" fill={riskColor} />
      {isSelected && <circle r="14" fill="none" stroke={riskColor} strokeWidth="1" opacity="0.5" />}
    </g>
  )
}

// ---------------------------------------------------------------------------
// Congressional Trades Tab
// ---------------------------------------------------------------------------
const AMOUNT_ORDER = ['$1,001 - $15,000', '$15,001 - $50,000', '$50,001 - $100,000', '$100,001 - $250,000', '$250,001 - $500,000', '$500,001 - $1,000,000', 'Over $1,000,000']

const PARTY_COLORS = { Democrat: '#3b82f6', Republican: '#ef4444', Independent: '#a855f7' }

function InsiderTrades() {
  const [trades, setTrades] = useState([])
  const [topMovers, setTopMovers] = useState([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [fetchedAt, setFetchedAt] = useState(null)
  const [offset, setOffset] = useState(0)

  // Filters
  const [chamber, setChamber] = useState('all')
  const [ticker, setTicker] = useState('')
  const [member, setMember] = useState('')
  const [tradeType, setTradeType] = useState('all')
  const [party, setParty] = useState('all')
  const [days, setDays] = useState(90)
  const [activeMovers, setActiveMovers] = useState(null)

  const LIMIT = 50

  const fetchTrades = useCallback(async (newOffset = 0) => {
    setLoading(true)
    try {
      const params = new URLSearchParams({
        chamber,
        ticker: activeMovers || ticker,
        member,
        trade_type: tradeType,
        party,
        days,
        limit: LIMIT,
        offset: newOffset,
      })
      const r = await fetch(`/api/insider/trades?${params}`)
      const j = await r.json()
      setTrades(j.trades || [])
      setTopMovers(j.top_movers || [])
      setTotal(j.total || 0)
      setFetchedAt(j.fetched_at)
      setOffset(newOffset)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }, [chamber, ticker, member, tradeType, party, days, activeMovers])

  useEffect(() => { fetchTrades(0) }, [fetchTrades])

  function handleMoversClick(tk) {
    setActiveMovers(prev => prev === tk ? null : tk)
  }

  const typeColor = (type = '') => {
    if (type.toLowerCase().includes('purchase')) return '#22c55e'
    if (type.toLowerCase().includes('sale')) return '#ef4444'
    return '#94a3b8'
  }

  return (
    <div className="insider-page">
      {/* Header */}
      <div className="insider-header">
        <div>
          <h2 style={{ margin: 0, color: '#e2e8f0' }}>Congressional Trade Tracker</h2>
          <div style={{ color: '#64748b', fontSize: '0.8rem', marginTop: '0.25rem' }}>
            {fetchedAt ? `Data from ${new Date(fetchedAt).toLocaleString()}` : 'Loading disclosures...'}
            {' · '}{total.toLocaleString()} trades matching filters
          </div>
        </div>
        <button className="refresh-btn" onClick={() => fetchTrades(0)} disabled={loading}
          style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <RefreshCw01 width={14} height={14} />
          {loading ? 'Loading...' : 'Refresh'}
        </button>
      </div>

      {/* Filters */}
      <div className="card" style={{ padding: '1rem', marginBottom: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem', color: '#94a3b8', fontSize: '0.85rem' }}>
          <FilterLines width={14} height={14} /> <strong>Filters</strong>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', alignItems: 'flex-end' }}>
          {/* Chamber */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <label style={{ color: '#64748b', fontSize: '0.72rem' }}>CHAMBER</label>
            <select value={chamber} onChange={e => setChamber(e.target.value)} className="filter-select">
              <option value="all">All</option>
              <option value="house">House</option>
              <option value="senate">Senate</option>
            </select>
          </div>
          {/* Ticker */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <label style={{ color: '#64748b', fontSize: '0.72rem' }}>TICKER</label>
            <input
              value={ticker} onChange={e => { setTicker(e.target.value.toUpperCase()); setActiveMovers(null) }}
              placeholder="e.g. AAPL" className="filter-input"
            />
          </div>
          {/* Member */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <label style={{ color: '#64748b', fontSize: '0.72rem' }}>MEMBER</label>
            <input value={member} onChange={e => setMember(e.target.value)} placeholder="Name search" className="filter-input" />
          </div>
          {/* Type */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <label style={{ color: '#64748b', fontSize: '0.72rem' }}>TYPE</label>
            <select value={tradeType} onChange={e => setTradeType(e.target.value)} className="filter-select">
              <option value="all">All</option>
              <option value="purchase">Purchase</option>
              <option value="sale">Sale</option>
            </select>
          </div>
          {/* Party */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <label style={{ color: '#64748b', fontSize: '0.72rem' }}>PARTY</label>
            <select value={party} onChange={e => setParty(e.target.value)} className="filter-select">
              <option value="all">All Parties</option>
              <option value="Democrat">Democrat</option>
              <option value="Republican">Republican</option>
              <option value="Independent">Independent</option>
            </select>
          </div>
          {/* Days */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <label style={{ color: '#64748b', fontSize: '0.72rem' }}>LAST N DAYS</label>
            <select value={days} onChange={e => setDays(Number(e.target.value))} className="filter-select">
              <option value={30}>30 days</option>
              <option value={90}>90 days</option>
              <option value={180}>180 days</option>
              <option value={365}>1 year</option>
              <option value={9999}>All time</option>
            </select>
          </div>
        </div>
      </div>

      <div className="insider-layout">
        {/* Top movers */}
        <div className="card" style={{ padding: '1rem', marginBottom: '1rem' }}>
          <h4 style={{ margin: '0 0 0.75rem', color: '#e2e8f0', fontSize: '0.85rem' }}>
            🔥 Most Traded Tickers
          </h4>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
            {topMovers.map(m => (
              <button key={m.ticker}
                onClick={() => handleMoversClick(m.ticker)}
                style={{
                  padding: '0.3rem 0.6rem', borderRadius: '0.4rem', border: 'none',
                  background: activeMovers === m.ticker ? '#3b82f6' : '#1e293b',
                  cursor: 'pointer', fontSize: '0.75rem',
                  color: activeMovers === m.ticker ? 'white' : '#e2e8f0',
                  display: 'flex', alignItems: 'center', gap: '0.4rem',
                }}>
                <strong>{m.ticker}</strong>
                <span style={{ color: '#22c55e', fontSize: '0.7rem' }}>▲{m.buys}</span>
                <span style={{ color: '#ef4444', fontSize: '0.7rem' }}>▼{m.sells}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Trades table */}
        <div className="card" style={{ padding: '1rem', overflowX: 'auto' }}>
          {loading ? (
            <div style={{ textAlign: 'center', padding: '2rem', color: '#64748b' }}>Loading trades...</div>
          ) : trades.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '2rem', color: '#64748b' }}>No trades match current filters.</div>
          ) : (
            <>
              <table className="insider-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Member</th>
                    <th>Chamber</th>
                    <th>Party</th>
                    <th>Ticker</th>
                    <th>Asset</th>
                    <th>Type</th>
                    <th>Amount</th>
                  </tr>
                </thead>
                <tbody>
                  {trades.map((t, i) => (
                    <tr key={i} className="insider-row">
                      <td style={{ color: '#64748b', whiteSpace: 'nowrap' }}>{t.transaction_date}</td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                          <Users01 width={12} height={12} color="#64748b" />
                          <span style={{ color: '#e2e8f0', fontSize: '0.82rem' }}>{t.member}</span>
                        </div>
                        {t.state && <div style={{ color: '#475569', fontSize: '0.7rem' }}>{t.state}{t.district ? ` · ${t.district}` : ''}</div>}
                      </td>
                      <td>
                        <span style={{
                          fontSize: '0.72rem', padding: '0.15rem 0.4rem',
                          borderRadius: '999px', background: '#1e293b',
                          color: t.source === 'Senate' ? '#a855f7' : '#60a5fa',
                          border: `1px solid ${t.source === 'Senate' ? '#a855f7' : '#60a5fa'}44`
                        }}>
                          {t.source === 'Senate' ? '🏛 Senate' : '🏢 House'}
                        </span>
                      </td>
                      <td>
                        <span style={{
                          fontSize: '0.72rem', padding: '0.15rem 0.4rem', borderRadius: '999px',
                          color: PARTY_COLORS[t.party] || '#94a3b8',
                          background: (PARTY_COLORS[t.party] || '#94a3b8') + '22',
                        }}>
                          {t.party || '—'}
                        </span>
                      </td>
                      <td>
                        <span style={{
                          fontWeight: 'bold', color: '#60a5fa', fontSize: '0.85rem',
                          fontFamily: 'monospace', letterSpacing: '0.05em'
                        }}>
                          {t.ticker || '—'}
                        </span>
                      </td>
                      <td style={{ color: '#94a3b8', fontSize: '0.78rem', maxWidth: '180px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {t.asset || '—'}
                      </td>
                      <td>
                        <span style={{
                          display: 'flex', alignItems: 'center', gap: '0.3rem',
                          color: typeColor(t.type), fontSize: '0.8rem', fontWeight: '500',
                        }}>
                          {t.type.toLowerCase().includes('purchase')
                            ? <TrendUp01 width={12} height={12} />
                            : t.type.toLowerCase().includes('sale')
                              ? <TrendDown01 width={12} height={12} />
                              : null}
                          {t.type}
                        </span>
                      </td>
                      <td style={{ color: '#94a3b8', fontSize: '0.8rem', whiteSpace: 'nowrap' }}>{t.amount || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* Pagination */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1rem', color: '#64748b', fontSize: '0.8rem' }}>
                <span>Showing {offset + 1}–{Math.min(offset + LIMIT, total)} of {total.toLocaleString()}</span>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button className="small-btn" disabled={offset === 0} onClick={() => fetchTrades(offset - LIMIT)}>← Prev</button>
                  <button className="small-btn" disabled={offset + LIMIT >= total} onClick={() => fetchTrades(offset + LIMIT)}>Next →</button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Portfolio Dashboard
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Global Market Intelligence
// ---------------------------------------------------------------------------
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
  const [nodeCount, setNodeCount] = useState(12)
  const [cacheInfo, setCacheInfo] = useState(null)

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
    // Check cache first
    const cached = gmiCacheGet(q, nodeCount)
    if (cached) {
      setData(cached)
      setKeyword(q)
      setCacheInfo('from cache')
      if (cached.nodes?.length) setRotation([-cached.nodes[0].lon, -cached.nodes[0].lat + 20])
      return
    }

    setLoading(true)
    setSelected(null)
    setData(null)
    setCacheInfo(null)
    try {
      const r = await fetch(`/api/globe-supply-chain?query=${encodeURIComponent(q)}&node_count=${nodeCount}`)
      const j = await r.json()
      setData(j)
      setKeyword(q)
      gmiCacheSet(q, nodeCount, j)
      setCacheInfo('fresh')
      if (j.nodes?.length) setRotation([-j.nodes[0].lon, -j.nodes[0].lat + 20])
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
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
      <div className="gmi-header">
        <div className="search-bar" style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <input
            value={inputVal}
            onChange={e => setInputVal(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') runSearch(inputVal) }}
            placeholder="Enter company, commodity or sector..."
            style={{ flex: 1, minWidth: '200px' }}
          />
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#94a3b8', fontSize: '0.8rem', whiteSpace: 'nowrap' }}>
            Nodes:
            <input type="number" min={3} max={20} value={nodeCount}
              onChange={e => setNodeCount(Math.min(20, Math.max(3, parseInt(e.target.value) || 12)))}
              style={{ width: '3.5rem', padding: '0.3rem', background: '#1e293b', border: '1px solid #334155', borderRadius: '0.4rem', color: 'white', textAlign: 'center' }}
            />
          </div>
          <button onClick={() => runSearch(inputVal)} disabled={loading} className="refresh-btn"
            style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            {loading ? <><RefreshCw01 width={14} height={14} /> Analyzing...</> : <><SearchLg width={14} height={14} /> Analyze Supply Chain</>}
          </button>
          {cacheInfo && (
            <span style={{
              fontSize: '0.72rem', padding: '0.2rem 0.5rem', borderRadius: '999px',
              background: cacheInfo === 'from cache' ? '#1e293b' : '#14532d',
              color: cacheInfo === 'from cache' ? '#64748b' : '#22c55e',
              border: `1px solid ${cacheInfo === 'from cache' ? '#334155' : '#22c55e'}`,
            }}>
              {cacheInfo === 'from cache' ? '⚡ Cached result' : '✓ Fresh data'}
            </span>
          )}
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
            <circle cx={width / 2} cy={height / 2} r={scale} fill="#0a0f1e" />
            <path d={path(d3geo.geoGraticule()())} fill="none" stroke="#1e293b" strokeWidth="0.4" opacity="0.6" />
            {land && <path d={path(land)} fill="#111827" stroke="rgba(255,255,255,0.35)" strokeWidth="0.6" />}

            {layers.routes && data?.routes?.map((r, i) => {
              const d = getRoutePath(r.from, r.to)
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
                  {layers.risk && <circle r="16" fill="none" stroke={riskColor} strokeWidth="1" opacity="0.3" />}
                  <GlobeDot riskColor={riskColor} isSelected={selected?.name === node.name} />
                  {layers.risk && node.risk?.score > 0.4 && (
                    <g transform="translate(10,-10)">
                      <circle r="6" fill={riskColor} />
                      <text textAnchor="middle" dominantBaseline="central" fontSize="7" fill="white" fontWeight="bold">
                        {Math.round(node.risk.score * 10)}
                      </text>
                    </g>
                  )}
                  <text y="22" textAnchor="middle" fontSize="8" fill="#94a3b8" style={{ pointerEvents: 'none' }}>
                    {node.city || node.name}
                  </text>
                </g>
              )
            })}
          </svg>

          <div style={{ position: 'absolute', bottom: '1rem', left: '1rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            {Object.keys(layers).map(k => (
              <button key={k} onClick={() => setLayers(p => ({ ...p, [k]: !p[k] }))}
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
                {[...data.nodes].sort((a, b) => (b.risk?.score || 0) - (a.risk?.score || 0)).map((node, i) => (
                  <div key={i} onClick={() => { setSelected(node); setRotation([-node.lon, -node.lat + 20]) }}
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

// ---------------------------------------------------------------------------
// App root
// ---------------------------------------------------------------------------
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
    const t = lookup.trim().toUpperCase()
    if (!t || portfolio.some(p => p.ticker === t)) return
    setPortfolio(p => [...p, { ticker: t, shares: 1 }])
  }

  return (
    <div className="app-container">
      <nav className="navbar">
        <div className="nav-brand">SENTIDEX <span className="pro-tag">PRO</span></div>
        <div className="nav-links">
          <button className={page === 'portfolio' ? 'active' : ''} onClick={() => setPage('portfolio')}>Portfolio</button>
          <button className={page === 'gmi' ? 'active' : ''} onClick={() => setPage('gmi')}>GMI</button>
          <button className={page === 'insider' ? 'active' : ''} onClick={() => setPage('insider')}>
            Insider Trades
          </button>
        </div>
      </nav>
      <main className="main-content">
        {page === 'portfolio' && (
          <PortfolioDashboard
            portfolio={portfolio} portfolioData={portfolioData} refresh={refresh} loading={loading}
            lookup={lookup} setLookup={setLookup} addTicker={addTicker}
            selectedSources={selectedSources} toggleSource={toggleSource} sourceOptions={sourceOptions}
            forecastBySelectedSources={forecastBySelectedSources} viewStates={viewStates} toggleViewState={toggleViewState}
          />
        )}
        {page === 'gmi' && <GlobalMarketIntelligence keyword={keyword} setKeyword={setKeyword} />}
        {page === 'insider' && <InsiderTrades />}
      </main>
    </div>
  )
}
