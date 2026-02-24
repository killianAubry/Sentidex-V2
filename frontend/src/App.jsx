import { useEffect, useMemo, useState } from 'react'
import { ChartCanvas, Chart } from 'react-stockcharts'
import { LineSeries } from 'react-stockcharts/lib/series'
import { XAxis, YAxis } from 'react-stockcharts/lib/axes'
import { CrossHairCursor, MouseCoordinateX, MouseCoordinateY } from 'react-stockcharts/lib/coordinates'
import { discontinuousTimeScaleProvider } from 'react-stockcharts/lib/scale'

const STORAGE_KEY = 'sentidex-portfolio-v1'

const emptyPortfolio = [
  { ticker: 'AAPL', shares: 4 },
  { ticker: 'MSFT', shares: 2 },
  { ticker: 'NVDA', shares: 1 },
]

function normalizeSeries(history, forecast, valueKey = 'close') {
  return [
    ...history.map((x) => ({ date: new Date(x.date), value: x[valueKey] })),
    ...forecast.map((x) => ({ date: new Date(x.date), value: x[valueKey] })),
  ]
}

function MinimalStockChart({ title, history = [], forecast = [], valueKey = 'close' }) {
  const data = useMemo(() => normalizeSeries(history, forecast, valueKey), [history, forecast, valueKey])

  if (!data.length) return <div className="card">No data available.</div>

  const xScaleProvider = discontinuousTimeScaleProvider.inputDateAccessor((d) => d.date)
  const { data: chartData, xScale, xAccessor, displayXAccessor } = xScaleProvider(data)

  return (
    <section className="card chart-card">
      <h2>{title}</h2>
      <ChartCanvas
        width={980}
        height={440}
        ratio={1}
        margin={{ left: 50, right: 50, top: 20, bottom: 30 }}
        data={chartData}
        seriesName={title}
        xScale={xScale}
        xAccessor={xAccessor}
        displayXAccessor={displayXAccessor}
      >
        <Chart id={1} yExtents={(d) => d.value}>
          <XAxis axisAt="bottom" orient="bottom" />
          <YAxis axisAt="left" orient="left" />
          <MouseCoordinateX />
          <MouseCoordinateY />
          <LineSeries yAccessor={(d) => d.value} stroke="#0ea5e9" />
        </Chart>
        <CrossHairCursor />
      </ChartCanvas>
    </section>
  )
}

export function App() {
  const [page, setPage] = useState('dashboard')
  const [queryTicker, setQueryTicker] = useState('AAPL')
  const [popular, setPopular] = useState([])
  const [portfolio, setPortfolio] = useState(emptyPortfolio)
  const [portfolioData, setPortfolioData] = useState(null)
  const [selectedPortfolioTicker, setSelectedPortfolioTicker] = useState('')
  const [sortMode, setSortMode] = useState('best')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      try {
        const parsed = JSON.parse(raw)
        if (Array.isArray(parsed) && parsed.length) setPortfolio(parsed)
      } catch {
        // ignore malformed local state
      }
    }
  }, [])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(portfolio))
  }, [portfolio])

  useEffect(() => {
    fetch('/api/quotes/popular')
      .then((r) => r.json())
      .then((json) => setPopular(json.quotes || []))
      .catch(() => setPopular([]))
  }, [])

  async function refreshPortfolioForecast() {
    setLoading(true)
    try {
      const res = await fetch('/api/portfolio/forecast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ positions: portfolio, sentiment_model: 'transformer' }),
      })
      const json = await res.json()
      setPortfolioData(json)
      const first = json?.positions?.[0]?.ticker
      if (first && !selectedPortfolioTicker) setSelectedPortfolioTicker(first)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (portfolio.length) refreshPortfolioForecast()
  }, [portfolio])

  async function addTicker() {
    const ticker = queryTicker.trim().toUpperCase()
    if (!ticker) return
    if (portfolio.some((p) => p.ticker === ticker)) return
    setPortfolio((prev) => [...prev, { ticker, shares: 1 }])
  }

  const sortedPositions = useMemo(() => {
    const list = [...(portfolioData?.positions || [])]
    if (sortMode === 'best') {
      list.sort((a, b) => b.forecastDeltaPct - a.forecastDeltaPct)
    } else if (sortMode === 'worst') {
      list.sort((a, b) => a.forecastDeltaPct - b.forecastDeltaPct)
    } else {
      list.sort((a, b) => a.ticker.localeCompare(b.ticker))
    }
    return list
  }, [portfolioData, sortMode])

  const selectedPosition = sortedPositions.find((x) => x.ticker === selectedPortfolioTicker)

  return (
    <main className="minimal-dashboard">
      <header className="topbar">
        <div className="brand">Sentidex</div>
        <div className="lookup">
          <input value={queryTicker} onChange={(e) => setQueryTicker(e.target.value.toUpperCase())} placeholder="Lookup ticker" />
          <button onClick={addTicker}>Add to portfolio</button>
          <button onClick={refreshPortfolioForecast} disabled={loading}>{loading ? 'Refreshing...' : 'Run Forecast'}</button>
        </div>
        <nav>
          <button className={page === 'dashboard' ? 'active' : ''} onClick={() => setPage('dashboard')}>Dashboard</button>
          <button className={page === 'portfolio' ? 'active' : ''} onClick={() => setPage('portfolio')}>Portfolio View</button>
        </nav>
      </header>

      <section className="popular-row">
        {popular.map((q) => (
          <article key={q.ticker} className="ticker-card" onClick={() => setQueryTicker(q.ticker)}>
            <h4>{q.ticker}</h4>
            <p>${q.price}</p>
          </article>
        ))}
      </section>

      {page === 'dashboard' ? (
        <>
          <MinimalStockChart
            title="Portfolio valuation (historical + forecast)"
            history={portfolioData?.portfolioValuation?.history || []}
            forecast={portfolioData?.portfolioValuation?.forecast || []}
            valueKey="value"
          />
          <section className="card stats-grid">
            <div>
              <h4>Current Portfolio Value</h4>
              <p>${portfolioData?.portfolioValuation?.currentValue ?? '-'}</p>
            </div>
            <div>
              <h4>Expected Week-End Value</h4>
              <p>${portfolioData?.portfolioValuation?.forecastWeekEndValue ?? '-'}</p>
            </div>
            <div>
              <h4>Positions</h4>
              <p>{portfolio.length}</p>
            </div>
          </section>
        </>
      ) : (
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
            {sortedPositions.map((p) => (
              <button key={p.ticker} className="position-btn" onClick={() => setSelectedPortfolioTicker(p.ticker)}>
                <span>{p.ticker} ({p.shares} sh)</span>
                <strong>{p.forecastDeltaPct}%</strong>
              </button>
            ))}
          </aside>

          <MinimalStockChart
            title={selectedPosition ? `${selectedPosition.ticker} individual chart` : 'Select a stock'}
            history={selectedPosition?.historicalPrices || []}
            forecast={(selectedPosition?.combinedForecast || []).map((x) => ({ ...x, close: x.predictedClose }))}
            valueKey="close"
          />
        </section>
      )}
    </main>
  )
}
