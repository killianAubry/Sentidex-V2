import { useMemo, useState } from 'react'

function LineChart({ history, forecast, label }) {
  const allPoints = [...history.map((p) => ({ ...p, v: p.close })), ...forecast.map((p) => ({ ...p, v: p.predictedClose }))]
  const width = 900
  const height = 320
  const pad = 32
  const min = Math.min(...allPoints.map((p) => p.v))
  const max = Math.max(...allPoints.map((p) => p.v))

  const points = allPoints
    .map((p, i) => {
      const x = pad + (i / Math.max(allPoints.length - 1, 1)) * (width - pad * 2)
      const y = height - pad - ((p.v - min) / Math.max(max - min, 1e-6)) * (height - pad * 2)
      return `${x},${y}`
    })
    .join(' ')

  const dividerIndex = history.length - 1
  const dividerX = pad + (dividerIndex / Math.max(allPoints.length - 1, 1)) * (width - pad * 2)

  return (
    <div>
      <h3>{label}</h3>
      <svg viewBox={`0 0 ${width} ${height}`} className="chart">
        <polyline fill="none" stroke="#4f46e5" strokeWidth="3" points={points} />
        <line x1={dividerX} y1={pad / 2} x2={dividerX} y2={height - pad / 2} stroke="#ef4444" strokeDasharray="6 6" />
        <text x={dividerX + 6} y={pad} fill="#ef4444" fontSize="12">Forecast starts</text>
      </svg>
      <small>
        Range: ${min.toFixed(2)} - ${max.toFixed(2)}
      </small>
    </div>
  )
}

export function App() {
  const [ticker, setTicker] = useState('AAPL')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [view, setView] = useState('combined')
  const [selectedOutlet, setSelectedOutlet] = useState('')

  const outlets = useMemo(() => {
    if (!result) return []
    return Object.keys(result.data.combinedChartData.perOutletForecast)
  }, [result])

  async function runForecast(e) {
    e.preventDefault()
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const res = await fetch(`/api/forecast?ticker=${encodeURIComponent(ticker)}`)
      if (!res.ok) throw new Error(`Request failed (${res.status})`)
      const json = await res.json()
      setResult(json)
      const firstOutlet = Object.keys(json.data.combinedChartData.perOutletForecast)[0]
      setSelectedOutlet(firstOutlet || '')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const history = result?.data?.combinedChartData?.historicalPrices || []
  const combinedForecast = result?.data?.combinedChartData?.combinedForecast || []
  const outletData = selectedOutlet ? result?.data?.combinedChartData?.perOutletForecast?.[selectedOutlet] : null

  return (
    <main className="container">
      <h1>Sentidex Stock Forecast</h1>
      <p>Generate per-news-outlet sentiment forecasts and a combined one-week outlook.</p>

      <form onSubmit={runForecast} className="row">
        <input value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} placeholder="Ticker (e.g. TSLA)" />
        <button type="submit" disabled={loading}>{loading ? 'Running...' : 'Forecast'}</button>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <section>
          <div className="row">
            <button onClick={() => setView('combined')} className={view === 'combined' ? 'active' : ''}>Combined sentiment</button>
            <button onClick={() => setView('outlet')} className={view === 'outlet' ? 'active' : ''}>Individual outlets</button>
            {view === 'outlet' && (
              <select value={selectedOutlet} onChange={(e) => setSelectedOutlet(e.target.value)}>
                {outlets.map((outlet) => <option key={outlet}>{outlet}</option>)}
              </select>
            )}
          </div>

          {view === 'combined' ? (
            <LineChart
              history={history}
              forecast={combinedForecast}
              label={`Combined Sentiment Score: ${result.data.combinedChartData.combinedSentimentScore}`}
            />
          ) : (
            outletData && (
              <LineChart
                history={history}
                forecast={outletData.nextWeekForecast}
                label={`${selectedOutlet} sentiment: ${outletData.avgSentiment}`}
              />
            )
          )}

          <h3>Generated JSON files</h3>
          <pre>{JSON.stringify(result.files, null, 2)}</pre>
        </section>
      )}
    </main>
  )
}
