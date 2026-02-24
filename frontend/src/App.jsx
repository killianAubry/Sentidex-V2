import { useMemo, useState } from 'react'
import {
  Chart,
  ChartCanvas,
  LineSeries,
  XAxis,
  YAxis,
  discontinuousTimeScaleProviderBuilder,
  MouseCoordinateX,
  MouseCoordinateY,
  CrossHairCursor,
} from 'react-financial-charts'
import { timeFormat } from 'd3-time-format'
import { format } from 'd3-format'

function buildSeries(history = [], forecast = []) {
  return [
    ...history.map((x) => ({ date: new Date(x.date), close: x.close, kind: 'history' })),
    ...forecast.map((x) => ({ date: new Date(x.date), close: x.predictedClose, kind: 'forecast' })),
  ]
}

function StockCenterChart({ history, combinedForecast, transformerForecast, selectedOutletForecast, mode, showTransformer }) {
  const chartData = useMemo(() => {
    const outletSeries = mode === 'outlet' ? selectedOutletForecast : combinedForecast
    return buildSeries(history, outletSeries)
  }, [history, combinedForecast, selectedOutletForecast, mode])

  if (!chartData.length) return <div className="panel">No chart data yet.</div>

  const xScaleProvider = discontinuousTimeScaleProviderBuilder().inputDateAccessor((d) => d.date)
  const { data, xScale, xAccessor, displayXAccessor } = xScaleProvider(chartData)
  const transformer = showTransformer ? buildSeries(history, transformerForecast) : []
  const txScaled = transformer.length ? xScaleProvider(transformer).data : []

  return (
    <div className="panel chart-panel">
      <h2>Price + Forecast</h2>
      <ChartCanvas height={460} width={900} ratio={1} margin={{ left: 60, right: 60, top: 20, bottom: 30 }}
        data={data} seriesName="Price" xScale={xScale} xAccessor={xAccessor} displayXAccessor={displayXAccessor}>
        <Chart id={1} yExtents={(d) => d.close}>
          <XAxis />
          <YAxis />
          <MouseCoordinateX displayFormat={timeFormat('%Y-%m-%d')} />
          <MouseCoordinateY displayFormat={format('.2f')} />
          <LineSeries yAccessor={(d) => d.close} strokeStyle="#22d3ee" />
          {showTransformer && txScaled.length > 0 && (
            <LineSeries data={txScaled} yAccessor={(d) => d.close} strokeStyle="#f59e0b" />
          )}
        </Chart>
        <CrossHairCursor />
      </ChartCanvas>
      <p className="muted">Cyan line = selected sentiment forecast. Amber line = transformer forecast.</p>
    </div>
  )
}

export function App() {
  const [ticker, setTicker] = useState('AAPL')
  const [response, setResponse] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const [mode, setMode] = useState('combined')
  const [sentimentModel, setSentimentModel] = useState('transformer')
  const [selectedOutlet, setSelectedOutlet] = useState('')
  const [showTransformer, setShowTransformer] = useState(true)
  const [showMacro, setShowMacro] = useState(true)
  const [showEarnings, setShowEarnings] = useState(true)
  const [showOptionsFlow, setShowOptionsFlow] = useState(true)

  const outlets = useMemo(() => {
    if (!response) return []
    return Object.keys(response.data.perOutletForecast)
  }, [response])

  const data = response?.data
  const history = data?.historicalPrices || []
  const combinedForecast = data?.combinedForecast || []
  const transformerForecast = data?.transformerForecast || []
  const selectedOutletForecast = selectedOutlet ? data?.perOutletForecast?.[selectedOutlet]?.nextWeekForecast || [] : []

  async function onSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const r = await fetch(`/api/forecast?ticker=${encodeURIComponent(ticker)}&sentiment_model=${sentimentModel}`)
      if (!r.ok) throw new Error(`Forecast request failed (${r.status})`)
      const json = await r.json()
      setResponse(json)
      const firstOutlet = Object.keys(json.data.perOutletForecast)[0]
      setSelectedOutlet(firstOutlet || '')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="dashboard">
      <header className="panel">
        <h1>Sentidex Pro Terminal</h1>
        <form className="toolbar" onSubmit={onSubmit}>
          <input value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} placeholder="Ticker" />
          <select value={sentimentModel} onChange={(e) => setSentimentModel(e.target.value)}>
            <option value="transformer">Transformer sentiment</option>
            <option value="lexicon">Lexicon sentiment</option>
          </select>
          <button type="submit" disabled={loading}>{loading ? 'Running...' : 'Run Forecast'}</button>
        </form>
        {error && <p className="error">{error}</p>}
      </header>

      <section className="layout">
        <aside className="panel side">
          <h3>Forecast Controls</h3>
          <label><input type="radio" checked={mode === 'combined'} onChange={() => setMode('combined')} /> Combined sentiment</label>
          <label><input type="radio" checked={mode === 'outlet'} onChange={() => setMode('outlet')} /> By outlet</label>
          {mode === 'outlet' && (
            <select value={selectedOutlet} onChange={(e) => setSelectedOutlet(e.target.value)}>
              {outlets.map((outlet) => <option key={outlet}>{outlet}</option>)}
            </select>
          )}
          <hr />
          <h3>Feature Toggles</h3>
          <label><input type="checkbox" checked={showTransformer} onChange={() => setShowTransformer((v) => !v)} /> Transformer forecast overlay</label>
          <label><input type="checkbox" checked={showMacro} onChange={() => setShowMacro((v) => !v)} /> Macro panel</label>
          <label><input type="checkbox" checked={showEarnings} onChange={() => setShowEarnings((v) => !v)} /> Earnings panel</label>
          <label><input type="checkbox" checked={showOptionsFlow} onChange={() => setShowOptionsFlow((v) => !v)} /> Options flow panel</label>
        </aside>

        <StockCenterChart
          history={history}
          combinedForecast={combinedForecast}
          transformerForecast={transformerForecast}
          selectedOutletForecast={selectedOutletForecast}
          mode={mode}
          showTransformer={showTransformer}
        />

        <aside className="panel side right">
          <h3>Providers Used</h3>
          <ul>{(data?.newsProvidersUsed || []).map((x) => <li key={x}>{x}</li>)}</ul>

          {showMacro && data?.macroData && (
            <div>
              <h3>Macro</h3>
              <pre>{JSON.stringify(data.macroData, null, 2)}</pre>
            </div>
          )}

          {showEarnings && data?.earningsData && (
            <div>
              <h3>Earnings</h3>
              <pre>{JSON.stringify(data.earningsData, null, 2)}</pre>
            </div>
          )}

          {showOptionsFlow && data?.optionsFlow && (
            <div>
              <h3>Options Flow</h3>
              <pre>{JSON.stringify(data.optionsFlow, null, 2)}</pre>
            </div>
          )}
        </aside>
      </section>

      {response && (
        <section className="panel">
          <h3>Generated JSON files</h3>
          <pre>{JSON.stringify(response.files, null, 2)}</pre>
        </section>
      )}
    </main>
  )
}
