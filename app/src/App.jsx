import { useCallback, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import './App.css'

const LABEL_ORDER = [
  { key: 'negative', title: 'Negative', classId: 0 },
  { key: 'neutral', title: 'Neutral', classId: 1 },
  { key: 'positive', title: 'Positive', classId: 2 },
]

const BAR_COLORS = {
  negative: '#c2410c',
  neutral: '#6b7280',
  positive: '#15803d',
}

function apiBase() {
  const base = import.meta.env.VITE_API_BASE_URL ?? ''
  return String(base).replace(/\/$/, '')
}

function buildChartRows(probabilities) {
  return LABEL_ORDER.map(({ key, title, classId }) => ({
    name: title,
    key,
    classId,
    p: probabilities[key] ?? 0,
  }))
}

export default function App() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  const analyze = useCallback(async () => {
    setError(null)
    setResult(null)
    const trimmed = text.trim()
    if (!trimmed) {
      setError('Enter some text to analyze.')
      return
    }
    setLoading(true)
    try {
      const url = `${apiBase()}/v1/predict`
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: trimmed }),
      })
      const body = await res.json().catch(() => ({}))
      if (!res.ok) {
        const detail =
          typeof body.detail === 'string'
            ? body.detail
            : Array.isArray(body.detail)
              ? body.detail.map((d) => d.msg ?? JSON.stringify(d)).join('; ')
              : res.statusText
        throw new Error(detail || `Request failed (${res.status})`)
      }
      setResult(body)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed.')
    } finally {
      setLoading(false)
    }
  }, [text])

  const chartData = result ? buildChartRows(result.probabilities) : []

  return (
    <div className="sentiment-app">
      <header className="sentiment-header">
        <h1>Tweet sentiment</h1>
        <p className="sentiment-lede">
          Runs against your FastAPI{' '}
          <code className="inline-code">POST /v1/predict</code> (Vite proxies{' '}
          <code className="inline-code">/v1</code> to port 8000 in dev).
        </p>
      </header>

      <section className="sentiment-panel">
        <label htmlFor="tweet-input" className="field-label">
          Text
        </label>
        <textarea
          id="tweet-input"
          className="tweet-input"
          rows={4}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste a tweet or sentence…"
          disabled={loading}
        />
        <div className="actions">
          <button
            type="button"
            className="btn-primary"
            onClick={() => void analyze()}
            disabled={loading}
          >
            {loading ? 'Analyzing…' : 'Analyze'}
          </button>
        </div>
        {error ? (
          <p className="error-banner" role="alert">
            {error}
          </p>
        ) : null}
      </section>

      {result ? (
        <section className="results" aria-live="polite">
          <h2>Result</h2>
          <div className="result-summary">
            <div className="prediction-pill">
              <span className="prediction-label">Predicted</span>
              <span className="prediction-value">{result.sentiment}</span>
              <span className="prediction-meta">
                class {result.class_id} · {(result.confidence * 100).toFixed(1)}%
                confidence
              </span>
            </div>
            <p className="original-text">
              <span className="original-label">Input</span>
              {result.text}
            </p>
          </div>

          <h3 className="subsection-title">Class probabilities</h3>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart
                data={chartData}
                layout="vertical"
                margin={{ top: 8, right: 24, left: 8, bottom: 8 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
                <XAxis
                  type="number"
                  domain={[0, 1]}
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                  stroke="var(--chart-axis)"
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={72}
                  stroke="var(--chart-axis)"
                  tick={{ fontSize: 13 }}
                />
                <Tooltip
                  formatter={(value) => `${((value ?? 0) * 100).toFixed(2)}%`}
                  contentStyle={{
                    background: 'var(--panel-bg)',
                    border: '1px solid var(--border)',
                    borderRadius: 8,
                  }}
                />
                <Bar dataKey="p" radius={[0, 6, 6, 0]} maxBarSize={28}>
                  {chartData.map((row) => (
                    <Cell
                      key={row.key}
                      fill={BAR_COLORS[row.key] ?? '#6366f1'}
                      opacity={result.class_id === row.classId ? 1 : 0.45}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <h3 className="subsection-title">Label marking</h3>
          <ul className="label-markers" aria-label="Sentiment class marking">
            {LABEL_ORDER.map(({ key, title, classId }) => {
              const active = result.class_id === classId
              const pct = ((result.probabilities?.[key] ?? 0) * 100).toFixed(1)
              return (
                <li
                  key={key}
                  className={`label-marker ${active ? 'label-marker--active' : ''}`}
                >
                  <span className="label-marker-title">{title}</span>
                  <span className="label-marker-id">id {classId}</span>
                  <span className="label-marker-pct">{pct}%</span>
                  {active ? (
                    <span className="label-marker-badge" aria-hidden="true">
                      ✓
                    </span>
                  ) : null}
                </li>
              )
            })}
          </ul>
        </section>
      ) : null}
    </div>
  )
}
