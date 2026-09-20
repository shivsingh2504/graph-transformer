import { useState, useCallback } from 'react'
import { Activity, Play, RefreshCw, Zap } from 'lucide-react'
import GraphCanvas from './GraphCanvas'

const API_BASE = 'http://127.0.0.1:8000/api'

function App() {
  const [graphData, setGraphData] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [predicting, setPredicting] = useState(false)
  const [error, setError] = useState(null)

  const generateGraph = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      setPrediction(null)
      const res = await fetch(`${API_BASE}/generate?num_nodes=15`)
      if (!res.ok) throw new Error('Failed to generate graph')
      const data = await res.json()
      setGraphData(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const predictPath = useCallback(async () => {
    if (!graphData) return
    try {
      setPredicting(true)
      setError(null)
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(graphData)
      })
      if (!res.ok) throw new Error('Prediction failed')
      const data = await res.json()
      setPrediction(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setPredicting(false)
    }
  }, [graphData])

  return (
    <div className="app-container">
      {/* Sidebar Controls */}
      <div className="sidebar glass-panel">
        <div className="logo">
          <Activity size={28} color="var(--accent-primary)" />
          Graphformer
        </div>

        <div className="section">
          <h2>Controls</h2>
          <button 
            className="btn-secondary" 
            onClick={generateGraph}
            disabled={loading || predicting}
          >
            {loading ? <RefreshCw className="spinner" size={18} /> : <RefreshCw size={18} />}
            Generate Random Graph
          </button>
          
          <button 
            className="btn-primary" 
            onClick={predictPath}
            disabled={!graphData || predicting || loading}
          >
            {predicting ? <Zap className="spinner" size={18} /> : <Play size={18} />}
            Predict Shortest Path
          </button>
        </div>

        {error && (
          <div className="section">
            <div className="stat-card" style={{ borderColor: 'var(--error)' }}>
              <span className="stat-label" style={{ color: 'var(--error)' }}>Error</span>
              <span style={{ fontSize: '0.9rem' }}>{error}</span>
            </div>
          </div>
        )}

        {graphData && (
          <div className="section">
            <h2>Graph Details</h2>
            <div className="stat-card">
              <span className="stat-label">Nodes</span>
              <span className="stat-value">{graphData.num_nodes}</span>
            </div>
            <div className="stat-card">
              <span className="stat-label">Edges</span>
              <span className="stat-value">{graphData.edges.length}</span>
            </div>
            <div className="stat-card">
              <span className="stat-label">Source &rarr; Target</span>
              <span className="stat-value" style={{ fontSize: '1rem', color: 'var(--accent-primary)' }}>
                {graphData.source} &rarr; {graphData.target}
              </span>
            </div>
          </div>
        )}

        {prediction && (
          <div className="section">
            <h2>Prediction Results</h2>
            <div className="stat-card">
              <span className="stat-label">Optimal Cost (Dijkstra)</span>
              <span className="stat-value">{prediction.dijkstra_cost}</span>
            </div>
            <div className="stat-card">
              <span className="stat-label">Model Prediction</span>
              <span className={`stat-value ${JSON.stringify(prediction.model_path) === JSON.stringify(prediction.dijkstra_path) ? 'success' : 'error'}`} style={{ fontSize: '1rem' }}>
                {prediction.model_path.join(' → ') || 'No Path'}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Main Graph Canvas */}
      <div className="main-content">
        {!graphData && !loading && (
          <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
            <p>Generate a graph to get started</p>
          </div>
        )}
        
        {graphData && (
          <>
            <div className="overlay-info">
              <div className="overlay-title">Graph Visualization</div>
              <div className="overlay-subtitle">
                Drag nodes to interact. Source: <strong style={{color:'#10b981'}}>{graphData.source}</strong>, Target: <strong style={{color:'#ef4444'}}>{graphData.target}</strong>
              </div>
            </div>
            <GraphCanvas graph={graphData} prediction={prediction} />
          </>
        )}
      </div>
    </div>
  )
}

export default App
