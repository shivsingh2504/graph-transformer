import { useRef, useEffect, useMemo } from 'react'
import ForceGraph2D from 'react-force-graph-2d'

export default function GraphCanvas({ graph, prediction }) {
  const fgRef = useRef()

  const gData = useMemo(() => {
    if (!graph) return { nodes: [], links: [] }

    const nodes = graph.node_ids.map(id => ({
      id,
      name: `Node ${id}`,
      val: 1,
      isSource: id === graph.source,
      isTarget: id === graph.target
    }))

    const links = graph.edges.map(e => ({
      source: e[0],
      target: e[1],
      weight: e[2]
    }))

    return { nodes, links }
  }, [graph])

  // Get the edges present in the predicted path and true path
  const truePathEdges = useMemo(() => {
    if (!prediction?.dijkstra_path) return new Set()
    const p = prediction.dijkstra_path
    const edges = new Set()
    for (let i = 0; i < p.length - 1; i++) {
      const u = Math.min(p[i], p[i+1])
      const v = Math.max(p[i], p[i+1])
      edges.add(`${u}-${v}`)
    }
    return edges
  }, [prediction])

  const predPathEdges = useMemo(() => {
    if (!prediction?.model_path) return new Set()
    const p = prediction.model_path
    const edges = new Set()
    for (let i = 0; i < p.length - 1; i++) {
      const u = Math.min(p[i], p[i+1])
      const v = Math.max(p[i], p[i+1])
      edges.add(`${u}-${v}`)
    }
    return edges
  }, [prediction])

  useEffect(() => {
    if (fgRef.current && gData.nodes.length) {
      fgRef.current.d3Force('charge').strength(-400)
      fgRef.current.d3Force('link').distance(60)
      fgRef.current.zoomToFit(400, 50)
    }
  }, [gData])

  return (
    <ForceGraph2D
      ref={fgRef}
      graphData={gData}
      nodeLabel="name"
      nodeColor={node => {
        if (node.isSource) return '#10b981' // emerald-500
        if (node.isTarget) return '#ef4444' // red-500
        return '#3b82f6' // blue-500
      }}
      nodeRelSize={6}
      linkColor={link => {
        const u = Math.min(link.source.id ?? link.source, link.target.id ?? link.target)
        const v = Math.max(link.source.id ?? link.source, link.target.id ?? link.target)
        const key = `${u}-${v}`

        const inTrue = truePathEdges.has(key)
        const inPred = predPathEdges.has(key)

        if (inTrue && inPred) return '#8b5cf6' // violet-500 (both)
        if (inTrue) return '#10b981' // optimal but model missed
        if (inPred) return '#ef4444' // model guessed wrong edge
        return 'rgba(255,255,255,0.1)' // default edge
      }}
      linkWidth={link => {
        const u = Math.min(link.source.id ?? link.source, link.target.id ?? link.target)
        const v = Math.max(link.source.id ?? link.source, link.target.id ?? link.target)
        const key = `${u}-${v}`
        if (truePathEdges.has(key) || predPathEdges.has(key)) return 4
        return 1
      }}
      linkDirectionalParticles={link => {
        const u = Math.min(link.source.id ?? link.source, link.target.id ?? link.target)
        const v = Math.max(link.source.id ?? link.source, link.target.id ?? link.target)
        const key = `${u}-${v}`
        return predPathEdges.has(key) ? 2 : 0
      }}
      linkDirectionalParticleSpeed={0.01}
      linkDirectionalParticleWidth={4}
      linkDirectionalParticleColor={() => '#a78bfa'}
      backgroundColor="transparent"
      width={window.innerWidth - 320}
      height={window.innerHeight}
      // Render text for node IDs and edge weights
      nodeCanvasObject={(node, ctx, globalScale) => {
        const label = String(node.id)
        const fontSize = 12/globalScale
        ctx.font = `${fontSize}px Inter`
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        
        ctx.beginPath()
        ctx.arc(node.x, node.y, 6, 0, 2 * Math.PI, false)
        ctx.fillStyle = node.isSource ? '#10b981' : (node.isTarget ? '#ef4444' : '#1e293b')
        ctx.fill()
        
        ctx.strokeStyle = node.isSource ? '#34d399' : (node.isTarget ? '#f87171' : '#3b82f6')
        ctx.lineWidth = 1.5/globalScale
        ctx.stroke()

        ctx.fillStyle = '#fff'
        ctx.fillText(label, node.x, node.y)
      }}
      linkCanvasObjectMode={() => 'after'}
      linkCanvasObject={(link, ctx, globalScale) => {
        const MAX_FONT_SIZE = 4
        const LABEL_NODE_MARGIN = 6 * 1.5
        
        const start = link.source
        const end = link.target
        
        if (typeof start !== 'object' || typeof end !== 'object') return
        
        const textPos = Object.assign(...['x', 'y'].map(c => ({
          [c]: start[c] + (end[c] - start[c]) / 2 
        })))
        
        const relLink = { x: end.x - start.x, y: end.y - start.y }
        let textAngle = Math.atan2(relLink.y, relLink.x)
        if (textAngle > Math.PI / 2) textAngle = -(Math.PI - textAngle)
        if (textAngle < -Math.PI / 2) textAngle = -(-Math.PI - textAngle)
        
        const fontSize = 10 / globalScale
        ctx.font = `${fontSize}px Inter`
        
        ctx.save()
        ctx.translate(textPos.x, textPos.y)
        ctx.rotate(textAngle)
        
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)'
        ctx.fillText(link.weight, 0, -3/globalScale)
        ctx.restore()
      }}
    />
  )
}
