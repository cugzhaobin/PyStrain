import React from 'react'
import type { GnssSite, DataBounds } from '../types'
import {
  useZoomableCanvas, makeProjection,
  drawGrid, drawColorbar, divergingColor,
} from '../canvasUtils'

// ── Velocity views ────────────────────────────────────────────────

interface VelocityProps {
  sites: GnssSite[]
  outliers: GnssSite[]
  showOutliers?: boolean
  velocityScale?: number
  bounds: DataBounds
}

export function RawVelocityView({ sites, outliers, showOutliers = true, velocityScale, bounds }: VelocityProps) {
  const { lonMin, lonMax, latMin, latMax } = bounds

  const canvasRef = useZoomableCanvas((ctx, w, h) => {
    const proj = makeProjection(w, h, lonMin, lonMax, latMin, latMax, 36)
    drawGrid(ctx, w, h, proj, lonMin, lonMax, latMin, latMax)

    const all = showOutliers ? [...sites, ...outliers] : sites
    if (all.length === 0) return

    // Auto-scale so the median arrow ≈ 15 px
    let scale: number
    if (velocityScale != null) {
      scale = velocityScale
    } else {
      const speeds = all.map(s => Math.hypot(s.ve, s.vn)).sort((a, b) => a - b)
      const median = speeds[Math.floor(speeds.length / 2)] || 1
      scale = 15 / median
    }

    // Draw velocity arrows
    all.forEach(s => {
      const x = proj.px(s.lon)
      const y = proj.py(s.lat)
      const isOut = s.isOutlier
      const dx = s.ve * scale
      const dy = -s.vn * scale

      const len = Math.hypot(dx, dy)
      if (len < 0.5) return

      ctx.save()
      ctx.strokeStyle = isOut ? 'rgba(239,68,68,0.9)' : 'rgba(56,189,248,0.85)'
      ctx.fillStyle   = isOut ? 'rgba(239,68,68,0.9)' : 'rgba(56,189,248,0.85)'
      ctx.lineWidth   = isOut ? 1.4 : 1.1

      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + dx, y + dy); ctx.stroke()

      const angle = Math.atan2(dy, dx)
      const hs = Math.min(len * 0.38, 7)
      ctx.beginPath()
      ctx.moveTo(x + dx, y + dy)
      ctx.lineTo(x + dx - hs * Math.cos(angle - 0.42), y + dy - hs * Math.sin(angle - 0.42))
      ctx.lineTo(x + dx - hs * Math.cos(angle + 0.42), y + dy - hs * Math.sin(angle + 0.42))
      ctx.closePath(); ctx.fill()
      ctx.restore()
    })

    // Scale bar
    const speeds = all.map(s => Math.hypot(s.ve, s.vn))
    const maxSpd = Math.max(...speeds)
    const refMm = maxSpd >= 20 ? 20 : maxSpd >= 10 ? 10 : maxSpd >= 5 ? 5 : 1
    const sbLen = refMm * scale
    const sx = 52, sy = h - 22
    ctx.save()
    ctx.strokeStyle = 'rgba(148,163,184,0.85)'; ctx.lineWidth = 1.5
    ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx + sbLen, sy); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(sx, sy - 3); ctx.lineTo(sx, sy + 3); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(sx + sbLen, sy - 3); ctx.lineTo(sx + sbLen, sy + 3); ctx.stroke()
    ctx.fillStyle = 'rgba(148,163,184,0.85)'; ctx.font = '9px JetBrains Mono, monospace'
    ctx.textAlign = 'left'; ctx.fillText(`${refMm} mm/yr`, sx, sy - 7)
    ctx.restore()

    // Legend
    const lx = w - 14, ly = 16
    ctx.save()
    ctx.fillStyle = 'rgba(56,189,248,0.85)'; ctx.fillRect(lx - 65, ly, 9, 9)
    ctx.fillStyle = 'rgba(148,163,184,0.8)'; ctx.font = '9px Inter, sans-serif'
    ctx.textAlign = 'left'; ctx.fillText('Used', lx - 53, ly + 8)
    if (showOutliers) {
      ctx.fillStyle = 'rgba(239,68,68,0.85)'; ctx.fillRect(lx - 65, ly + 14, 9, 9)
      ctx.fillStyle = 'rgba(148,163,184,0.8)'; ctx.fillText('Outlier', lx - 53, ly + 22)
    }
    ctx.restore()

    void drawColorbar // not used here
  }, [sites, outliers, showOutliers, velocityScale, bounds])

  return (
    <canvas ref={canvasRef} className="w-full h-full" style={{ display: 'block' }} />
  )
}

// ── Outlier view ──────────────────────────────────────────────────

interface OutlierProps { sites: GnssSite[]; outliers: GnssSite[]; bounds: DataBounds }

export function OutlierView({ sites, outliers, bounds }: OutlierProps) {
  const { lonMin, lonMax, latMin, latMax } = bounds

  const canvasRef = useZoomableCanvas((ctx, w, h) => {
    const proj = makeProjection(w, h, lonMin, lonMax, latMin, latMax, 36)
    drawGrid(ctx, w, h, proj, lonMin, lonMax, latMin, latMax)

    sites.forEach(s => {
      const x = proj.px(s.lon), y = proj.py(s.lat)
      ctx.beginPath(); ctx.arc(x, y, 2, 0, Math.PI * 2)
      ctx.fillStyle = 'rgba(100,116,139,0.5)'; ctx.fill()
    })

    outliers.forEach(s => {
      const x = proj.px(s.lon), y = proj.py(s.lat)
      ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2)
      ctx.fillStyle = 'rgba(239,68,68,0.12)'; ctx.fill()
      ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2)
      ctx.fillStyle = 'rgba(239,68,68,0.9)'; ctx.fill()
      ctx.strokeStyle = 'rgba(252,165,165,0.6)'; ctx.lineWidth = 0.8; ctx.stroke()
    })

    ctx.fillStyle = 'rgba(239,68,68,0.85)'
    ctx.font = 'bold 11px Inter, sans-serif'
    ctx.textAlign = 'right'
    ctx.fillText(`${outliers.length} outliers detected`, w - 14, 20)
  }, [sites, outliers, bounds])

  return <canvas ref={canvasRef} className="w-full h-full" style={{ display: 'block' }} />
}

// ── Triangulation view ────────────────────────────────────────────

interface TriProps {
  sites: GnssSite[]
  triangles: import('../types').StrainTriangle[]
  bounds: DataBounds
}

export function TriangulationView({ sites, triangles, bounds }: TriProps) {
  const { lonMin, lonMax, latMin, latMax } = bounds

  const canvasRef = useZoomableCanvas((ctx, w, h) => {
    const proj = makeProjection(w, h, lonMin, lonMax, latMin, latMax, 36)
    drawGrid(ctx, w, h, proj, lonMin, lonMax, latMin, latMax)

    triangles.forEach(t => {
      ctx.beginPath()
      ctx.moveTo(proj.px(t.lon0), proj.py(t.lat0))
      ctx.lineTo(proj.px(t.lon1), proj.py(t.lat1))
      ctx.lineTo(proj.px(t.lon2), proj.py(t.lat2))
      ctx.closePath()
      ctx.fillStyle = t.isGood ? 'rgba(56,189,248,0.04)' : 'rgba(239,68,68,0.06)'
      ctx.fill()
      ctx.strokeStyle = t.isGood ? 'rgba(56,189,248,0.3)' : 'rgba(239,68,68,0.3)'
      ctx.lineWidth = 0.5; ctx.stroke()
    })

    sites.forEach(s => {
      const x = proj.px(s.lon), y = proj.py(s.lat)
      ctx.beginPath(); ctx.arc(x, y, 2, 0, Math.PI * 2)
      ctx.fillStyle = 'rgba(250,204,21,0.8)'; ctx.fill()
    })

    const good = triangles.filter(t => t.isGood).length
    ctx.fillStyle = 'rgba(56,189,248,0.8)'; ctx.font = '10px Inter, sans-serif'
    ctx.textAlign = 'right'
    ctx.fillText(`${good} valid triangles  |  ${sites.length} sites`, w - 14, 20)
  }, [sites, triangles, bounds])

  return <canvas ref={canvasRef} className="w-full h-full" style={{ display: 'block' }} />
}

// ── Strain rate views ─────────────────────────────────────────────

interface StrainProps {
  triangles: import('../types').StrainTriangle[]
  bounds: DataBounds
}

export function DilatationView({ triangles, bounds }: StrainProps) {
  const { lonMin, lonMax, latMin, latMax } = bounds
  const good = triangles.filter(t => t.isGood)
  const vals = good.map(t => t.dilatation)
  const vmax = vals.length ? Math.max(Math.abs(Math.min(...vals)), Math.abs(Math.max(...vals))) : 1
  const vmin = -vmax

  const canvasRef = useZoomableCanvas((ctx, w, h) => {
    const proj = makeProjection(w, h, lonMin, lonMax, latMin, latMax, 36)
    drawGrid(ctx, w, h, proj, lonMin, lonMax, latMin, latMax)

    good.forEach(t => {
      ctx.beginPath()
      ctx.moveTo(proj.px(t.lon0), proj.py(t.lat0))
      ctx.lineTo(proj.px(t.lon1), proj.py(t.lat1))
      ctx.lineTo(proj.px(t.lon2), proj.py(t.lat2))
      ctx.closePath()
      ctx.fillStyle = divergingColor(t.dilatation, vmin, vmax, 0.82)
      ctx.fill()
      ctx.strokeStyle = 'rgba(30,41,59,0.4)'; ctx.lineWidth = 0.3; ctx.stroke()
    })

    drawColorbar(ctx, 52, h - 28, 160, 10, vmin, vmax, 'Dilatation (nstrain/yr)', divergingColor)
  }, [good, vmin, vmax, bounds])

  return <canvas ref={canvasRef} className="w-full h-full" style={{ display: 'block' }} />
}

export function MaxShearView({ triangles, bounds }: StrainProps) {
  const { lonMin, lonMax, latMin, latMax } = bounds
  const good = triangles.filter(t => t.isGood)
  const vals = good.map(t => t.maxShear)
  const vmin = 0, vmax = vals.length ? Math.max(...vals) : 1

  const canvasRef = useZoomableCanvas((ctx, w, h) => {
    const proj = makeProjection(w, h, lonMin, lonMax, latMin, latMax, 36)
    drawGrid(ctx, w, h, proj, lonMin, lonMax, latMin, latMax)

    good.forEach(t => {
      ctx.beginPath()
      ctx.moveTo(proj.px(t.lon0), proj.py(t.lat0))
      ctx.lineTo(proj.px(t.lon1), proj.py(t.lat1))
      ctx.lineTo(proj.px(t.lon2), proj.py(t.lat2))
      ctx.closePath()
      const tv = vmax > 0 ? Math.max(0, Math.min(1, t.maxShear / vmax)) : 0
      const r = Math.round(20  + tv * 200)
      const g = Math.round(180 - tv * 155)
      const b = Math.round(200 - tv * 190)
      ctx.fillStyle = `rgba(${r},${g},${b},0.82)`
      ctx.fill()
      ctx.strokeStyle = 'rgba(30,41,59,0.4)'; ctx.lineWidth = 0.3; ctx.stroke()
    })

    // Custom colorbar
    const cbX = 52, cbY = h - 28, cbW = 160, cbH = 10
    for (let i = 0; i < cbW; i++) {
      const tv = i / cbW
      const r = Math.round(20 + tv * 200)
      const g = Math.round(180 - tv * 155)
      const b = Math.round(200 - tv * 190)
      ctx.fillStyle = `rgba(${r},${g},${b},0.82)`
      ctx.fillRect(cbX + i, cbY, 1, cbH)
    }
    ctx.strokeStyle = 'rgba(100,116,139,0.4)'; ctx.lineWidth = 0.5; ctx.strokeRect(cbX, cbY, cbW, cbH)
    ctx.fillStyle = 'rgba(148,163,184,0.9)'; ctx.font = '9px JetBrains Mono, monospace'
    ctx.textAlign = 'left';   ctx.fillText('0', cbX, cbY + cbH + 10)
    ctx.textAlign = 'right';  ctx.fillText(vmax.toFixed(0), cbX + cbW, cbY + cbH + 10)
    ctx.textAlign = 'center'; ctx.fillText('Max Shear (nstrain/yr)', cbX + cbW / 2, cbY - 4)
  }, [good, vmax, bounds])

  return <canvas ref={canvasRef} className="w-full h-full" style={{ display: 'block' }} />
}

export function PrincipalStrainView({ triangles, bounds }: StrainProps) {
  const { lonMin, lonMax, latMin, latMax } = bounds
  const good = triangles.filter(t => t.isGood)
  const maxAbs = good.length
    ? Math.max(...good.flatMap(t => [Math.abs(t.e1), Math.abs(t.e2)]))
    : 1

  const canvasRef = useZoomableCanvas((ctx, w, h) => {
    const proj = makeProjection(w, h, lonMin, lonMax, latMin, latMax, 36)
    drawGrid(ctx, w, h, proj, lonMin, lonMax, latMin, latMax)

    // Triangle outlines
    good.forEach(t => {
      ctx.beginPath()
      ctx.moveTo(proj.px(t.lon0), proj.py(t.lat0))
      ctx.lineTo(proj.px(t.lon1), proj.py(t.lat1))
      ctx.lineTo(proj.px(t.lon2), proj.py(t.lat2))
      ctx.closePath()
      ctx.strokeStyle = 'rgba(100,116,139,0.2)'; ctx.lineWidth = 0.3; ctx.stroke()
    })

    // Principal strain crosses
    // azimuth = angle of e2 (extension axis) from East, CCW, degrees
    const SCALE = 0.25 / (maxAbs || 1)
    good.forEach(t => {
      const cx = proj.px(t.clon), cy = proj.py(t.clat)
      const az  = t.azimuth * Math.PI / 180
      // e2 direction in screen space: East→right (cos az), North→up (sin az) → screen y flipped
      const e2x = Math.cos(az),  e2y = -Math.sin(az)
      // e1 direction: perpendicular to e2
      const e1x = Math.sin(az),  e1y =  Math.cos(az)

      const drawArm = (e: number, ux: number, uy: number) => {
        const l = Math.abs(e) * SCALE * 28
        const c = e < 0 ? 'rgba(59,130,246,0.85)' : 'rgba(239,68,68,0.85)'
        ctx.strokeStyle = c; ctx.fillStyle = c; ctx.lineWidth = 1.3
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + ux * l, cy + uy * l); ctx.stroke()
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx - ux * l, cy - uy * l); ctx.stroke()
        ctx.beginPath(); ctx.arc(cx + ux * l, cy + uy * l, 1.5, 0, Math.PI * 2); ctx.fill()
        ctx.beginPath(); ctx.arc(cx - ux * l, cy - uy * l, 1.5, 0, Math.PI * 2); ctx.fill()
      }

      drawArm(t.e2, e2x, e2y)   // extension arm along azimuth direction
      drawArm(t.e1, e1x, e1y)   // compression arm perpendicular
    })

    // Legend
    const lx = w - 14, ly = 14
    ctx.save()
    ctx.strokeStyle = 'rgba(59,130,246,0.85)'; ctx.lineWidth = 1.5
    ctx.beginPath(); ctx.moveTo(lx - 50, ly + 4); ctx.lineTo(lx - 38, ly + 4); ctx.stroke()
    ctx.fillStyle = 'rgba(148,163,184,0.7)'; ctx.font = '9px Inter, sans-serif'
    ctx.textAlign = 'left'; ctx.fillText('Compression', lx - 36, ly + 7)
    ctx.strokeStyle = 'rgba(239,68,68,0.85)'
    ctx.beginPath(); ctx.moveTo(lx - 50, ly + 18); ctx.lineTo(lx - 38, ly + 18); ctx.stroke()
    ctx.fillText('Extension', lx - 36, ly + 21)
    ctx.restore()
  }, [good, maxAbs, bounds])

  return <canvas ref={canvasRef} className="w-full h-full" style={{ display: 'block' }} />
}
