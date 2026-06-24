/**
 * useCanvas / useZoomableCanvas — ResizeObserver-based canvas hooks
 *
 * Key fix: canvas.width/height attributes start at 0 until explicitly set.
 * getBoundingClientRect() returns real CSS dimensions immediately after mount.
 */
import { useRef, useEffect, useCallback } from 'react'

export function useCanvas(
  draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void,
  deps: unknown[]
) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const render = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    // Use getBoundingClientRect — always the real CSS layout size
    const rect = canvas.getBoundingClientRect()
    const cssW = rect.width
    const cssH = rect.height
    if (cssW <= 0 || cssH <= 0) return

    const dpr  = window.devicePixelRatio || 1
    const bufW = Math.round(cssW * dpr)
    const bufH = Math.round(cssH * dpr)

    // Resize backing buffer only when dimensions change
    if (canvas.width !== bufW || canvas.height !== bufH) {
      canvas.width  = bufW
      canvas.height = bufH
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Reset to clean DPR-scaled transform, then draw in CSS-pixel space
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, cssW, cssH)
    draw(ctx, cssW, cssH)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ro = new ResizeObserver(() => render())
    ro.observe(canvas)
    render() // immediate first draw after mount

    return () => ro.disconnect()
  }, [render])

  return canvasRef
}

// ── Zoom / pan canvas hook ────────────────────────────────────────

interface Viewport { zoom: number; panX: number; panY: number }
const DEFAULT_VP: Viewport = { zoom: 1, panX: 0, panY: 0 }

/**
 * useZoomableCanvas — same draw API as useCanvas but adds:
 * - Mouse-wheel zoom (zooms toward cursor)
 * - Left-drag to pan
 * - Double-click to reset view
 *
 * The viewport transform is applied automatically before calling `draw`,
 * so draw callbacks work in the same coordinate space as before.
 */
export function useZoomableCanvas(
  draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void,
  deps: unknown[]
) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const vpRef     = useRef<Viewport>({ ...DEFAULT_VP })
  const renderRef = useRef<() => void>(() => { /* noop until mounted */ })

  const render = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const cssW = rect.width, cssH = rect.height
    if (cssW <= 0 || cssH <= 0) return

    const dpr  = window.devicePixelRatio || 1
    const bufW = Math.round(cssW * dpr)
    const bufH = Math.round(cssH * dpr)
    if (canvas.width !== bufW || canvas.height !== bufH) {
      canvas.width = bufW; canvas.height = bufH
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, cssW, cssH)

    // Apply viewport transform, then draw in unzoomed coordinate space
    const { zoom, panX, panY } = vpRef.current
    ctx.save()
    ctx.translate(cssW / 2 + panX, cssH / 2 + panY)
    ctx.scale(zoom, zoom)
    ctx.translate(-cssW / 2, -cssH / 2)
    draw(ctx, cssW, cssH)
    ctx.restore()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  useEffect(() => { renderRef.current = render }, [render])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ro = new ResizeObserver(() => renderRef.current())
    ro.observe(canvas)
    render()
    return () => ro.disconnect()
  }, [render])

  // Zoom/pan handlers — attached once per canvas mount, use renderRef
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = canvas.getBoundingClientRect()
      const mx = e.clientX - rect.left - rect.width  / 2
      const my = e.clientY - rect.top  - rect.height / 2
      const vp = vpRef.current
      const factor = e.deltaY > 0 ? 0.88 : 1.14
      const newZoom = Math.max(0.3, Math.min(30, vp.zoom * factor))
      vpRef.current = {
        zoom: newZoom,
        panX: mx - (mx - vp.panX) * (newZoom / vp.zoom),
        panY: my - (my - vp.panY) * (newZoom / vp.zoom),
      }
      renderRef.current()
    }

    let dragging = false, lastX = 0, lastY = 0
    const onMouseDown = (e: MouseEvent) => {
      if (e.button !== 0) return
      dragging = true; lastX = e.clientX; lastY = e.clientY
      canvas.style.cursor = 'grabbing'
    }
    const onMouseMove = (e: MouseEvent) => {
      if (!dragging) return
      const dx = e.clientX - lastX, dy = e.clientY - lastY
      lastX = e.clientX; lastY = e.clientY
      vpRef.current = { ...vpRef.current, panX: vpRef.current.panX + dx, panY: vpRef.current.panY + dy }
      renderRef.current()
    }
    const onMouseUp = () => { dragging = false; canvas.style.cursor = 'grab' }
    const onDblClick = () => { vpRef.current = { ...DEFAULT_VP }; renderRef.current() }

    canvas.addEventListener('wheel', onWheel, { passive: false })
    canvas.addEventListener('mousedown', onMouseDown)
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    canvas.addEventListener('dblclick', onDblClick)
    canvas.style.cursor = 'grab'

    return () => {
      canvas.removeEventListener('wheel', onWheel)
      canvas.removeEventListener('mousedown', onMouseDown)
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
      canvas.removeEventListener('dblclick', onDblClick)
      canvas.style.cursor = ''
    }
  }, [])  // attach once per canvas mount

  return canvasRef
}

/** Map geographic [lonMin,lonMax]×[latMin,latMax] → canvas CSS-pixel space */
export function makeProjection(
  w: number, h: number,
  lonMin: number, lonMax: number,
  latMin: number, latMax: number,
  margin = 36
) {
  const innerW = w - margin * 2
  const innerH = h - margin * 2
  const scaleX = innerW / (lonMax - lonMin)
  const scaleY = innerH / (latMax - latMin)
  const scale  = Math.min(scaleX, scaleY)
  const offX   = margin + (innerW - scale * (lonMax - lonMin)) / 2
  const offY   = margin + (innerH - scale * (latMax - latMin)) / 2
  return {
    px: (lon: number) => offX + (lon - lonMin) * scale,
    py: (lat: number) => offY + (latMax - lat) * scale,
    scale,
  }
}

/** Diverging colormap: blue → white → red */
export function divergingColor(value: number, vmin: number, vmax: number, alpha = 0.85): string {
  const t = Math.max(0, Math.min(1, (value - vmin) / (vmax - vmin)))
  if (t < 0.5) {
    const s = t * 2
    const r = Math.round(37  + s * (255 - 37))
    const g = Math.round(99  + s * (255 - 99))
    const b = Math.round(235 + s * (255 - 235))
    return `rgba(${r},${g},${b},${alpha})`
  } else {
    const s = (t - 0.5) * 2
    const r = 255
    const g = Math.round(255 - s * (255 - 45))
    const b = Math.round(255 - s * (255 - 45))
    return `rgba(${r},${g},${b},${alpha})`
  }
}

/** Sequential colormap: teal → orange → red */
export function sequentialColor(value: number, vmin: number, vmax: number, alpha = 0.85): string {
  const t = Math.max(0, Math.min(1, (value - vmin) / (vmax - vmin)))
  const r = Math.round(20  + t * 220)
  const g = Math.round(200 - t * 160)
  const b = Math.round(220 - t * 200)
  return `rgba(${r},${g},${b},${alpha})`
}

/** Draw lat/lon grid lines + axis labels */
export function drawGrid(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  proj: ReturnType<typeof makeProjection>,
  lonMin: number, lonMax: number,
  latMin: number, latMax: number
) {
  ctx.save()
  ctx.strokeStyle = 'rgba(100,116,139,0.18)'
  ctx.lineWidth   = 0.5

  for (let lon = Math.ceil(lonMin); lon <= lonMax; lon++) {
    const x = proj.px(lon)
    ctx.beginPath(); ctx.moveTo(x, proj.py(latMin)); ctx.lineTo(x, proj.py(latMax)); ctx.stroke()
  }
  for (let lat = Math.ceil(latMin); lat <= latMax; lat++) {
    const y = proj.py(lat)
    ctx.beginPath(); ctx.moveTo(proj.px(lonMin), y); ctx.lineTo(proj.px(lonMax), y); ctx.stroke()
  }

  ctx.fillStyle = 'rgba(148,163,184,0.75)'
  ctx.font      = '9px JetBrains Mono, monospace'
  ctx.textAlign = 'center'
  for (let lon = Math.ceil(lonMin); lon <= lonMax; lon += 2)
    ctx.fillText(`${lon}°E`, proj.px(lon), h - 6)
  ctx.textAlign = 'right'
  for (let lat = Math.ceil(latMin); lat <= latMax; lat += 2)
    ctx.fillText(`${lat}°N`, proj.px(lonMin) - 4, proj.py(lat) + 3)

  ctx.restore()
}

/** Draw a horizontal colorbar */
export function drawColorbar(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, bw: number, bh: number,
  vmin: number, vmax: number,
  label: string,
  colorFn: (v: number, mn: number, mx: number) => string
) {
  ctx.save()
  for (let i = 0; i < bw; i++) {
    const v = vmin + (i / bw) * (vmax - vmin)
    ctx.fillStyle = colorFn(v, vmin, vmax)
    ctx.fillRect(x + i, y, 1, bh)
  }
  ctx.strokeStyle = 'rgba(100,116,139,0.4)'
  ctx.lineWidth   = 0.5
  ctx.strokeRect(x, y, bw, bh)
  ctx.fillStyle = 'rgba(148,163,184,0.9)'
  ctx.font      = '9px JetBrains Mono, monospace'
  ctx.textAlign = 'left';   ctx.fillText(vmin.toFixed(0), x, y + bh + 10)
  ctx.textAlign = 'right';  ctx.fillText(vmax.toFixed(0), x + bw, y + bh + 10)
  ctx.textAlign = 'center'; ctx.fillText(label, x + bw / 2, y - 4)
  ctx.restore()
}
