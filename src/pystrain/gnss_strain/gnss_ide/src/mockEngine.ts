/**
 * GNSS Strain Rate computation engine.
 * - Real file parsers (GMT 8-column, GLOBK 13-column)
 * - Real Delaunay triangulation (delaunator) with analytic strain tensor
 * - Mock data generator for UI demonstration when no file is loaded
 */

import type { GnssSite, StrainTriangle, ComputeResult, ComputeParams, DataBounds } from './types'
import Delaunator from 'delaunator'

// ── Seeded RNG (mock data only) ──────────────────────────────────
function seededRng(seed: number) {
  let s = seed
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff
    return (s >>> 0) / 0xffffffff
  }
}

// ── File parsers ─────────────────────────────────────────────────

/**
 * Parse GMT 8-column velocity file.
 * Columns: lon lat ve vn se sn corr name
 * Comment lines start with #, >, or *
 */
export function parseGmtFile(text: string): GnssSite[] {
  const sites: GnssSite[] = []
  let idx = 0
  for (const raw of text.split('\n')) {
    const line = raw.trim()
    if (!line || line.startsWith('#') || line.startsWith('>') || line.startsWith('*')) continue
    const cols = line.split(/\s+/)
    if (cols.length < 6) continue
    const lon = parseFloat(cols[0])
    const lat = parseFloat(cols[1])
    const ve  = parseFloat(cols[2])
    const vn  = parseFloat(cols[3])
    const se  = parseFloat(cols[4])
    const sn  = parseFloat(cols[5])
    if ([lon, lat, ve, vn, se, sn].some(isNaN)) continue
    const name = cols[7] ?? `PT${String(idx).padStart(4, '0')}`
    sites.push({ lon, lat, ve, vn, se, sn, name })
    idx++
  }
  return sites
}

/**
 * Parse GLOBK velocity file.
 *
 * camp_eura.vel format (13+ cols, comment lines start with *):
 *   lon  lat  ve  vn  ve_adj  vn_adj  se  sn  rho  vh  vh_adj  sh  name
 *   col0 col1 col2 col3 col4   col5   col6 col7 ...              last
 *
 * Standard GLOBK .vel format (name-first, 7+ cols):
 *   name  lon  lat  ve  vn  se  sn  ...
 */
export function parseGlobkFile(text: string): GnssSite[] {
  const sites: GnssSite[] = []

  let firstData = ''
  for (const raw of text.split('\n')) {
    const t = raw.trim()
    if (t && !t.startsWith('#') && !t.startsWith('*') && !t.startsWith('>')) {
      firstData = t; break
    }
  }
  const firstCols = firstData.split(/\s+/)
  const lonFirst = !isNaN(parseFloat(firstCols[0]))

  for (const raw of text.split('\n')) {
    const line = raw.trim()
    if (!line || line.startsWith('#') || line.startsWith('*') || line.startsWith('>')) continue
    const cols = line.split(/\s+/)

    let lon: number, lat: number, ve: number, vn: number, se: number, sn: number, name: string

    if (lonFirst) {
      // camp_eura format: lon lat ve vn ve_adj vn_adj se sn rho ... name
      if (cols.length < 8) continue
      lon  = parseFloat(cols[0])
      lat  = parseFloat(cols[1])
      ve   = parseFloat(cols[2])
      vn   = parseFloat(cols[3])
      se   = parseFloat(cols[6])
      sn   = parseFloat(cols[7])
      name = cols[cols.length - 1]
    } else {
      // Standard GLOBK: name lon lat ve vn se sn ...
      if (cols.length < 7) continue
      name = cols[0]
      lon  = parseFloat(cols[1])
      lat  = parseFloat(cols[2])
      ve   = parseFloat(cols[3])
      vn   = parseFloat(cols[4])
      se   = parseFloat(cols[5])
      sn   = parseFloat(cols[6])
    }

    if ([lon, lat, ve, vn, se, sn].some(isNaN)) continue
    sites.push({ lon, lat, ve, vn, se, sn, name })
  }
  return sites
}

/**
 * Auto-detect format and parse a velocity file.
 */
export function parseVelocityFile(text: string, format: ComputeParams['velFormat']): GnssSite[] {
  if (format === 'globk') return parseGlobkFile(text)
  if (format === 'gmt')   return parseGmtFile(text)
  const firstData = text.split('\n').find(l => {
    const t = l.trim()
    return t && !t.startsWith('#') && !t.startsWith('*') && !t.startsWith('>')
  }) ?? ''
  const cols = firstData.trim().split(/\s+/)
  const col0IsNum = !isNaN(parseFloat(cols[0]))
  if (!col0IsNum)        return parseGlobkFile(text)   // name-first GLOBK
  if (cols.length >= 12) return parseGlobkFile(text)   // lon-first GLOBK (13-col)
  return parseGmtFile(text)                             // GMT 8-col
}

// ── Math helpers ─────────────────────────────────────────────────

/** Haversine distance in km */
function distKm(lon1: number, lat1: number, lon2: number, lat2: number): number {
  const R = 6371
  const dLat = (lat2 - lat1) * Math.PI / 180
  const dLon = (lon2 - lon1) * Math.PI / 180
  const a = Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLon / 2) ** 2
  return 2 * R * Math.asin(Math.sqrt(Math.min(1, a)))
}

/** Sorted median */
function median(arr: number[]): number {
  const s = [...arr].sort((a, b) => a - b)
  const m = Math.floor(s.length / 2)
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}

/** Normalized MAD scale estimate (σ ≈ MAD × 1.4826) */
function madScale(arr: number[]): number {
  const med = median(arr)
  return median(arr.map(v => Math.abs(v - med))) * 1.4826
}

// ── Data processing ───────────────────────────────────────────────

/** Compute geographic bounds with optional padding */
function computeBounds(sites: GnssSite[], padPct = 0.08): DataBounds {
  const lons = sites.map(s => s.lon)
  const lats = sites.map(s => s.lat)
  const lonSpan = Math.max(...lons) - Math.min(...lons)
  const latSpan = Math.max(...lats) - Math.min(...lats)
  const pad = Math.max(padPct, 0.02)
  return {
    lonMin: Math.min(...lons) - lonSpan * pad,
    lonMax: Math.max(...lons) + lonSpan * pad,
    latMin: Math.min(...lats) - latSpan * pad,
    latMax: Math.max(...lats) + latSpan * pad,
  }
}

/**
 * MAD-based outlier detection on velocity residuals.
 * Uses global median as reference (appropriate after plate-rotation removal).
 */
function detectOutliers(sites: GnssSite[], madFactor: number): GnssSite[] {
  if (sites.length < 5) return sites
  const veArr = sites.map(s => s.ve)
  const vnArr = sites.map(s => s.vn)
  const medVe = median(veArr), sigVe = madScale(veArr)
  const medVn = median(vnArr), sigVn = madScale(vnArr)
  return sites.map(s => ({
    ...s,
    isOutlier: (
      (sigVe > 0.01 && Math.abs(s.ve - medVe) > madFactor * sigVe) ||
      (sigVn > 0.01 && Math.abs(s.vn - medVn) > madFactor * sigVn)
    ),
  }))
}

/** Spatial thinning: keep one site per minSpacingKm cell */
function thinSites(sites: GnssSite[], minSpacingKm: number): GnssSite[] {
  const kept: GnssSite[] = []
  for (const s of sites) {
    if (!kept.some(k => distKm(s.lon, s.lat, k.lon, k.lat) < minSpacingKm))
      kept.push(s)
  }
  return kept
}

// ── Analytic strain tensor ────────────────────────────────────────

interface StrainData {
  clon: number; clat: number
  dilatation: number; maxShear: number
  e1: number; e2: number; azimuth: number
}

/**
 * Compute analytic strain tensor for a triangle (constant-strain element).
 *
 * Converts triangle to local Cartesian (km) centred on centroid; applies
 * finite-element shape-function derivatives to compute velocity gradient tensor.
 * Units: input ve/vn in mm/yr, output in nstrain/yr.
 *
 *   exx = ∂ve/∂x,  eyy = ∂vn/∂y,  exy = (∂ve/∂y + ∂vn/∂x) / 2
 *   dilatation = exx + eyy
 *   shear      = sqrt(((exx-eyy)/2)² + exy²)
 *   maxShear   = 2 × shear
 *   e1 = mean − shear   (most compressive)
 *   e2 = mean + shear   (most extensive)
 *   azimuth = 0.5 × atan2(2×exy, exx−eyy)  [angle of e2 from East, CCW, degrees]
 */
function strainFromTriangle(s0: GnssSite, s1: GnssSite, s2: GnssSite): StrainData | null {
  const clon = (s0.lon + s1.lon + s2.lon) / 3
  const clat = (s0.lat + s1.lat + s2.lat) / 3
  const cosLat = Math.cos(clat * Math.PI / 180)
  const KPD = 111.32  // km per degree of latitude

  // Local Cartesian (km), centred on centroid
  const x = [
    (s0.lon - clon) * cosLat * KPD,
    (s1.lon - clon) * cosLat * KPD,
    (s2.lon - clon) * cosLat * KPD,
  ]
  const y = [
    (s0.lat - clat) * KPD,
    (s1.lat - clat) * KPD,
    (s2.lat - clat) * KPD,
  ]

  // Twice the signed area (km²)
  const A2 = x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1])
  if (Math.abs(A2) < 1e-6) return null  // degenerate

  // Velocity gradient: (mm/yr) / km = µstrain/yr; × 1000 → nstrain/yr
  const S = 1000 / A2
  const ve = [s0.ve, s1.ve, s2.ve]
  const vn = [s0.vn, s1.vn, s2.vn]

  const dvedx = S * (ve[0] * (y[1] - y[2]) + ve[1] * (y[2] - y[0]) + ve[2] * (y[0] - y[1]))
  const dvedy = S * (ve[0] * (x[2] - x[1]) + ve[1] * (x[0] - x[2]) + ve[2] * (x[1] - x[0]))
  const dvndx = S * (vn[0] * (y[1] - y[2]) + vn[1] * (y[2] - y[0]) + vn[2] * (y[0] - y[1]))
  const dvndy = S * (vn[0] * (x[2] - x[1]) + vn[1] * (x[0] - x[2]) + vn[2] * (x[1] - x[0]))

  const exx = dvedx
  const eyy = dvndy
  const exy = 0.5 * (dvedy + dvndx)

  const dilatation = exx + eyy
  const mean       = (exx + eyy) / 2
  const shear      = Math.sqrt(((exx - eyy) / 2) ** 2 + exy ** 2)
  const e1         = mean - shear   // most compressive
  const e2         = mean + shear   // most extensive
  const azimuth    = 0.5 * Math.atan2(2 * exy, exx - eyy) * 180 / Math.PI

  return { clon, clat, dilatation, maxShear: 2 * shear, e1, e2, azimuth }
}

// ── Delaunay triangulation + quality filter ───────────────────────

function triangulateAndFilter(
  sites: GnssSite[],
  params: ComputeParams,
  log: string[]
): StrainTriangle[] {
  if (sites.length < 3) {
    log.push('    Too few sites for triangulation.')
    return []
  }

  // Build Delaunay triangulation
  const del = Delaunator.from(sites, s => s.lon, s => s.lat)
  const idx = del.triangles

  // Compute all triangles and their max edge length
  const raw: Array<{ tri: StrainTriangle; maxEdgeKm: number; minAngle: number }> = []

  for (let i = 0; i < idx.length; i += 3) {
    const [i0, i1, i2] = [idx[i], idx[i + 1], idx[i + 2]]
    const [s0, s1, s2] = [sites[i0], sites[i1], sites[i2]]

    const d01 = distKm(s0.lon, s0.lat, s1.lon, s1.lat)
    const d12 = distKm(s1.lon, s1.lat, s2.lon, s2.lat)
    const d20 = distKm(s2.lon, s2.lat, s0.lon, s0.lat)
    const maxEdgeKm = Math.max(d01, d12, d20)

    // Min angle via law of cosines (cos is decreasing, so min angle ↔ max cosine)
    const cosA = (d01 ** 2 + d20 ** 2 - d12 ** 2) / (2 * d01 * d20)
    const cosB = (d01 ** 2 + d12 ** 2 - d20 ** 2) / (2 * d01 * d12)
    const cosC = (d12 ** 2 + d20 ** 2 - d01 ** 2) / (2 * d12 * d20)
    const maxCos = Math.max(cosA, cosB, cosC)
    const minAngle = Math.acos(Math.max(-1, Math.min(1, maxCos))) * 180 / Math.PI

    const strain = strainFromTriangle(s0, s1, s2)
    if (!strain) continue

    raw.push({
      tri: {
        lon0: s0.lon, lat0: s0.lat,
        lon1: s1.lon, lat1: s1.lat,
        lon2: s2.lon, lat2: s2.lat,
        isGood: true,  // assigned below
        ...strain,
      },
      maxEdgeKm,
      minAngle,
    })
  }

  // Percentile-based max-edge threshold
  const edges = raw.map(r => r.maxEdgeKm).sort((a, b) => a - b)
  const pctlIdx = Math.min(edges.length - 1, Math.floor(params.maxEdgePctl / 100 * edges.length))
  const edgeThresh = (edges[pctlIdx] ?? Infinity) * params.maxEdgeFactor

  // Apply quality filters and mark isGood
  const triangles: StrainTriangle[] = raw.map(r => ({
    ...r.tri,
    isGood:
      r.minAngle >= params.minAngleDeg &&
      r.maxEdgeKm <= edgeThresh &&
      (params.maxEdgeKm === null || r.maxEdgeKm <= params.maxEdgeKm),
  }))

  const good = triangles.filter(t => t.isGood).length
  log.push(`    Delaunay: ${raw.length} triangles  |  after filter: ${good} good`)
  log.push(`    Edge threshold: ${edgeThresh.toFixed(0)} km  (p${params.maxEdgePctl}×${params.maxEdgeFactor})`)

  return triangles
}

// ── Shared computation pipeline ───────────────────────────────────

function runPipeline(rawSites: GnssSite[], params: ComputeParams, label: string): ComputeResult {
  const log: string[] = []
  const push = (s: string) => log.push(s)

  push('='.repeat(54))
  push(`GNSS Strain Rate Calculation  [${label}]`)
  push('='.repeat(54))
  push('')
  push(`[1] Velocity data  (format=${params.velFormat})`)
  push(`    Sites: ${rawSites.length}`)

  // Thinning
  let workSites = rawSites
  if (params.minSpacingKm !== null && params.minSpacingKm > 0) {
    workSites = thinSites(rawSites, params.minSpacingKm)
    push(`    Thinning (min_spacing=${params.minSpacingKm} km): ${rawSites.length} → ${workSites.length}`)
  }

  // Bounds
  if (workSites.length === 0) {
    const bounds: DataBounds = { lonMin: 0, lonMax: 1, latMin: 0, latMax: 1 }
    return { rawSites, cleanSites: [], outlierSites: [], triangles: [], bounds, stats: { nInput: 0, nClean: 0, nOutlier: 0, nTriangles: 0, nGoodTri: 0, dilatMin: 0, dilatMax: 0, shearMax: 0 }, log }
  }
  const lons = workSites.map(s => s.lon), lats = workSites.map(s => s.lat)
  push(`    lon [${Math.min(...lons).toFixed(1)}, ${Math.max(...lons).toFixed(1)}]  lat [${Math.min(...lats).toFixed(1)}, ${Math.max(...lats).toFixed(1)}]`)

  // Outlier detection
  push('')
  push(`[2] Outlier detection (MAD×${params.madFactor})`)
  const markedSites = detectOutliers(workSites, params.madFactor)
  const cleanSites   = markedSites.filter(s => !s.isOutlier)
  const outlierSites = markedSites.filter(s => s.isOutlier)
  push(`    Outliers: ${outlierSites.length} / ${markedSites.length}`)
  push(`    Clean:    ${cleanSites.length}`)

  // Triangulation
  push('')
  push(`[3] Delaunay triangulation + quality control`)
  push(`    min_angle=${params.minAngleDeg}°  max_edge=p${params.maxEdgePctl}×${params.maxEdgeFactor}${params.maxEdgeKm !== null ? `  (hard cap ${params.maxEdgeKm} km)` : ''}`)
  const triangles = triangulateAndFilter(cleanSites, params, log)
  const goodTri = triangles.filter(t => t.isGood)

  // Strain stats
  push('')
  push('[4] Strain rate statistics')
  let dilatMin = 0, dilatMax = 0, shearMax = 0
  if (goodTri.length > 0) {
    const dilatVals = goodTri.map(t => t.dilatation)
    const shearVals = goodTri.map(t => t.maxShear)
    dilatMin = Math.min(...dilatVals)
    dilatMax = Math.max(...dilatVals)
    shearMax = Math.max(...shearVals)
    push(`    Dilatation range: ${dilatMin.toFixed(1)} to ${dilatMax.toFixed(1)} nstrain/yr`)
    push(`    Max shear range:  0 to ${shearMax.toFixed(1)} nstrain/yr`)
    push(`    Mean dilatation:  ${median(dilatVals).toFixed(1)} nstrain/yr`)
  } else {
    push('    No valid triangles.')
  }

  push('')
  push(`[5] Monte Carlo uncertainty (iterations=${params.mcIterations})`)
  push('    [skipped in real-time mode]')

  push('')
  push(`[6] Output → ${params.outputDir}/`)
  push('')
  push('='.repeat(54))
  push('Complete!')
  push('='.repeat(54))

  const bounds = computeBounds(workSites)

  return {
    rawSites,
    cleanSites,
    outlierSites,
    triangles,
    bounds,
    stats: {
      nInput:     rawSites.length,
      nClean:     cleanSites.length,
      nOutlier:   outlierSites.length,
      nTriangles: triangles.length,
      nGoodTri:   goodTri.length,
      dilatMin,
      dilatMax,
      shearMax,
    },
    log,
  }
}

// ── Mock data generator ───────────────────────────────────────────

/**
 * Generate a synthetic GNSS velocity field with a tectonic shear zone.
 * The resulting velocities are processed through the same real computation
 * pipeline so that strain rates and triangulation parameters are meaningful.
 */
export function generateMockData(params: ComputeParams, seed = 42): ComputeResult {
  const rng = seededRng(seed)

  // East China region
  const lonMin = 105, lonMax = 125
  const latMin = 25,  latMax = 45
  const N = 220

  // Right-lateral shear zone along ~115°E
  const lonFault = 115
  // Eastward velocity increases across fault (right-lateral in north-going sense)
  const halfSlip = 4.0  // mm/yr half-amplitude

  const rawSites: GnssSite[] = []
  for (let i = 0; i < N; i++) {
    const lon = lonMin + rng() * (lonMax - lonMin)
    const lat = latMin + rng() * (latMax - latMin)

    // Distance east of fault (km)
    const d = (lon - lonFault) * Math.cos(lat * Math.PI / 180) * 111.32
    const tanhD = Math.tanh(d / 300)  // 300 km characteristic half-width

    // Velocity field: background plate motion + shear zone
    const ve = 8  + halfSlip * tanhD + (rng() - 0.5) * 1.0
    const vn = -2 - halfSlip * tanhD * 0.5 + (rng() - 0.5) * 0.8

    rawSites.push({
      lon, lat,
      ve, vn,
      se: 0.3 + rng() * 0.4,
      sn: 0.3 + rng() * 0.4,
      name: `SITE${String(i).padStart(4, '0')}`,
    })
  }

  return runPipeline(rawSites, params, 'mock data')
}

// ── Real data pipeline ────────────────────────────────────────────

/**
 * Run the computation pipeline on real loaded GNSS sites.
 */
export function computeFromSites(
  inputSites: GnssSite[],
  params: ComputeParams,
  _seed = 42  // kept for API compatibility; unused
): ComputeResult {
  return runPipeline(inputSites, params, 'real data')
}
