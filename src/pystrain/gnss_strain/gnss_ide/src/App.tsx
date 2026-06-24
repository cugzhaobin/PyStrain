import React, { useState, useCallback, useEffect, useRef } from 'react'
import {
  Map, AlertOctagon, Wind, Network,
  Activity, Crosshair, Terminal, Download,
} from 'lucide-react'
import { cn } from './lib/utils'
import { ParamPanel }    from './components/ParamPanel'
import { StatsBar }      from './components/StatsBar'
import { LogPanel }      from './components/LogPanel'
import {
  RawVelocityView, OutlierView, TriangulationView,
  DilatationView, MaxShearView, PrincipalStrainView,
} from './components/MapViews'
import { generateMockData, parseVelocityFile, computeFromSites } from './mockEngine'
import { DEFAULT_PARAMS }   from './types'
import type { ComputeParams, ComputeResult } from './types'

// ── Tab config ─────────────────────────────────────────────────
const TABS = [
  { id: 'raw',       label: 'Raw Velocity',    icon: Map },
  { id: 'outlier',   label: 'Outliers',         icon: AlertOctagon },
  { id: 'clean',     label: 'Clean Velocity',   icon: Wind },
  { id: 'tri',       label: 'Triangulation',    icon: Network },
  { id: 'dilat',     label: 'Dilatation',       icon: Activity },
  { id: 'shear',     label: 'Max Shear',        icon: Crosshair },
  { id: 'principal', label: 'Principal Strain', icon: Crosshair },
  { id: 'log',       label: 'Run Log',          icon: Terminal },
] as const
type TabId = typeof TABS[number]['id']

// ── Stage descriptions ─────────────────────────────────────────
const STAGES: [number, string][] = [
  [10, 'Loading velocity data…'],
  [25, 'KNN outlier pre-screening…'],
  [40, 'Delaunay triangulation…'],
  [55, 'Iterative outlier removal…'],
  [70, 'Computing strain rates…'],
  [82, 'Monte Carlo uncertainty…'],
  [94, 'Writing output…'],
  [100, 'Done!'],
]

function downloadCanvas(canvasId: string, filename: string) {
  const canvas = document.querySelector<HTMLCanvasElement>(canvasId)
  if (!canvas) return
  const a = document.createElement('a')
  a.href = canvas.toDataURL('image/png')
  a.download = filename
  a.click()
}

export default function App() {
  const [params, setParams] = useState<ComputeParams>(DEFAULT_PARAMS)
  const [result, setResult] = useState<ComputeResult | null>(null)
  const [activeTab, setActiveTab] = useState<TabId>('raw')
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [stage, setStage] = useState('')
  const [hasRun, setHasRun] = useState(false)
  // Loaded real data file (null = use mock)
  const [inputFile, setInputFile] = useState<File | null>(null)
  const loadedSitesRef = useRef<import('./types').GnssSite[] | null>(null)
  // Stable seed so live-preview keeps the same random site layout
  const seedRef = useRef(42)
  // Ref for the view container — used for reliable canvas PNG export
  const viewRef = useRef<HTMLDivElement>(null)

  // ── Live preview: re-run instantly whenever params change ──
  useEffect(() => {
    if (!hasRun) return
    const sites = loadedSitesRef.current
    const res = sites
      ? computeFromSites(sites, params, seedRef.current)
      : generateMockData(params, seedRef.current)
    setResult(res)
  // params is the only real trigger; hasRun is read once
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params])

  // ── Handle file load ──────────────────────────────────────
  const handleFileLoad = useCallback((file: File) => {
    setInputFile(file)
    file.text().then(text => {
      const sites = parseVelocityFile(text, params.velFormat)
      loadedSitesRef.current = sites
      // Immediately preview the loaded data
      const res = computeFromSites(sites, params, seedRef.current)
      setResult(res)
      setHasRun(true)
      setActiveTab('raw')
    })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params])

  const handleRun = useCallback(async () => {
    setIsRunning(true)
    setActiveTab('raw')
    // Pick a new seed on each explicit run so results feel fresh
    if (!loadedSitesRef.current) seedRef.current = Date.now() & 0xffff

    for (const [pct, msg] of STAGES) {
      setProgress(pct)
      setStage(msg)
      await new Promise(r => setTimeout(r, 220 + Math.random() * 180))
    }

    const sites = loadedSitesRef.current
    const res = sites
      ? computeFromSites(sites, params, seedRef.current)
      : generateMockData(params, seedRef.current)
    setResult(res)
    setHasRun(true)
    setIsRunning(false)
    setProgress(0)
    setStage('')
  }, [params])

  const handleReset = useCallback(() => {
    setParams(DEFAULT_PARAMS)
    setResult(null)
    setHasRun(false)
    setInputFile(null)
    loadedSitesRef.current = null
    seedRef.current = 42
    setActiveTab('raw')
  }, [])

  // Plot: switch to raw velocity tab (data already loaded by handleFileLoad)
  const handlePlot = useCallback(() => {
    setActiveTab('raw')
  }, [])

  const logLines = result?.log ?? []

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: 'hsl(var(--ide-bg))' }}>

      {/* ── LEFT SIDEBAR ────────────────────────────────── */}
      <div
        className="w-72 shrink-0 flex flex-col border-r"
        style={{ borderColor: 'hsl(var(--ide-border))' }}
      >
        <ParamPanel
          params={params}
          onChange={setParams}
          onRun={handleRun}
          onReset={handleReset}
          onPlot={handlePlot}
          isRunning={isRunning}
          progress={progress}
          stage={stage}
          inputFileName={inputFile?.name}
          onFileLoad={handleFileLoad}
        />
      </div>

      {/* ── RIGHT MAIN ──────────────────────────────────── */}
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">

        {/* Top bar */}
        <div
          className="flex items-center gap-3 px-4 py-2 border-b shrink-0"
          style={{
            borderColor: 'hsl(var(--ide-border))',
            background: 'hsl(var(--ide-surface))',
          }}
        >
          <span style={{ color: 'hsl(var(--muted))', fontSize: '11px' }}>
            GNSS Strain Rate Calculator
          </span>
          <span style={{ color: 'hsl(var(--ide-border))' }}>·</span>
          <span style={{ color: 'hsl(var(--accent))', fontSize: '11px', fontFamily: 'JetBrains Mono, monospace' }}>
            {result
              ? `${result.stats.nInput} sites  →  ${result.stats.nGoodTri} triangles`
              : 'No data loaded'}
          </span>
          <div className="flex-1" />
          {result && (
            <button
              onClick={() => {
                const data = JSON.stringify(result.stats, null, 2)
                const a = document.createElement('a')
                a.href = URL.createObjectURL(new Blob([data], { type: 'application/json' }))
                a.download = 'strain_stats.json'
                a.click()
              }}
              className="flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors hover:brightness-110"
              style={{
                background: 'hsl(var(--ide-elevated))',
                border: '1px solid hsl(var(--ide-border))',
                color: 'hsl(var(--muted))',
              }}
            >
              <Download size={10} /> Stats
            </button>
          )}
        </div>

        {/* Stats bar */}
        {result && (
          <div
            className="px-4 py-2 border-b shrink-0 overflow-x-auto"
            style={{
              borderColor: 'hsl(var(--ide-border))',
              background: 'hsl(var(--ide-sidebar))',
            }}
          >
            <StatsBar result={result} />
          </div>
        )}

        {/* Tab bar */}
        <div
          className="flex items-end gap-0 border-b shrink-0 overflow-x-auto"
          style={{ borderColor: 'hsl(var(--ide-border))', background: 'hsl(var(--ide-surface))' }}
        >
          {TABS.map(tab => {
            const Icon = tab.icon
            const isActive = activeTab === tab.id
            const disabled = !hasRun && tab.id !== 'log'
            return (
              <button
                key={tab.id}
                disabled={disabled}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  'flex items-center gap-1.5 px-3 py-2 text-xs whitespace-nowrap',
                  'border-b-2 transition-all',
                  isActive
                    ? 'border-[hsl(var(--accent))]'
                    : 'border-transparent hover:border-[hsl(var(--ide-border))]',
                  disabled && 'opacity-30 cursor-not-allowed'
                )}
                style={{
                  color: isActive ? 'hsl(var(--accent))' : 'hsl(var(--muted))',
                  background: isActive ? 'hsl(var(--ide-elevated))' : 'transparent',
                }}
              >
                <Icon size={11} />
                {tab.label}
              </button>
            )
          })}
        </div>

        {/* View area */}
        <div className="flex-1 overflow-hidden relative">

          {/* Empty / welcome state */}
          {!hasRun && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 pointer-events-none select-none">
              <img
                src="/src/assets/images/hero_bg.png"
                alt="GNSS background"
                className="absolute inset-0 w-full h-full object-cover opacity-20"
                style={{ filter: 'blur(1px)' }}
              />
              <div className="relative z-10 text-center space-y-2">
                <div
                  className="text-2xl font-semibold text-gradient-accent"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  GNSS Strain Rate IDE
                </div>
                <div style={{ color: 'hsl(var(--muted))', fontSize: '13px' }}>
                  Configure parameters on the left, then click&nbsp;
                  <span style={{ color: 'hsl(var(--accent))' }}>Run Calculation</span>
                </div>
                <div
                  className="flex items-center justify-center gap-4 pt-2"
                  style={{ color: 'hsl(var(--muted))', fontSize: '11px' }}
                >
                  <span>GMT 8-column</span>
                  <span style={{ color: 'hsl(var(--ide-border))' }}>|</span>
                  <span>GLOBK 13-column</span>
                  <span style={{ color: 'hsl(var(--ide-border))' }}>|</span>
                  <span>Delaunay triangulation</span>
                  <span style={{ color: 'hsl(var(--ide-border))' }}>|</span>
                  <span>Monte Carlo uncertainty</span>
                </div>
              </div>
            </div>
          )}

          {/* Running skeleton */}
          {isRunning && (
            <div className="absolute inset-0 flex items-center justify-center z-20"
              style={{ background: 'hsl(var(--ide-bg) / 0.7)', backdropFilter: 'blur(4px)' }}>
              <div className="text-center space-y-3">
                <div
                  className="w-10 h-10 rounded-full border-2 border-t-transparent animate-spin mx-auto"
                  style={{ borderColor: 'hsl(var(--accent))', borderTopColor: 'transparent' }}
                />
                <div style={{ color: 'hsl(var(--accent))', fontSize: '13px' }}>{stage}</div>
                <div
                  className="h-1.5 rounded-full overflow-hidden"
                  style={{ width: '200px', background: 'hsl(var(--ide-elevated))' }}
                >
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${progress}%`,
                      background: 'hsl(var(--accent))',
                      boxShadow: '0 0 10px hsl(var(--accent) / 0.7)',
                    }}
                  />
                </div>
                <div style={{ color: 'hsl(var(--muted))', fontSize: '11px' }}>{progress}%</div>
              </div>
            </div>
          )}

          {/* Actual views */}
          {result && !isRunning && (
            <div ref={viewRef} className="absolute inset-0 p-3 animate-fade-up">
              {activeTab === 'raw' && (
                <RawVelocityView
                  sites={result.rawSites}
                  outliers={result.outlierSites}
                  showOutliers={true}
                  bounds={result.bounds}
                />
              )}
              {activeTab === 'outlier' && (
                <OutlierView
                  sites={result.cleanSites}
                  outliers={result.outlierSites}
                  bounds={result.bounds}
                />
              )}
              {activeTab === 'clean' && (
                <RawVelocityView
                  sites={result.cleanSites}
                  outliers={[]}
                  showOutliers={false}
                  bounds={result.bounds}
                />
              )}
              {activeTab === 'tri' && (
                <TriangulationView
                  sites={result.cleanSites}
                  triangles={result.triangles}
                  bounds={result.bounds}
                />
              )}
              {activeTab === 'dilat' && (
                <DilatationView triangles={result.triangles} bounds={result.bounds} />
              )}
              {activeTab === 'shear' && (
                <MaxShearView triangles={result.triangles} bounds={result.bounds} />
              )}
              {activeTab === 'principal' && (
                <PrincipalStrainView triangles={result.triangles} bounds={result.bounds} />
              )}
              {activeTab === 'log' && (
                <div className="h-full rounded-lg overflow-hidden border"
                  style={{ borderColor: 'hsl(var(--ide-border))', background: 'hsl(var(--ide-surface))' }}>
                  <LogPanel lines={logLines} />
                </div>
              )}
            </div>
          )}
          {!result && activeTab === 'log' && (
            <div className="absolute inset-0 p-3">
              <div className="h-full rounded-lg overflow-hidden border"
                style={{ borderColor: 'hsl(var(--ide-border))', background: 'hsl(var(--ide-surface))' }}>
                <LogPanel lines={[]} />
              </div>
            </div>
          )}

          {/* Download button floating */}
          {result && activeTab !== 'log' && !isRunning && (
            <button
              onClick={() => {
                const canvas = viewRef.current?.querySelector<HTMLCanvasElement>('canvas')
                if (!canvas) return
                const a = document.createElement('a')
                a.href = canvas.toDataURL('image/png')
                a.download = `gnss_${activeTab}.png`
                a.click()
              }}
              className="absolute bottom-5 right-5 flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs shadow-lg transition-all hover:brightness-110 active:scale-95"
              style={{
                background: 'hsl(var(--ide-elevated))',
                border: '1px solid hsl(var(--ide-border))',
                color: 'hsl(var(--muted))',
              }}
            >
              <Download size={11} /> Save PNG
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

void downloadCanvas
