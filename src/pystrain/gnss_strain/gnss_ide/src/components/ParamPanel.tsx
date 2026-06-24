import React, { useRef, useState } from 'react'
import {
  ChevronDown, ChevronRight,
  Database, Triangle, AlertCircle, Sliders, Activity,
  Play, RotateCcw, Download, Upload, BarChart2,
} from 'lucide-react'
import { cn } from '../lib/utils'
import type { ComputeParams } from '../types'
import { DEFAULT_PARAMS } from '../types'

interface Props {
  params: ComputeParams
  onChange: (p: ComputeParams) => void
  onRun: () => void
  onReset: () => void
  onPlot: () => void          // trigger showing raw velocity view
  isRunning: boolean
  progress: number   // 0-100
  stage: string
  inputFileName?: string
  onFileLoad?: (file: File) => void
}

// ── Collapsible section ────────────────────────────────────────
function Section({
  icon: Icon, title, children, defaultOpen = true,
}: {
  icon: React.ElementType
  title: string
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border-b" style={{ borderColor: 'hsl(var(--ide-border))' }}>
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-3 py-2 hover:bg-[hsl(var(--ide-elevated))] transition-colors"
      >
        <Icon size={12} style={{ color: 'hsl(var(--accent))' }} />
        <span className="section-label flex-1 text-left">{title}</span>
        {open
          ? <ChevronDown size={10} style={{ color: 'hsl(var(--muted))' }} />
          : <ChevronRight size={10} style={{ color: 'hsl(var(--muted))' }} />}
      </button>
      {open && (
        <div className="px-3 pb-3 pt-1 space-y-2 animate-slide-in-left">
          {children}
        </div>
      )}
    </div>
  )
}

// ── Param row ──────────────────────────────────────────────────
function ParamRow({ label, hint, children }: {
  label: string; hint?: string; children: React.ReactNode
}) {
  return (
    <div className="flex items-center justify-between gap-2 min-h-[26px]">
      <div className="flex-1">
        <div style={{ color: 'hsl(var(--foreground))', fontSize: '11.5px' }}>{label}</div>
        {hint && <div style={{ color: 'hsl(var(--muted))', fontSize: '10px' }}>{hint}</div>}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  )
}

// ── Slider + number combined ───────────────────────────────────
function SliderNum({
  value, onChange, min, max, step = 1, format
}: {
  value: number; onChange: (v: number) => void
  min: number; max: number; step?: number
  format?: (v: number) => string
}) {
  return (
    <div className="flex items-center gap-2">
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: '80px' }}
      />
      <span style={{
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: '11px',
        color: 'hsl(var(--accent))',
        minWidth: '42px',
        textAlign: 'right',
      }}>
        {format ? format(value) : value}
      </span>
    </div>
  )
}

// ── Toggle with label ──────────────────────────────────────────
function ToggleNull({
  value, onChange, label, children
}: {
  value: number | null
  onChange: (v: number | null) => void
  label: string
  children: (v: number) => React.ReactNode
}) {
  const enabled = value !== null
  return (
    <div className="space-y-1">
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={enabled}
          onChange={e => onChange(e.target.checked ? 50 : null)}
        />
        <span style={{ color: 'hsl(var(--foreground))', fontSize: '11.5px' }}>{label}</span>
      </label>
      {enabled && value !== null && (
        <div className="pl-5 animate-fade-up">
          {children(value)}
        </div>
      )}
    </div>
  )
}

// ── Main sidebar ───────────────────────────────────────────────
export function ParamPanel({ params, onChange, onRun, onReset, onPlot, isRunning, progress, stage, inputFileName, onFileLoad }: Props) {
  const set = <K extends keyof ComputeParams>(key: K, val: ComputeParams[K]) =>
    onChange({ ...params, [key]: val })

  // Hidden file input — triggered programmatically by the Import button
  const fileInputRef = useRef<HTMLInputElement>(null)

  return (
    <div
      className="flex flex-col h-full overflow-hidden"
      style={{ background: 'hsl(var(--ide-sidebar))' }}
    >
      {/* Header */}
      <div
        className="flex items-center gap-2 px-3 py-3 border-b shrink-0"
        style={{ borderColor: 'hsl(var(--ide-border))' }}
      >
        <div
          className="w-6 h-6 rounded flex items-center justify-center text-xs font-bold shrink-0"
          style={{ background: 'hsl(var(--accent))', color: 'hsl(222 24% 9%)' }}
        >G</div>
        <div>
          <div style={{ color: 'hsl(var(--foreground))', fontSize: '12px', fontWeight: 600, lineHeight: 1.2 }}>
            GNSS Strain IDE
          </div>
          <div style={{ color: 'hsl(var(--muted))', fontSize: '10px' }}>
            v2.0 · Delaunay + MC
          </div>
        </div>
      </div>

      {/* Scrollable params */}
      <div className="flex-1 overflow-y-auto">

        {/* ── Data ───────────────────── */}
        <Section icon={Database} title="Data">
          {/* Hidden real file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".vel,.dat,.txt,.csv"
            className="hidden"
            onChange={e => {
              const f = e.target.files?.[0]
              if (f && onFileLoad) onFileLoad(f)
              e.target.value = ''
            }}
          />

          {/* Import + Plot buttons */}
          <div className="flex gap-2">
            <button
              className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded text-xs font-medium transition-all hover:brightness-110 active:scale-95"
              style={{
                background: 'hsl(var(--ide-elevated))',
                border: '1px solid hsl(var(--ide-border))',
                color: 'hsl(var(--muted))',
              }}
              onClick={() => fileInputRef.current?.click()}
              title="Load a velocity file (.vel / .dat / .txt)"
            >
              <Upload size={11} />
              导入
            </button>
            <button
              className={cn(
                'flex-1 flex items-center justify-center gap-1.5 py-2 rounded text-xs font-medium transition-all active:scale-95',
                inputFileName
                  ? 'hover:brightness-110'
                  : 'opacity-50 cursor-not-allowed'
              )}
              style={{
                background: inputFileName ? 'hsl(var(--accent) / 0.15)' : 'hsl(var(--ide-elevated))',
                border: `1px solid ${inputFileName ? 'hsl(var(--accent) / 0.5)' : 'hsl(var(--ide-border))'}`,
                color: inputFileName ? 'hsl(var(--accent))' : 'hsl(var(--muted))',
              }}
              disabled={!inputFileName}
              onClick={onPlot}
              title="Show velocity field in the right panel"
            >
              <BarChart2 size={11} />
              绘图
            </button>
          </div>

          {/* Loaded filename display */}
          {inputFileName ? (
            <div
              className="flex items-center gap-1.5 px-2 py-1 rounded truncate"
              style={{
                background: 'hsl(var(--ide-elevated))',
                border: '1px solid hsl(var(--ide-border))',
                color: 'hsl(var(--accent))',
                fontSize: '10px',
                fontFamily: 'JetBrains Mono, monospace',
              }}
            >
              <span className="truncate">{inputFileName}</span>
            </div>
          ) : (
            <div style={{ color: 'hsl(var(--muted))', fontSize: '10px' }}>
              先点击导入加载速度文件
            </div>
          )}

          <ParamRow label="Format">
            <select
              value={params.velFormat}
              onChange={e => set('velFormat', e.target.value as ComputeParams['velFormat'])}
              style={{ width: '100px' }}
            >
              <option value="auto">auto</option>
              <option value="gmt">gmt (8 col)</option>
              <option value="globk">globk (13 col)</option>
            </select>
          </ParamRow>
          <ParamRow label="Output dir">
            <input
              type="text"
              value={params.outputDir}
              onChange={e => set('outputDir', e.target.value)}
              style={{
                background: 'hsl(var(--ide-elevated))',
                border: '1px solid hsl(var(--ide-border))',
                borderRadius: '5px',
                padding: '3px 6px',
                fontSize: '11px',
                color: 'hsl(var(--foreground))',
                width: '100px',
              }}
            />
          </ParamRow>
        </Section>

        {/* ── Density control ─────────── */}
        <Section icon={Sliders} title="Density Control">
          <ToggleNull
            value={params.minSpacingKm}
            onChange={v => set('minSpacingKm', v)}
            label="Site thinning"
          >
            {v => (
              <ParamRow label="Min spacing (km)">
                <SliderNum
                  value={v} onChange={nv => set('minSpacingKm', nv)}
                  min={5} max={200} step={5}
                />
              </ParamRow>
            )}
          </ToggleNull>
          <ToggleNull
            value={params.maxEdgeKm}
            onChange={v => set('maxEdgeKm', v)}
            label="Max edge length"
          >
            {v => (
              <ParamRow label="Max edge (km)">
                <SliderNum
                  value={v} onChange={nv => set('maxEdgeKm', nv)}
                  min={50} max={1000} step={10}
                />
              </ParamRow>
            )}
          </ToggleNull>
        </Section>

        {/* ── Triangulation ───────────── */}
        <Section icon={Triangle} title="Triangulation Quality">
          <ParamRow label="Min angle (°)">
            <SliderNum
              value={params.minAngleDeg}
              onChange={v => set('minAngleDeg', v)}
              min={1} max={30} step={1}
              format={v => `${v}°`}
            />
          </ParamRow>
          <ParamRow label="Edge percentile">
            <SliderNum
              value={params.maxEdgePctl}
              onChange={v => set('maxEdgePctl', v)}
              min={50} max={99} step={1}
              format={v => `${v}%`}
            />
          </ParamRow>
          <ParamRow label="Edge factor">
            <SliderNum
              value={params.maxEdgeFactor}
              onChange={v => set('maxEdgeFactor', v)}
              min={1.0} max={4.0} step={0.1}
              format={v => v.toFixed(1)}
            />
          </ParamRow>
        </Section>

        {/* ── Outlier ─────────────────── */}
        <Section icon={AlertCircle} title="Outlier Detection">
          <ParamRow label="KNN neighbors">
            <SliderNum
              value={params.kNeighbors}
              onChange={v => set('kNeighbors', v)}
              min={3} max={20} step={1}
            />
          </ParamRow>
          <ParamRow label="MAD factor">
            <SliderNum
              value={params.madFactor}
              onChange={v => set('madFactor', v)}
              min={1.0} max={8.0} step={0.1}
              format={v => v.toFixed(1)}
            />
          </ParamRow>
          <ParamRow label="IQR factor">
            <SliderNum
              value={params.iqrFactor}
              onChange={v => set('iqrFactor', v)}
              min={0.5} max={5.0} step={0.1}
              format={v => v.toFixed(1)}
            />
          </ParamRow>
          <ParamRow label="Max iterations">
            <SliderNum
              value={params.maxOutlierIter}
              onChange={v => set('maxOutlierIter', v)}
              min={1} max={20} step={1}
            />
          </ParamRow>
        </Section>

        {/* ── Smoothing ───────────────── */}
        <Section icon={Activity} title="Smoothing & Uncertainty" defaultOpen={false}>
          <ParamRow label="Smooth weight">
            <SliderNum
              value={params.smoothWeight}
              onChange={v => set('smoothWeight', v)}
              min={0} max={1} step={0.05}
              format={v => v.toFixed(2)}
            />
          </ParamRow>
          <ParamRow label="Smooth iter">
            <SliderNum
              value={params.smoothIter}
              onChange={v => set('smoothIter', v)}
              min={0} max={10} step={1}
            />
          </ParamRow>
          <ParamRow label="Monte Carlo">
            <SliderNum
              value={params.mcIterations}
              onChange={v => set('mcIterations', v)}
              min={50} max={2000} step={50}
            />
          </ParamRow>
        </Section>
      </div>

      {/* Progress */}
      {isRunning && (
        <div
          className="px-3 py-2 border-t shrink-0"
          style={{ borderColor: 'hsl(var(--ide-border))' }}
        >
          <div className="flex items-center gap-2 mb-1">
            <div
              className="w-2 h-2 rounded-full animate-pulse-dot"
              style={{ background: 'hsl(var(--accent))' }}
            />
            <span style={{ color: 'hsl(var(--muted))', fontSize: '10px' }}>{stage}</span>
          </div>
          <div
            className="h-1 rounded-full overflow-hidden"
            style={{ background: 'hsl(var(--ide-elevated))' }}
          >
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{
                width: `${progress}%`,
                background: 'hsl(var(--accent))',
                boxShadow: '0 0 8px hsl(var(--accent) / 0.6)',
              }}
            />
          </div>
        </div>
      )}

      {/* Action buttons */}
      <div
        className="p-3 space-y-2 border-t shrink-0"
        style={{ borderColor: 'hsl(var(--ide-border))' }}
      >
        {/* import/export config */}
        <div className="flex gap-2">
          <button
            className="flex-1 flex items-center justify-center gap-1 py-1.5 rounded text-xs transition-colors"
            style={{
              background: 'hsl(var(--ide-elevated))',
              border: '1px solid hsl(var(--ide-border))',
              color: 'hsl(var(--muted))',
            }}
            onClick={() => {
              const cfg = JSON.stringify(params, null, 2)
              const a = document.createElement('a')
              a.href = URL.createObjectURL(new Blob([cfg], { type: 'application/json' }))
              a.download = 'gnss_strain_params.json'
              a.click()
            }}
          >
            <Download size={10} /> Export
          </button>
          <label
            className="flex-1 flex items-center justify-center gap-1 py-1.5 rounded text-xs cursor-pointer transition-colors"
            style={{
              background: 'hsl(var(--ide-elevated))',
              border: '1px solid hsl(var(--ide-border))',
              color: 'hsl(var(--muted))',
            }}
          >
            <Upload size={10} /> Import
            <input
              type="file" accept=".json" className="hidden"
              onChange={e => {
                const f = e.target.files?.[0]
                if (!f) return
                f.text().then(t => {
                  try { onChange({ ...DEFAULT_PARAMS, ...JSON.parse(t) }) } catch { /* ignore */ }
                })
              }}
            />
          </label>
        </div>

        <button
          disabled={isRunning}
          onClick={onRun}
          className={cn(
            'w-full flex items-center justify-center gap-2 py-2.5 rounded-md font-semibold text-xs transition-all',
            isRunning ? 'opacity-60 cursor-not-allowed' : 'hover:brightness-110 active:scale-95'
          )}
          style={{
            background: isRunning ? 'hsl(var(--ide-elevated))' : 'hsl(var(--accent))',
            color: isRunning ? 'hsl(var(--muted))' : 'hsl(222 24% 9%)',
            boxShadow: isRunning ? 'none' : '0 0 16px hsl(var(--accent) / 0.35)',
          }}
        >
          <Play size={12} />
          {isRunning ? 'Computing…' : 'Run Calculation'}
        </button>

        <button
          onClick={onReset}
          disabled={isRunning}
          className="w-full flex items-center justify-center gap-2 py-1.5 rounded-md text-xs transition-colors"
          style={{
            background: 'transparent',
            border: '1px solid hsl(var(--ide-border))',
            color: 'hsl(var(--muted))',
          }}
        >
          <RotateCcw size={10} /> Reset Defaults
        </button>
      </div>
    </div>
  )
}
