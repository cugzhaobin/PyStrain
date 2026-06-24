import React from 'react'
import type { ComputeResult } from '../types'

interface Props { result: ComputeResult }

function Stat({ label, value, unit, accent }: {
  label: string; value: string | number; unit?: string; accent?: boolean
}) {
  return (
    <div
      className="flex flex-col items-center px-4 py-2 rounded-md"
      style={{ background: 'hsl(var(--ide-surface))' }}
    >
      <span style={{ fontSize: '18px', fontWeight: 700, fontFamily: 'JetBrains Mono, monospace',
        color: accent ? 'hsl(var(--accent))' : 'hsl(var(--foreground))' }}>
        {value}
        {unit && <span style={{ fontSize: '11px', marginLeft: '2px', color: 'hsl(var(--muted))' }}>{unit}</span>}
      </span>
      <span style={{ fontSize: '10px', color: 'hsl(var(--muted))', whiteSpace: 'nowrap' }}>{label}</span>
    </div>
  )
}

export function StatsBar({ result }: Props) {
  const s = result.stats
  const pct = ((s.nClean / s.nInput) * 100).toFixed(0)
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <Stat label="Input sites"   value={s.nInput} accent />
      <Stat label="Used"          value={s.nClean} />
      <Stat label="Outliers"      value={s.nOutlier} />
      <Stat label="Retention"     value={pct} unit="%" />
      <Stat label="Valid triangles" value={s.nGoodTri} accent />
      <Stat label="Dilat. range"  value={`${s.dilatMin.toFixed(0)} ~ ${s.dilatMax.toFixed(0)}`} unit=" nε/yr" />
      <Stat label="Max shear"     value={s.shearMax.toFixed(0)} unit=" nε/yr" />
    </div>
  )
}
