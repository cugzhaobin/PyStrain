import React, { useEffect, useRef } from 'react'

interface Props { lines: string[] }

export function LogPanel({ lines }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [lines])

  return (
    <div
      ref={ref}
      className="h-full overflow-y-auto p-3"
      style={{
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: '11px',
        lineHeight: '1.6',
        color: 'hsl(var(--muted))',
      }}
    >
      {lines.map((line, i) => {
        const isHeader = line.startsWith('=') || line.startsWith('#')
        const isStep   = /^\[\d\]/.test(line)
        const isGood   = line.includes('Done') || line.includes('Complete')
        const isDanger = line.includes('Warning') || line.includes('Removed')
        return (
          <div key={i} style={{
            color: isHeader ? 'hsl(var(--accent))'
              : isStep    ? 'hsl(var(--foreground))'
              : isGood    ? 'hsl(var(--success))'
              : isDanger  ? 'hsl(var(--warning))'
              : undefined,
            fontWeight: isStep ? 600 : undefined,
          }}>
            {line || '\u00A0'}
          </div>
        )
      })}
      {lines.length === 0 && (
        <div style={{ color: 'hsl(var(--muted))', fontStyle: 'italic' }}>
          Run calculation to see output log…
        </div>
      )}
    </div>
  )
}
