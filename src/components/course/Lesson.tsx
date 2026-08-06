import type { ReactNode } from 'react'
import { Lightbulb, Map, Terminal, GitBranch, BookOpen, AlertCircle } from 'lucide-react'

export const ACCENTS = {
  slate: { text: 'text-slate-600', bg: 'bg-slate-100', border: 'border-slate-300', solid: 'bg-slate-600', soft: 'bg-slate-50', ring: 'ring-slate-200' },
  orange: { text: 'text-orange-600', bg: 'bg-orange-100', border: 'border-orange-300', solid: 'bg-orange-500', soft: 'bg-orange-50', ring: 'ring-orange-200' },
  emerald: { text: 'text-emerald-600', bg: 'bg-emerald-100', border: 'border-emerald-300', solid: 'bg-emerald-500', soft: 'bg-emerald-50', ring: 'ring-emerald-200' },
  sky: { text: 'text-sky-600', bg: 'bg-sky-100', border: 'border-sky-300', solid: 'bg-sky-500', soft: 'bg-sky-50', ring: 'ring-sky-200' },
  violet: { text: 'text-violet-600', bg: 'bg-violet-100', border: 'border-violet-300', solid: 'bg-violet-500', soft: 'bg-violet-50', ring: 'ring-violet-200' },
} as const

export type Accent = keyof typeof ACCENTS

export function LessonHeader({ id, no, title, subtitle, accent }: { id: string; no: string; title: string; subtitle: string; accent: Accent }) {
  const a = ACCENTS[accent]
  return (
    <div id={id} className="scroll-mt-24 mb-10">
      <div className="flex items-center gap-3 mb-3">
        <span className={`px-3 py-1 rounded-full text-xs font-bold text-white ${a.solid}`}>{no}</span>
        <span className={`text-xs font-mono uppercase tracking-widest ${a.text}`}>{subtitle}</span>
      </div>
      <h2 className="text-3xl md:text-4xl font-extrabold text-slate-900 tracking-tight">{title}</h2>
      <div className={`mt-4 h-1.5 w-24 rounded-full ${a.solid}`} />
    </div>
  )
}

export function H3({ children, accent = 'slate' }: { children: ReactNode; accent?: Accent }) {
  const a = ACCENTS[accent]
  return <h3 className={`text-xl font-bold text-slate-900 mt-12 mb-4 flex items-center gap-2`}><span className={`w-1.5 h-6 rounded ${a.solid} inline-block`} />{children}</h3>
}

export function P({ children }: { children: ReactNode }) {
  return <p className="text-slate-600 leading-8 my-3 text-[15px]">{children}</p>
}

export function StepBox({ icon, title, children, accent = 'slate' }: { icon: 'idea' | 'map' | 'code' | 'flow' | 'warn' | 'book'; title: string; children: ReactNode; accent?: Accent }) {
  const a = ACCENTS[accent]
  const icons = {
    idea: <Lightbulb size={18} />, map: <Map size={18} />, code: <Terminal size={18} />,
    flow: <GitBranch size={18} />, warn: <AlertCircle size={18} />, book: <BookOpen size={18} />,
  }
  return (
    <div className={`rounded-xl border ${a.border} ${a.soft} p-5 my-6 ring-1 ${a.ring}`}>
      <div className={`flex items-center gap-2 font-bold mb-2 ${a.text}`}>{icons[icon]}{title}</div>
      <div className="text-slate-600 leading-7 text-[14.5px]">{children}</div>
    </div>
  )
}

// ---------- SVG 流程图元件 ----------
export function FlowNode({ x, y, w, h, label, sub, fill, stroke, textFill = '#fff', rx = 10, fontSize = 13 }: {
  x: number; y: number; w: number; h: number; label: string; sub?: string;
  fill: string; stroke?: string; textFill?: string; rx?: number; fontSize?: number
}) {
  const lines = label.split('\n')
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={rx} fill={fill} stroke={stroke || fill} strokeWidth={1.5} />
      {lines.map((l, i) => (
        <text key={i} x={x + w / 2} y={y + (sub ? h / 2 - 6 : h / 2) + (i - (lines.length - 1) / 2) * (fontSize + 3) + fontSize / 2.6}
          textAnchor="middle" fill={textFill} fontSize={fontSize} fontWeight={600}>{l}</text>
      ))}
      {sub && <text x={x + w / 2} y={y + h / 2 + 16} textAnchor="middle" fill={textFill} fontSize={10.5} opacity={0.85}>{sub}</text>}
    </g>
  )
}

export function Arrow({ x1, y1, x2, y2, color = '#94a3b8', dashed = false, label, labelOffset = -6 }: {
  x1: number; y1: number; x2: number; y2: number; color?: string; dashed?: boolean; label?: string; labelOffset?: number
}) {
  const id = `ah-${color.replace('#', '')}`
  const midX = (x1 + x2) / 2
  const midY = (y1 + y2) / 2
  return (
    <g>
      <defs>
        <marker id={id} markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L6,3 L0,6 Z" fill={color} />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={1.8} strokeDasharray={dashed ? '5 4' : undefined} markerEnd={`url(#${id})`} />
      {label && <text x={midX} y={midY + labelOffset} textAnchor="middle" fontSize={10.5} fill="#64748b" fontWeight={500}>{label}</text>}
    </g>
  )
}

export function ElbowArrow({ points, color = '#94a3b8', dashed = false, label, labelPos }: {
  points: [number, number][]; color?: string; dashed?: boolean; label?: string; labelPos?: [number, number]
}) {
  const d = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0]},${p[1]}`).join(' ')
  const last = points[points.length - 1]
  const prev = points[points.length - 2]
  // 末段方向，画一个简单三角箭头
  const dx = last[0] - prev[0]
  const dy = last[1] - prev[1]
  const len = Math.hypot(dx, dy) || 1
  const ux = dx / len, uy = dy / len
  const tip = `${last[0]},${last[1]}`
  const b1 = `${last[0] - ux * 8 - uy * 4},${last[1] - uy * 8 + ux * 4}`
  const b2 = `${last[0] - ux * 8 + uy * 4},${last[1] - uy * 8 - ux * 4}`
  return (
    <g>
      <path d={d} fill="none" stroke={color} strokeWidth={1.8} strokeDasharray={dashed ? '5 4' : undefined} />
      <polygon points={`${tip} ${b1} ${b2}`} fill={color} />
      {label && labelPos && <text x={labelPos[0]} y={labelPos[1]} textAnchor="middle" fontSize={10.5} fill="#64748b" fontWeight={500}>{label}</text>}
    </g>
  )
}

export function DiagramFrame({ children, viewBox, caption }: { children: ReactNode; viewBox: string; caption?: string }) {
  return (
    <div className="my-6 rounded-xl border border-slate-200 bg-white p-4 shadow-sm overflow-x-auto">
      <svg viewBox={viewBox} className="w-full min-w-[640px]" style={{ maxHeight: 560 }}>{children}</svg>
      {caption && <div className="text-center text-xs text-slate-500 mt-2">{caption}</div>}
    </div>
  )
}

// 伪代码块
export function PseudoCode({ lines, accent = 'slate' }: { lines: (string | { t: string; indent?: number; note?: string })[]; accent?: Accent }) {
  const a = ACCENTS[accent]
  return (
    <div className={`my-4 rounded-xl border ${a.border} bg-white shadow-sm overflow-hidden`}>
      <div className={`px-4 py-2 ${a.bg} text-xs font-bold ${a.text} font-mono tracking-wider`}>PSEUDOCODE · 伪代码流程</div>
      <div className="p-4 font-mono text-[13px] leading-7 overflow-x-auto">
        {lines.map((l, i) => {
          const item = typeof l === 'string' ? { t: l, indent: 0 } : l
          return (
            <div key={i} className="flex items-baseline gap-3 whitespace-nowrap" style={{ paddingLeft: (item.indent || 0) * 24 }}>
              <span className="text-slate-300 select-none w-6 text-right shrink-0">{String(i + 1).padStart(2, '0')}</span>
              <span className="text-slate-800">{item.t}</span>
              {item.note && <span className="text-slate-400 text-xs">// {item.note}</span>}
            </div>
          )
        })}
      </div>
    </div>
  )
}
