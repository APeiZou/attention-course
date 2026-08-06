import { ExternalLink, FileText, Github, BookOpen } from 'lucide-react'

export interface RefItem {
  type: 'paper' | 'code' | 'doc'
  label: string
  desc?: string
  url: string
}

const ICONS = {
  paper: <FileText size={15} className="text-red-500 shrink-0" />,
  code: <Github size={15} className="text-slate-800 shrink-0" />,
  doc: <BookOpen size={15} className="text-blue-500 shrink-0" />,
}

export default function RefBox({ title = '参考依据（论文原文与官方代码）', items }: { title?: string; items: RefItem[] }) {
  return (
    <div className="my-8 rounded-xl border border-slate-300 bg-white shadow-sm overflow-hidden">
      <div className="px-5 py-3 bg-slate-100 border-b border-slate-200 flex items-center gap-2">
        <BookMarked />
        <span className="font-bold text-slate-800 text-sm">{title}</span>
      </div>
      <ul className="divide-y divide-slate-100">
        {items.map((r, i) => (
          <li key={i}>
            <a href={r.url} target="_blank" rel="noreferrer"
              className="flex items-start gap-3 px-5 py-3 hover:bg-slate-50 transition-colors group">
              {ICONS[r.type]}
              <span className="flex-1 min-w-0">
                <span className="text-sm font-semibold text-slate-800 group-hover:text-blue-600 transition-colors block">{r.label}</span>
                {r.desc && <span className="text-xs text-slate-500 block mt-0.5">{r.desc}</span>}
                <span className="text-[11px] text-slate-400 font-mono block mt-0.5 break-all">{r.url}</span>
              </span>
              <ExternalLink size={13} className="text-slate-300 group-hover:text-blue-500 shrink-0 mt-1" />
            </a>
          </li>
        ))}
      </ul>
    </div>
  )
}

function BookMarked() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-slate-600">
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" /><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
    </svg>
  )
}
