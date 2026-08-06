import { useState } from 'react'
import { Check, Copy } from 'lucide-react'

// 轻量 Python 语法高亮
const KEYWORDS = new Set([
  'import', 'from', 'class', 'def', 'return', 'if', 'else', 'elif', 'for',
  'while', 'in', 'not', 'and', 'or', 'None', 'True', 'False', 'pass',
  'with', 'as', 'raise', 'self', 'lambda', 'assert', 'is', 'del', 'yield',
])

function highlightLine(line: string, key: number) {
  // 注释
  const commentIdx = line.indexOf('#')
  let code = line
  let comment = ''
  if (commentIdx !== -1) {
    // 简单判断：# 不在引号内（课程代码足够简单）
    code = line.slice(0, commentIdx)
    comment = line.slice(commentIdx)
  }

  const tokens: React.ReactNode[] = []
  const regex = /("""[\s\S]*?"""|"[^"]*"|'[^']*'|\b\d+\.?\d*\b|[A-Za-z_][A-Za-z0-9_]*|\*\*|\S)/g
  let m: RegExpExecArray | null
  let i = 0
  let lastIdx = 0
  while ((m = regex.exec(code)) !== null) {
    if (m.index > lastIdx) {
      tokens.push(<span key={i++}>{code.slice(lastIdx, m.index)}</span>)
    }
    lastIdx = m.index + m[0].length
    const tok = m[0]
    if (/^["']/.test(tok)) {
      tokens.push(<span key={i++} className="text-emerald-300">{tok}</span>)
    } else if (/^\d/.test(tok)) {
      tokens.push(<span key={i++} className="text-orange-300">{tok}</span>)
    } else if (KEYWORDS.has(tok)) {
      tokens.push(<span key={i++} className="text-fuchsia-400">{tok}</span>)
    } else if (/^[A-Z]/.test(tok)) {
      tokens.push(<span key={i++} className="text-sky-300">{tok}</span>)
    } else if (tok === 'nn' || tok === 'torch' || tok === 'F' || tok === 'math' || tok === 'einops') {
      tokens.push(<span key={i++} className="text-amber-300">{tok}</span>)
    } else {
      tokens.push(<span key={i++} className="text-slate-200">{tok}</span>)
    }
  }
  if (lastIdx < code.length) {
    tokens.push(<span key={i++} className="text-slate-200">{code.slice(lastIdx)}</span>)
  }

  return (
    <div key={key} className="leading-6">
      {tokens}
      {comment && <span className="text-slate-500 italic">{comment}</span>}
      {line === '' && <span>&nbsp;</span>}
    </div>
  )
}

export default function CodeBlock({ code, title, lang = 'python' }: { code: string; title?: string; lang?: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }
  return (
    <div className="rounded-xl overflow-hidden border border-slate-700 shadow-lg my-4">
      <div className="flex items-center justify-between bg-slate-800 px-4 py-2">
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-red-400" />
          <span className="w-3 h-3 rounded-full bg-yellow-400" />
          <span className="w-3 h-3 rounded-full bg-green-400" />
          <span className="ml-3 text-xs text-slate-400 font-mono">{title || lang}</span>
        </div>
        <button onClick={copy} className="flex items-center gap-1 text-xs text-slate-400 hover:text-white transition-colors">
          {copied ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
          {copied ? '已复制' : '复制'}
        </button>
      </div>
      <pre className="bg-slate-900 p-4 overflow-x-auto text-[13px] font-mono">
        {code.split('\n').map((line, i) => highlightLine(line, i))}
      </pre>
    </div>
  )
}
