import { useEffect, useState } from 'react'
import { Sparkles, Flame, Zap, Compass, Layers, Scale, ChevronUp, BookMarked } from 'lucide-react'
import Hero from '../sections/Hero'
import Attention101 from '../sections/Attention101'
import CBAMLesson from '../sections/CBAMLesson'
import ECALesson from '../sections/ECALesson'
import CoordAttLesson from '../sections/CoordAttLesson'
import HRFormerLesson from '../sections/HRFormerLesson'
import Compare from '../sections/Compare'

const NAV = [
  { id: 'hero', label: '课程首页', icon: <BookMarked size={15} /> },
  { id: 'intro', label: '第 0 课 · 注意力机制入门', icon: <Sparkles size={15} /> },
  { id: 'cbam', label: '第 1 课 · CBAM', icon: <Flame size={15} /> },
  { id: 'eca', label: '第 2 课 · ECA', icon: <Zap size={15} /> },
  { id: 'coordatt', label: '第 3 课 · CoordAtt', icon: <Compass size={15} /> },
  { id: 'hrformer', label: '第 4 课 · HRFormer', icon: <Layers size={15} /> },
  { id: 'compare', label: '总结对比与选型', icon: <Scale size={15} /> },
]

export default function Home() {
  const [active, setActive] = useState('hero')
  const [showTop, setShowTop] = useState(false)

  useEffect(() => {
    const onScroll = () => {
      setShowTop(window.scrollY > 600)
      let current = 'hero'
      for (const n of NAV) {
        const el = document.getElementById(n.id)
        if (el && el.getBoundingClientRect().top < 160) current = n.id
      }
      setActive(current)
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <div className="min-h-screen bg-slate-50">
      {/* 侧边导航 */}
      <aside className="hidden lg:flex fixed left-0 top-0 bottom-0 w-64 flex-col bg-slate-900 text-slate-300 z-40">
        <div className="px-5 py-6 border-b border-slate-800">
          <div className="text-white font-bold text-lg leading-tight">注意力机制<br />实战教学课</div>
          <div className="text-xs text-slate-500 mt-1 font-mono">CBAM · ECA · CoordAtt · HRFormer</div>
        </div>
        <nav className="flex-1 py-4 overflow-y-auto">
          {NAV.map(n => (
            <button key={n.id} onClick={() => scrollTo(n.id)}
              className={`w-full flex items-center gap-3 px-5 py-3 text-sm transition-colors text-left ${active === n.id ? 'bg-slate-800 text-white border-r-2 border-orange-400' : 'hover:bg-slate-800/60'}`}>
              {n.icon}{n.label}
            </button>
          ))}
        </nav>
        <div className="px-5 py-4 text-[11px] text-slate-500 border-t border-slate-800">
          原理 · 图解 · 伪代码 · PyTorch 实现
        </div>
      </aside>

      {/* 移动端顶部导航 */}
      <div className="lg:hidden sticky top-0 z-40 bg-slate-900 text-white px-4 py-3 flex items-center gap-3 overflow-x-auto">
        {NAV.map(n => (
          <button key={n.id} onClick={() => scrollTo(n.id)}
            className={`whitespace-nowrap text-xs px-3 py-1.5 rounded-full ${active === n.id ? 'bg-orange-500' : 'bg-slate-800'}`}>
            {n.label.replace(/第 \d 课 · /, '')}
          </button>
        ))}
      </div>

      <main className="lg:ml-64">
        <div className="max-w-4xl mx-auto px-5 md:px-10 pb-32">
          <Hero />
          <Attention101 />
          <CBAMLesson />
          <ECALesson />
          <CoordAttLesson />
          <HRFormerLesson />
          <Compare />
        </div>
      </main>

      {showTop && (
        <button onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
          className="fixed bottom-6 right-6 z-50 w-11 h-11 rounded-full bg-slate-900 text-white shadow-xl flex items-center justify-center hover:bg-slate-700 transition-colors">
          <ChevronUp size={20} />
        </button>
      )}
    </div>
  )
}
