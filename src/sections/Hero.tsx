import { Flame, Zap, Compass, Layers, MousePointerClick, Lightbulb, GitBranch, Terminal, Link2, CheckCircle2 } from 'lucide-react'

const CARDS = [
  { icon: <Flame size={22} />, name: 'CBAM', full: 'Convolutional Block Attention Module', conf: 'ECCV 2018', color: 'from-orange-400 to-red-400', desc: '通道 + 空间 双重注意力，即插即用的经典模块' },
  { icon: <Zap size={22} />, name: 'ECA', full: 'Efficient Channel Attention', conf: 'CVPR 2020', color: 'from-emerald-400 to-teal-500', desc: '一维卷积实现的高效通道注意力，仅几十个参数' },
  { icon: <Compass size={22} />, name: 'CoordAtt', full: 'Coordinate Attention', conf: 'CVPR 2021', color: 'from-sky-400 to-blue-500', desc: '把位置坐标信息编码进通道注意力，移动端友好' },
  { icon: <Layers size={22} />, name: 'HRFormer', full: 'High-Resolution Transformer', conf: 'NeurIPS 2021', color: 'from-violet-400 to-purple-500', desc: '多分辨率并行 + 局部窗口自注意力的骨干网络' },
]

const FEATURES = [
  { icon: <Lightbulb size={16} />, text: '生活化类比讲透原理，零基础可学' },
  { icon: <MousePointerClick size={16} />, text: '手绘风格 SVG 原理结构图' },
  { icon: <GitBranch size={16} />, text: '逐行编号的伪代码流程图' },
  { icon: <Terminal size={16} />, text: '可直接运行的 PyTorch 完整实现' },
  { icon: <Link2 size={16} />, text: '每个结论附论文 / 官方仓库链接' },
]

export default function Hero() {
  return (
    <div id="hero" className="scroll-mt-24 pt-12 pb-16">
      <div className="rounded-3xl bg-gradient-to-br from-slate-900 via-slate-800 to-indigo-950 text-white p-8 md:p-12 shadow-2xl relative overflow-hidden">
        <div className="absolute -right-20 -top-20 w-72 h-72 rounded-full bg-orange-500/10 blur-2xl" />
        <div className="absolute -left-10 -bottom-24 w-72 h-72 rounded-full bg-violet-500/10 blur-2xl" />
        <div className="relative">
          <div className="inline-flex items-center gap-2 bg-white/10 rounded-full px-4 py-1.5 text-xs font-medium tracking-wider mb-6">
            <CheckCircle2 size={14} className="text-emerald-400" /> 零基础友好 · 附完整代码与权威参考
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold leading-tight tracking-tight">
            注意力机制与<br className="md:hidden" />高分辨率表征<span className="text-orange-400">实战课</span>
          </h1>
          <p className="mt-5 text-slate-300 leading-8 max-w-2xl text-[15px]">
            本课程面向刚入门深度学习的同学，用「生活类比 → 原理图解 → 伪代码 → PyTorch 代码」四步走的方式，
            讲透计算机视觉中四个里程碑式的模块：<b className="text-orange-300">CBAM</b>、<b className="text-emerald-300">ECA</b>、
            <b className="text-sky-300">CoordAtt</b> 与 <b className="text-violet-300">HRFormer</b>。
            学完后你可以把它们插进自己的 ResNet / MobileNet / 检测分割模型中。
          </p>
          <div className="mt-6 flex flex-wrap gap-x-6 gap-y-2 text-xs text-slate-400">
            {FEATURES.map((f, i) => (
              <span key={i} className="flex items-center gap-1.5">{f.icon}{f.text}</span>
            ))}
          </div>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-4 mt-8">
        {CARDS.map(c => (
          <div key={c.name} className="rounded-2xl bg-white border border-slate-200 p-5 shadow-sm hover:shadow-md transition-shadow">
            <div className="flex items-center gap-3">
              <div className={`w-11 h-11 rounded-xl bg-gradient-to-br ${c.color} text-white flex items-center justify-center shadow`}>{c.icon}</div>
              <div>
                <div className="font-extrabold text-slate-900">{c.name} <span className="ml-2 text-[10px] font-bold text-slate-500 bg-slate-100 rounded px-1.5 py-0.5">{c.conf}</span></div>
                <div className="text-[11px] text-slate-400 font-mono">{c.full}</div>
              </div>
            </div>
            <p className="mt-3 text-sm text-slate-600 leading-6">{c.desc}</p>
          </div>
        ))}
      </div>

      <div className="mt-10 rounded-2xl border border-amber-200 bg-amber-50 p-5 text-sm text-amber-900 leading-7">
        <b>学前准备：</b>只需要会一点 Python、知道「卷积神经网络（CNN）」和「特征图」是什么即可。
        如果你听说过 ResNet、知道 PyTorch 里 <code className="bg-amber-100 px-1 rounded">nn.Module</code> 的写法，那已经绰绰有余。
        课程中的每一个公式都会用图形和类比解释，不堆数学符号。
      </div>
    </div>
  )
}
