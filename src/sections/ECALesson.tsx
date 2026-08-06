import { LessonHeader, H3, P, StepBox, DiagramFrame, FlowNode, Arrow, ElbowArrow, PseudoCode } from '../components/course/Lesson'
import CodeBlock from '../components/course/CodeBlock'
import RefBox from '../components/course/RefBox'

const ecaCode = `import math
import torch
import torch.nn as nn

class ECABlock(nn.Module):
    """ECA: Efficient Channel Attention (CVPR 2020)
    与 SE 的三点不同:
      1) 不降维! 没有 C -> C/16 -> C 的瓶颈
      2) 不用全连接, 改用一个核大小为 k 的「一维卷积」
      3) k 根据通道数 C 自适应计算, 不用手调
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # 自适应核大小公式 (论文 Eq.(9)): 通道越多, 邻居范围越大
        #   k = | log2(C)/gamma + b/gamma | 再取最近的奇数
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1                    # 保证 k 是奇数(卷积核对称)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k,
                              padding=k // 2, bias=False)  # 整个模块只有 k 个参数!
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)              # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C) —— 把通道排成一列
        y = self.conv(y)                  # 一维卷积: 每个通道只和相邻 k 个通道交流
        y = self.sigmoid(y)               # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y.expand_as(x)         # 逐通道加权


if __name__ == '__main__':
    for c in [64, 128, 512, 2048]:
        m = ECABlock(c)
        k = m.conv.kernel_size[0]
        print(f'通道数 C={c:5d} -> 自适应卷积核 k={k}, 模块参数={sum(p.numel() for p in m.parameters())} 个')
    # C=64 -> k=3; C=128 -> k=5; C=512 -> k=5; C=2048 -> k=7  (与论文 Table 2 一致)

    x = torch.randn(4, 512, 7, 7)
    print(ECABlock(512)(x).shape)  # torch.Size([4, 512, 7, 7])`

export default function ECALesson() {
  return (
    <section className="mt-24">
      <LessonHeader id="eca" no="LESSON 2" title="ECA：只用 5 个参数的高效通道注意力" subtitle="Efficient Channel Attention · CVPR 2020" accent="emerald" />

      <H3 accent="emerald">一个类比：全员大会 vs 邻居开小会</H3>
      <P>
        SE 模块的全连接层像一场「全员大会」：每个通道都要和其他<b>所有</b>通道交换信息（C² 级别的连接），
        而且还要先把信息「压缩成摘要」（降维到 C/16）再「还原」——压缩过程必然会丢失信息。
        ECA 的作者做了实验（论文 Fig.2 与 Table 1）发现：<b>降维对通道注意力是有害的，而「每个通道只和邻近的几个通道交流」就完全够用</b>。
        于是 ECA 把「全员大会」改成了「邻居开小会」：每个通道只跟左右相邻的 k 个通道开一个一维小会，用一个<b>一维卷积</b>来实现。
      </P>

      <DiagramFrame viewBox="0 0 780 300" caption="图 2-1：SE 与 ECA 的对比——ECA 去掉了降维和全连接，仅用一个核大小为 k 的一维卷积">
        {/* SE */}
        <text x={185} y={25} textAnchor="middle" fontSize={13} fontWeight={700} fill="#64748b">SE（对比）</text>
        <FlowNode x={30} y={45} w={90} h={56} rx={10} fill="#64748b" label={"GAP\nC×1×1"} fontSize={11} />
        <Arrow x1={120} y1={73} x2={150} y2={73} />
        <FlowNode x={150} y={45} w={100} h={56} rx={10} fill="#f43f5e" label={"FC 降维\nC → C/16"} sub="信息损失！" fontSize={11} />
        <Arrow x1={250} y1={73} x2={280} y2={73} />
        <FlowNode x={280} y={45} w={100} h={56} rx={10} fill="#8b5cf6" label={"FC 升维\nC/16 → C"} fontSize={11} />
        <text x={185} y={135} textAnchor="middle" fontSize={11} fill="#e11d48">参数 2C²/16（ResNet-50 stage4: 约 52 万个）</text>

        {/* 分隔线 */}
        <line x1={410} y1={30} x2={410} y2={280} stroke="#e2e8f0" strokeWidth={2} strokeDasharray="6 4" />

        {/* ECA */}
        <text x={595} y={25} textAnchor="middle" fontSize={13} fontWeight={700} fill="#059669">ECA（本课主角）</text>
        <FlowNode x={440} y={45} w={90} h={56} rx={10} fill="#64748b" label={"GAP\nC×1×1"} fontSize={11} />
        <Arrow x1={530} y1={73} x2={560} y2={73} />
        <FlowNode x={560} y={45} w={130} h={56} rx={10} fill="#10b981" label={"1D 卷积 (核 k)\n不降维"} sub="只有 k 个参数" fontSize={11} />
        <text x={595} y={135} textAnchor="middle" fontSize={11} fill="#047857">参数只有 k 个（通常 3~9 个！）</text>

        {/* 一维卷积示意 */}
        <text x={595} y={175} textAnchor="middle" fontSize={12} fontWeight={700} fill="#334155">一维卷积「开小会」示意（k = 3）</text>
        {[0, 1, 2, 3, 4, 5, 6].map(i => (
          <g key={i}>
            <rect x={470 + i * 36} y={195} width={30} height={30} rx={6}
              fill={i === 3 ? '#10b981' : (i >= 2 && i <= 4 ? '#a7f3d0' : '#e2e8f0')}
              stroke={i >= 2 && i <= 4 ? '#10b981' : '#cbd5e1'} strokeWidth={1.5} />
            <text x={485 + i * 36} y={214} textAnchor="middle" fontSize={10} fill={i === 3 ? '#fff' : '#64748b'}>c{i + 1}</text>
          </g>
        ))}
        <path d="M470+108,0" />
        <ElbowArrow points={[[556, 195], [556, 178], [539, 178]]} color="#10b981" />
        <ElbowArrow points={[[592, 195], [592, 178], [575, 178]]} color="#10b981" />
        <text x={595} y={255} textAnchor="middle" fontSize={11} fill="#64748b">通道 c₄ 的权重只由 c₃、c₄、c₅ 三个邻居共同决定</text>
        <text x={595} y={275} textAnchor="middle" fontSize={11} fill="#64748b">所有通道共享同一组卷积核 → 参数极少</text>
      </DiagramFrame>

      <H3 accent="emerald">k 怎么定？——一个公式自动算</H3>
      <P>
        一维卷积核大小 k 决定了「开会的规模」：k 越大，每个通道能看到的邻居越多。ECA 的洞察是：
        <b>通道数 C 越多，需要的交互范围越大</b>，而且人眼感知上这种关系大致是指数级的。于是论文给出了自适应公式（γ=2，b=1）：
      </P>
      <div className="my-5 rounded-xl bg-emerald-50 border border-emerald-200 p-5 text-center">
        <div className="text-lg font-mono text-emerald-900">
          k = ψ(C) = | log₂(C)/γ + b/γ |<sub>odd</sub> &nbsp;=&nbsp; | log₂(C)/2 + 1/2 |<sub>odd</sub>
        </div>
        <div className="text-xs text-emerald-700 mt-2">下标 odd 表示「取最近的奇数」。例如 C=64 → k=3；C=512 → k=5；C=2048 → k=7（与论文 Table 2 完全一致）</div>
      </div>

      <H3 accent="emerald">伪代码流程</H3>
      <PseudoCode accent="emerald" lines={[
        { t: '输入: 特征图 F (C×H×W)' },
        { t: 'k = | log₂(C)/2 + 1/2 | 取最近奇数', note: '自适应卷积核大小' },
        { t: 'z = GAP(F)', note: '全局平均池化 → C 个数' },
        { t: 's = σ( Conv1D_k(z) )', note: '一维卷积跨通道交互，不降维，仅 k 个参数' },
        { t: 'F_out = F × s', note: '逐通道加权' },
        { t: '输出: F_out (C×H×W)' },
      ]} />

      <H3 accent="emerald">PyTorch 完整实现（与官方仓库一致）</H3>
      <CodeBlock code={ecaCode} title="eca.py —— ECA 完整实现（含自适应核验证）" />

      <StepBox icon="idea" title="小白理解要点" accent="emerald">
        <ul className="list-disc pl-5 space-y-1.5">
          <li><b>ECA 的全部可学习参数就是那一个一维卷积核</b>——k 通常为 3~9，所以整个模块只有 3~9 个参数，对比 SE 的几十万个，几乎等于「免费」。</li>
          <li><b>为什么必须保证不降维？</b>论文实验证明，降维（C→C/16）会显著损害通道注意力的学习效果；ECA 直接在原始 C 维上做卷积，完整保留通道信息。</li>
          <li><b>效果有多强？</b>论文报告：ResNet-50 加 ECA，参数只增加 <b>80 个</b>、计算量增加 0.00047 GFLOPs（约万分之 1.2），ImageNet Top-1 却提升超过 <b>2%</b>，超过 SE 和 CBAM（论文 Table 3）。</li>
          <li><b>适用场景：</b>对延迟/参数极度敏感的场景（移动端、实时检测），ECA 往往是第一选择。</li>
        </ul>
      </StepBox>

      <RefBox items={[
        { type: 'paper', label: 'ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks（arXiv:1910.03151）', desc: 'Wang et al., CVPR 2020，自适应核公式与实验数据出处', url: 'https://arxiv.org/abs/1910.03151' },
        { type: 'code', label: '官方代码仓库 BangguWu/ECANet', desc: '论文作者开源的 PyTorch 实现，本课代码与其保持一致', url: 'https://github.com/BangguWu/ECANet' },
        { type: 'paper', label: 'CVPR 2020 开放获取页面（openaccess.thecvf.com）', desc: '论文正式版 PDF 与补充材料', url: 'https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.html' },
      ]} />
    </section>
  )
}
