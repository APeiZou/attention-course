import { LessonHeader, H3, P, StepBox, DiagramFrame, FlowNode, Arrow, PseudoCode } from '../components/course/Lesson'
import CodeBlock from '../components/course/CodeBlock'
import RefBox from '../components/course/RefBox'

const seCode = `import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) 通道注意力 —— 注意力的"祖师爷"模块
    论文: Squeeze-and-Excitation Networks (CVPR 2018)
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)          # Squeeze: C×H×W -> C×1×1
        self.fc = nn.Sequential(                          # Excitation: 学习每个通道的重要性
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),                                 # 权重压到 0~1 之间
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)                   # 每个通道压成一个数
        y = self.fc(y).view(b, c, 1, 1)                   # 得到 c 个通道权重
        return x * y.expand_as(x)                         # 逐通道加权（重要通道放大）

# 试一下：输入 4 张图、64 通道、56×56 的特征图
x = torch.randn(4, 64, 56, 56)
print(SEBlock(64)(x).shape)   # torch.Size([4, 64, 56, 56]) —— 形状不变，即插即用`

export default function Attention101() {
  return (
    <section className="mt-20">
      <LessonHeader id="intro" no="LESSON 0" title="注意力机制入门：让网络学会「挑重点」" subtitle="Attention Mechanism Basics" accent="slate" />

      <H3 accent="slate">一个生活类比：鸡尾酒会效应</H3>
      <P>
        想象你在一个嘈杂的鸡尾酒会上，周围几十个人同时说话，但你依然能听清你朋友的声音——因为大脑自动<b>「调高」</b>了朋友的音量、
        <b>「调低」</b>了其他人的音量。这就是<b>注意力</b>：信息太多，算力有限，必须有选择地放大重要信息、抑制无关信息。
      </P>
      <P>
        神经网络也一样。卷积网络提取出的<b>特征图（Feature Map）</b>里，有的通道负责检测「猫的耳朵」，有的通道负责检测「背景纹理」；
        有的位置是目标主体，有的位置是无关背景。如果一视同仁地处理，网络就会浪费精力。注意力模块的作用，
        就是给特征图算出一套<b>权重</b>：重要的乘以接近 1 的数（放大），不重要的乘以接近 0 的数（抑制）。
      </P>

      <H3 accent="slate">先认识特征图：一个 C × H × W 的「魔方」</H3>
      <P>
        一张图片经过几层卷积后，会变成形状为 <code className="bg-slate-200 px-1.5 py-0.5 rounded text-sm font-mono">C × H × W</code> 的特征图：
        <b>H、W</b> 是空间尺寸（高、宽），<b>C</b> 是通道数。可以把每个通道想象成一张「专家图纸」——通道 1 专门画边缘，通道 2 专门画颜色块……
        注意力有两条天然的切入维度：
      </P>

      <DiagramFrame viewBox="0 0 760 300" caption="图 0-1：特征图的两个注意力维度——通道注意力回答「看什么」，空间注意力回答「看哪里」">
        {/* 特征图魔方 */}
        <g>
          {[0, 1, 2, 3, 4].map(i => (
            <g key={i}>
              <rect x={60 + i * 14} y={90 - i * 12} width={110} height={150} rx={4}
                fill={['#fde68a', '#fcd34d', '#fbbf24', '#f59e0b', '#d97706'][i]} stroke="#92400e" strokeWidth={1} opacity={0.9} />
            </g>
          ))}
          <text x={130} y={280} textAnchor="middle" fontSize={12} fill="#64748b">特征图 F：C × H × W</text>
          <text x={40} y={70} fontSize={11} fill="#92400e" fontWeight={700}>通道 C</text>
          <text x={215} y={180} fontSize={11} fill="#92400e" fontWeight={700}>空间 H × W</text>
        </g>
        {/* 通道注意力 */}
        <Arrow x1={210} y1={90} x2={330} y2={60} color="#f97316" />
        <FlowNode x={330} y={25} w={150} h={70} rx={12} fill="#f97316" label={"通道注意力\nChannel Attention"} sub="每个通道一个权重 (C 个数)" fontSize={12} />
        <text x={405} y={120} textAnchor="middle" fontSize={11.5} fill="#9a3412">「<tspan fontWeight={700}>什么</tspan>特征重要？」</text>
        {/* 空间注意力 */}
        <Arrow x1={210} y1={180} x2={330} y2={215} color="#0ea5e9" />
        <FlowNode x={330} y={180} w={150} h={70} rx={12} fill="#0ea5e9" label={"空间注意力\nSpatial Attention"} sub="每个位置一个权重 (H×W 个数)" fontSize={12} />
        <text x={405} y={275} textAnchor="middle" fontSize={11.5} fill="#075985">「<tspan fontWeight={700}>哪里</tspan>重要？」</text>
        {/* 汇总 */}
        <Arrow x1={480} y1={60} x2={590} y2={130} color="#94a3b8" />
        <Arrow x1={480} y1={215} x2={590} y2={150} color="#94a3b8" />
        <FlowNode x={590} y={110} w={150} h={70} rx={12} fill="#334155" label={"加权后的特征图\nF' = F × 权重"} sub="形状不变，内容被增强" fontSize={12} />
      </DiagramFrame>

      <H3 accent="slate">祖师爷模块：SE（Squeeze-and-Excitation）</H3>
      <P>
        2018 年 ImageNet 挑战赛冠军方案 SENet 提出的 SE 模块，是几乎所有「通道注意力」的祖师爷，CBAM 和 ECA 都是在它的基础上改进的。
        它的思想用一句话概括：<b>先把每个通道的全局信息压成一个数（Squeeze），再用一个小型全连接网络学习通道之间的重要性（Excitation），最后把权重乘回去（Scale）</b>。
      </P>

      <DiagramFrame viewBox="0 0 760 170" caption="图 0-2：SE 模块结构——Squeeze（全局平均池化）→ Excitation（两层全连接）→ Scale（逐通道相乘）">
        <FlowNode x={10} y={50} w={100} h={70} rx={10} fill="#64748b" label={"输入特征\nC × H × W"} fontSize={12} />
        <Arrow x1={110} y1={85} x2={150} y2={85} />
        <FlowNode x={150} y={50} w={120} h={70} rx={10} fill="#0ea5e9" label={"全局平均池化\nGAP"} sub="Squeeze" fontSize={12} />
        <Arrow x1={270} y1={85} x2={310} y2={85} />
        <FlowNode x={310} y={50} w={140} h={70} rx={10} fill="#8b5cf6" label={"FC → ReLU\nFC → Sigmoid"} sub="Excitation（降维再升维）" fontSize={12} />
        <Arrow x1={450} y1={85} x2={490} y2={85} />
        <FlowNode x={490} y={50} w={110} h={70} rx={10} fill="#f59e0b" label={"通道权重\nC × 1 × 1"} sub="每个值在 0~1 之间" fontSize={12} />
        <Arrow x1={600} y1={85} x2={640} y2={85} />
        <FlowNode x={640} y={50} w={110} h={70} rx={10} fill="#10b981" label={"逐通道相乘\n输出 C × H × W"} sub="Scale" fontSize={12} />
      </DiagramFrame>

      <H3 accent="slate">SE 的伪代码流程</H3>
      <PseudoCode accent="slate" lines={[
        { t: '输入: 特征图 F (形状 C×H×W)', note: '' },
        { t: 'z = GAP(F)', note: 'Squeeze：对每个通道做全局平均池化，得到 C 个数' },
        { t: 's = Sigmoid( FC2( ReLU( FC1(z) ) ) )', note: 'Excitation：先降维 C→C/16，再升维回 C' },
        { t: 'F_out = F × s', note: 'Scale：权重广播后逐通道相乘' },
        { t: '输出: F_out (形状仍为 C×H×W)', note: '即插即用，不改变形状' },
      ]} />

      <H3 accent="slate">SE 的 PyTorch 实现（20 行，先跑起来）</H3>
      <CodeBlock code={seCode} title="se_block.py —— SE 通道注意力完整实现" />

      <StepBox icon="idea" title="小白理解要点" accent="slate">
        <ul className="list-disc pl-5 space-y-1.5">
          <li><b>为什么先降维再升维？</b>压缩参数（C² → 2C²/16），并制造一个「瓶颈」迫使网络学习通道间最重要的关系。</li>
          <li><b>Sigmoid 是什么？</b>把任意实数压到 0~1 的 S 形曲线，输出正好可以当「音量旋钮」。</li>
          <li><b>SE 的局限：</b>只看通道、不看位置；而且「降维」这一步会丢失通道信息——这正是后面 ECA 要改进的地方。</li>
        </ul>
      </StepBox>

      <StepBox icon="map" title="课程地图：四个模块分别解决什么问题" accent="slate">
        <ul className="list-disc pl-5 space-y-1.5">
          <li><b className="text-orange-600">CBAM</b>：SE 只看「什么重要」，CBAM 补上「哪里重要」——通道注意力 + 空间注意力串联。</li>
          <li><b className="text-emerald-600">ECA</b>：SE 的全连接降维太重且有副作用，ECA 用一个一维卷积搞定，几乎零参数。</li>
          <li><b className="text-sky-600">CoordAtt</b>：全局池化把位置信息抹掉了，CoordAtt 沿横竖两个方向分别池化，把坐标信息编码进注意力。</li>
          <li><b className="text-violet-600">HRFormer</b>：跳出 CNN，用 Transformer 的自注意力做多分辨率并行表征，适合关键点检测、分割等密集预测任务。</li>
        </ul>
      </StepBox>

      <RefBox items={[
        { type: 'paper', label: 'Squeeze-and-Excitation Networks（SE 论文，CVPR 2018）', desc: '通道注意力的开山之作，arXiv:1709.01507', url: 'https://arxiv.org/abs/1709.01507' },
        { type: 'code', label: 'SENet 官方 Caffe 代码仓库（hujie-frank/SENet）', desc: 'SE 模块官方实现', url: 'https://github.com/hujie-frank/SENet' },
        { type: 'paper', label: 'Attention Is All You Need（Transformer 自注意力，NeurIPS 2017）', desc: '自注意力机制的源头，arXiv:1706.03762，第 4 课 HRFormer 的基础', url: 'https://arxiv.org/abs/1706.03762' },
      ]} />
    </section>
  )
}
