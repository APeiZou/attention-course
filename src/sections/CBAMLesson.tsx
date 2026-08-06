import { LessonHeader, H3, P, StepBox, DiagramFrame, FlowNode, Arrow, ElbowArrow, PseudoCode } from '../components/course/Lesson'
import CodeBlock from '../components/course/CodeBlock'
import RefBox from '../components/course/RefBox'

const cbamCode = `import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """CBAM 第一步：通道注意力 —— 判断「什么特征重要」
    核心技巧：平均池化 + 最大池化 双管齐下，再共用一个 MLP
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)   # 全局平均池化：看整体趋势
        self.max_pool = nn.AdaptiveMaxPool2d(1)   # 全局最大池化：抓最强响应
        # 两个池化结果共用的 MLP（先降维再升维，论文中 reduction=16）
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(b, c))   # (B, C)
        max_out = self.mlp(self.max_pool(x).view(b, c))   # (B, C)
        weight = self.sigmoid(avg_out + max_out)          # 两条路的结果相加再归一化
        return x * weight.view(b, c, 1, 1)                # 逐通道加权


class SpatialAttention(nn.Module):
    """CBAM 第二步：空间注意力 —— 判断「哪里重要」
    核心技巧：沿通道维度做平均/最大池化，拼成双通道，再用 7×7 卷积看大范围
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)      # 通道平均: (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)    # 通道最大: (B, 1, H, W)
        feat = torch.cat([avg_out, max_out], dim=1)       # 拼成 2 通道
        weight = self.sigmoid(self.conv(feat))            # 7×7 卷积 -> (B, 1, H, W)
        return x * weight                                 # 逐位置加权


class CBAM(nn.Module):
    """CBAM 完整模块 = 通道注意力 -> 空间注意力（串联，顺序不能反）
    论文: CBAM: Convolutional Block Attention Module (ECCV 2018)
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)   # 先回答 What
        x = self.spatial_att(x)   # 再回答 Where
        return x


if __name__ == '__main__':
    x = torch.randn(4, 64, 56, 56)
    cbam = CBAM(64)
    print(cbam(x).shape)  # torch.Size([4, 64, 56, 56])
    print('参数量:', sum(p.numel() for p in cbam.parameters()))  # 约 8.5K，非常轻`

export default function CBAMLesson() {
  return (
    <section className="mt-24">
      <LessonHeader id="cbam" no="LESSON 1" title="CBAM：通道 + 空间的双重聚光灯" subtitle="Convolutional Block Attention Module · ECCV 2018" accent="orange" />

      <H3 accent="orange">一个类比：舞台灯光师的两步操作</H3>
      <P>
        想象一位舞台灯光师：第一步，他决定<b>用哪些颜色的灯</b>（暖光突出主角，冷光留给背景）——这相当于在<b>通道</b>维度上做选择；
        第二步，他决定<b>把光打在舞台的哪个位置</b>（追光灯跟着主角走）——这相当于在<b>空间</b>维度上做选择。
        CBAM 就是这位灯光师：先用<b>通道注意力（Channel Attention）</b>回答「<b>什么（What）</b>特征重要」，
        再用<b>空间注意力（Spatial Attention）</b>回答「<b>哪里（Where）</b>重要」，两步<b>串联</b>，依次对特征图加权。
      </P>

      <DiagramFrame viewBox="0 0 780 150" caption="图 1-1：CBAM 总览——通道注意力与空间注意力串联（论文消融实验证明「先通道后空间」效果最好）">
        <FlowNode x={10} y={40} w={110} h={70} rx={10} fill="#64748b" label={"输入特征 F\nC × H × W"} fontSize={12} />
        <Arrow x1={120} y1={75} x2={165} y2={75} color="#94a3b8" />
        <FlowNode x={165} y={40} w={150} h={70} rx={10} fill="#f97316" label={"通道注意力模块\nM_c(F) ∈ R^{C×1×1}"} sub="回答 What —— 什么重要" fontSize={12} />
        <Arrow x1={315} y1={75} x2={360} y2={75} color="#94a3b8" label="逐通道相乘 ⊗" />
        <FlowNode x={360} y={40} w={110} h={70} rx={10} fill="#fdba74" textFill="#7c2d12" label={"中间特征 F'"} fontSize={12} />
        <Arrow x1={470} y1={75} x2={515} y2={75} color="#94a3b8" />
        <FlowNode x={515} y={40} w={150} h={70} rx={10} fill="#0ea5e9" label={"空间注意力模块\nM_s(F') ∈ R^{1×H×W}"} sub="回答 Where —— 哪里重要" fontSize={12} />
        <Arrow x1={665} y1={75} x2={710} y2={75} color="#94a3b8" label="逐位置相乘 ⊗" labelOffset={-8} />
        <FlowNode x={710} y={40} w={60} h={70} rx={10} fill="#10b981" label={"F''"} fontSize={13} />
      </DiagramFrame>

      <H3 accent="orange">通道注意力：为什么平均池化和最大池化要一起用？</H3>
      <P>
        只看平均池化，相当于问「这个通道整体活跃吗？」；只看最大池化，相当于问「这个通道有没有出现过强烈响应？」——
        一个看整体、一个抓亮点，信息是<b>互补</b>的。CBAM 把两路结果送进<b>同一个共享 MLP</b>（先降维 16 倍再升回来），
        最后相加、过 Sigmoid，得到 C 个通道权重。
      </P>

      <DiagramFrame viewBox="0 0 780 260" caption="图 1-2：CBAM 通道注意力内部结构（AvgPool 与 MaxPool 共享同一个 MLP）">
        <FlowNode x={10} y={95} w={110} h={70} rx={10} fill="#64748b" label={"输入特征 F\nC × H × W"} fontSize={12} />
        <ElbowArrow points={[[120, 115], [160, 115], [160, 55], [190, 55]]} color="#f97316" />
        <ElbowArrow points={[[120, 145], [160, 145], [160, 205], [190, 205]]} color="#0ea5e9" />
        <FlowNode x={190} y={20} w={130} h={70} rx={10} fill="#f97316" label={"全局平均池化\nAvgPool → C×1×1"} sub="整体有多活跃" fontSize={11.5} />
        <FlowNode x={190} y={170} w={130} h={70} rx={10} fill="#0ea5e9" label={"全局最大池化\nMaxPool → C×1×1"} sub="最强响应有多亮" fontSize={11.5} />
        <Arrow x1={320} y1={55} x2={370} y2={55} color="#f97316" />
        <Arrow x1={320} y1={205} x2={370} y2={205} color="#0ea5e9" />
        <FlowNode x={370} y={95} w={150} h={70} rx={10} fill="#8b5cf6" label={"共享 MLP\nC → C/16 → C"} sub="两路共用它（参数省一半）" fontSize={12} />
        <ElbowArrow points={[[370, 55], [345, 55], [345, 130], [368, 130]]} color="#f97316" />
        <ElbowArrow points={[[370, 205], [345, 205], [345, 140], [368, 140]]} color="#0ea5e9" />
        <FlowNode x={520} y={110} w={50} h={50} rx={25} fill="#f59e0b" label="+" fontSize={16} />
        <Arrow x1={570} y1={135} x2={615} y2={135} color="#94a3b8" />
        <FlowNode x={615} y={100} w={70} h={70} rx={10} fill="#f59e0b" label={"Sigmoid\nσ"} fontSize={12} />
        <Arrow x1={685} y1={135} x2={725} y2={135} color="#94a3b8" />
        <FlowNode x={725} y={100} w={50} h={70} rx={10} fill="#10b981" label={"M_c"} fontSize={12} />
      </DiagramFrame>

      <H3 accent="orange">空间注意力：为什么用 7×7 大卷积核？</H3>
      <P>
        通道注意力做完后，CBAM 换一个视角：沿<b>通道轴</b>做平均池化和最大池化，把 C 个通道压成 2 张「位置图」（每张 H×W），
        拼接成 2 通道后用一个 <b>7×7 卷积</b>生成空间权重图。为什么卷积核要这么大？
        因为卷积核多大，「视野」就多大——7×7 意味着判断某个位置重不重要时，能看到周围 3 个像素范围内的上下文。
        论文消融实验表明 7×7 优于 3×3，再大（9×9 以上）收益下降且计算变贵。
      </P>

      <DiagramFrame viewBox="0 0 780 130" caption="图 1-3：CBAM 空间注意力内部结构">
        <FlowNode x={10} y={30} w={110} h={70} rx={10} fill="#fdba74" textFill="#7c2d12" label={"输入特征 F'\nC × H × W"} fontSize={12} />
        <Arrow x1={120} y1={65} x2={165} y2={65} color="#94a3b8" />
        <FlowNode x={165} y={30} w={170} h={70} rx={10} fill="#f97316" label={"通道维度 AvgPool + MaxPool"} sub="各得 1 × H × W，拼成 2 × H × W" fontSize={11.5} />
        <Arrow x1={335} y1={65} x2={380} y2={65} color="#94a3b8" />
        <FlowNode x={380} y={30} w={130} h={70} rx={10} fill="#8b5cf6" label={"7×7 卷积\n2 → 1 通道"} sub="大视野看空间上下文" fontSize={12} />
        <Arrow x1={510} y1={65} x2={555} y2={65} color="#94a3b8" />
        <FlowNode x={555} y={30} w={90} h={70} rx={10} fill="#f59e0b" label={"Sigmoid σ"} sub="M_s: 1×H×W" fontSize={12} />
        <Arrow x1={645} y1={65} x2={690} y2={65} color="#94a3b8" />
        <FlowNode x={690} y={30} w={80} h={70} rx={10} fill="#10b981" label={"F' ⊗ M_s\n= F''"} fontSize={11.5} />
      </DiagramFrame>

      <H3 accent="orange">伪代码流程</H3>
      <PseudoCode accent="orange" lines={[
        { t: '输入: 特征图 F (C×H×W)' },
        { t: '─── 第一步: 通道注意力 ───' },
        { t: 'avg = MLP( AvgPool(F) )', indent: 1, note: '全局平均池化 → 共享 MLP' },
        { t: 'mx  = MLP( MaxPool(F) )', indent: 1, note: '全局最大池化 → 同一个 MLP' },
        { t: "M_c = σ(avg + mx)", indent: 1, note: 'C 个通道权重' },
        { t: "F'  = F ⊗ M_c", indent: 1, note: '逐通道加权' },
        { t: '─── 第二步: 空间注意力 ───' },
        { t: "a = ChannelAvg(F'); m = ChannelMax(F')", indent: 1, note: '沿通道轴池化，各得 1×H×W' },
        { t: 'M_s = σ( Conv7×7( concat(a, m) ) )', indent: 1, note: '拼接成 2 通道 → 7×7 卷积' },
        { t: "F'' = F' ⊗ M_s", indent: 1, note: '逐位置加权' },
        { t: "输出: F'' (C×H×W)" },
      ]} />

      <H3 accent="orange">PyTorch 完整实现</H3>
      <CodeBlock code={cbamCode} title="cbam.py —— CBAM 完整实现（可直接运行）" />

      <StepBox icon="idea" title="小白理解要点" accent="orange">
        <ul className="list-disc pl-5 space-y-1.5">
          <li><b>共享 MLP 是精髓：</b>两条池化路径用同一个 MLP，既省参数，又让两路信息在同一空间中比较。</li>
          <li><b>为什么是「串联」不是「并联」？</b>论文 Table 4 的消融实验表明，先通道后空间的串联结构（S-C 顺序反了也更差）效果最好。</li>
          <li><b>插在哪里？</b>论文做法是把 CBAM 插在 ResNet 每个残差块的<b>残差输出之后、shortcut 相加之前</b>；参数量仅增加约 0.4%。</li>
          <li><b>效果：</b>ImageNet 上 ResNet-50 + CBAM，Top-1 错误率从 24.56% 降到 22.66%（论文 Table 3）。</li>
        </ul>
      </StepBox>

      <RefBox items={[
        { type: 'paper', label: 'CBAM: Convolutional Block Attention Module（arXiv:1807.06521）', desc: 'Woo et al., ECCV 2018，本课所有结构图与实验数据的原始出处', url: 'https://arxiv.org/abs/1807.06521' },
        { type: 'code', label: '官方代码仓库 Jongchan/attention-module', desc: '论文作者开源的 BAM/CBAM 实现（PyTorch/TensorFlow/MXNet 等）', url: 'https://github.com/Jongchan/attention-module' },
        { type: 'paper', label: 'ECCV 2018 论文集官方页面（Springer LNCS 11211）', desc: 'DOI: 10.1007/978-3-030-01234-2_1', url: 'https://link.springer.com/chapter/10.1007/978-3-030-01234-2_1' },
      ]} />
    </section>
  )
}
