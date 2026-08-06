import { LessonHeader, H3, P, StepBox, DiagramFrame, FlowNode, Arrow, ElbowArrow, PseudoCode } from '../components/course/Lesson'
import CodeBlock from '../components/course/CodeBlock'
import RefBox from '../components/course/RefBox'

const hrformerCode = `import torch
import torch.nn as nn

# ---------------------------------------------------------------
# 教学版 HRFormer 核心: 局部窗口自注意力 + 卷积 FFN + 多分辨率交换
# 真实 HRFormer = HRNet 骨架, 把每个卷积块替换成下面的 Transformer 块
# ---------------------------------------------------------------

def window_partition(x, ws):
    """把特征图 (B, H, W, C) 切成不重叠的 ws×ws 小窗口"""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    # (B, 行窗口数, 列窗口数, ws, ws, C) -> 把窗口展平成序列
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)

def window_reverse(windows, ws, H, W):
    """窗口自注意力的逆操作: 把窗口拼回完整特征图"""
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    """局部窗口多头自注意力 (与 Swin 的窗口思想一致)
    全局自注意力是 HW×HW 的两两交互, 太贵;
    窗口内自注意力只有 ws²×ws², 计算量随分辨率线性增长
    """
    def __init__(self, dim, ws=7, num_heads=4):
        super().__init__()
        self.ws, self.num_heads = ws, num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        # 相对位置偏置: 告诉模型窗口内每个位置的相对方位
        self.rel_bias = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, num_heads))
        coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing='ij'))
        rel = (coords.flatten(1).T[:, None] - coords.flatten(1).T[None]) + ws - 1
        self.register_buffer('rel_idx', rel[..., 0] * (2 * ws - 1) + rel[..., 1])

    def forward(self, x):                        # x: (窗口数, ws*ws, C)
        n, N, C = x.shape
        qkv = self.qkv(x).reshape(n, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)      # Q·K^T / √d
        attn = attn + self.rel_bias[self.rel_idx.view(-1)].view(N, N, -1).permute(2, 0, 1)
        out = (attn.softmax(-1) @ v).transpose(1, 2).reshape(n, N, C)  # 注意力加权求和
        return self.proj(out)


class ConvFFN(nn.Module):
    """HRFormer 的特色 FFN: 两个 1×1 卷积之间夹一个 3×3 深度卷积
    把 CNN 的局部感受野「偷渡」进 Transformer, 弥补窗口边界的割裂
    """
    def __init__(self, dim, expand=4):
        super().__init__()
        hidden = dim * expand
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)  # 深度卷积
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):                        # x: (B, C, H, W) 卷积视角
        return self.fc2(self.act(self.dwconv(self.act(self.fc1(x)))))


class HRFormerBlock(nn.Module):
    """一个 HRFormer 基本块 = LayerNorm + 窗口自注意力 + 残差
                              + LayerNorm + ConvFFN    + 残差"""
    def __init__(self, dim, ws=7, num_heads=4):
        super().__init__()
        self.ws = ws
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, ws, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = ConvFFN(dim)

    def forward(self, x):                        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # ---- 窗口自注意力分支 (转成序列视角) ----
        t = x.permute(0, 2, 3, 1)                # (B, H, W, C)
        win = window_partition(self.norm1(t), self.ws)
        t = t + window_reverse(self.attn(win), self.ws, H, W)
        x = t.permute(0, 3, 1, 2)                # 残差后回到 (B, C, H, W)
        # ---- ConvFFN 分支 (卷积视角) ----
        t = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.ffn(t)


class ExchangeUnit(nn.Module):
    """多分辨率交换单元 (继承自 HRNet):
    高分辨率分支 -> 下采样后加到低分辨率分支
    低分辨率分支 -> 上采样后加到高分辨率分支"""
    def __init__(self, c_high, c_low):
        super().__init__()
        self.down = nn.Conv2d(c_high, c_low, 3, stride=2, padding=1)  # 高 -> 低
        self.up = nn.Conv2d(c_low, c_high, 1)                          # 低 -> 高 (再上采样)

    def forward(self, high, low):
        to_low = self.down(high)                                       # 下采样对齐分辨率
        to_high = nn.functional.interpolate(self.up(low), size=high.shape[-2:],
                                            mode='bilinear', align_corners=False)
        return high + to_high, low + to_low


if __name__ == '__main__':
    high = torch.randn(2, 32, 64, 64)     # 高分辨率分支 (细节)
    low = torch.randn(2, 64, 32, 32)      # 低分辨率分支 (语义)
    block_h, block_l = HRFormerBlock(32, ws=8), HRFormerBlock(64, ws=8)
    exch = ExchangeUnit(32, 64)
    high, low = block_h(high), block_l(low)   # 各自做 Transformer
    high, low = exch(high, low)               # 跨分辨率交换信息
    print(high.shape, low.shape)              # [2,32,64,64] [2,64,32,32]`

export default function HRFormerLesson() {
  return (
    <section className="mt-24">
      <LessonHeader id="hrformer" no="LESSON 4" title="HRFormer：高分辨率 + Transformer 的强强联合" subtitle="High-Resolution Transformer · NeurIPS 2021" accent="violet" />

      <H3 accent="violet">前情提要：它融合了两个「名门」的思想</H3>
      <P>
        HRFormer 是两个经典工作的结晶，先花一分钟认识它们：
      </P>
      <ul className="list-disc pl-6 space-y-2 text-slate-600 leading-7 text-[15px]">
        <li><b>HRNet（CVPR 2019）：</b>传统 CNN 会一路下采样，分辨率越降越低（细节丢失）。HRNet 反其道而行——<b>全程保留一条高分辨率分支</b>，并行的还有若干低分辨率分支，分支之间定期「交换信息」。这在人体姿态估计（找关键点）上大放异彩：关键点定位需要像素级的精确度。</li>
        <li><b>Transformer 自注意力：</b>让特征图上任意两个位置直接「对话」（Q·K 算相关性，加权 V 求和），建模长距离依赖的能力远超卷积。</li>
      </ul>
      <P>
        <b>HRFormer 的一句话定义：把 HRNet 骨架中的卷积块，替换成「局部窗口自注意力」的 Transformer 块。</b>
        既保留 HRNet 的高分辨率多分支结构（细节不丢），又获得 Transformer 的强表征能力。
      </P>

      <StepBox icon="idea" title="类比理解：多机位直播团队" accent="violet">
        把 HRFormer 想象成一个直播团队：<b>高清机位</b>（高分辨率分支）全程盯着舞台细节不眨眼；
        <b>远景机位</b>（低分辨率分支）把握全场调度（语义信息）。导演台定期把两个机位的画面<b>互相推流</b>（交换单元）。
        而每个机位内部的摄影师不再「一格一格扫」（卷积），而是<b>分片区互相对讲</b>（窗口自注意力）——片区内任意两个位置直接沟通，效率和理解力都更强。
      </StepBox>

      <H3 accent="violet">整体架构：多分辨率并行 + 交换</H3>
      <DiagramFrame viewBox="0 0 800 300" caption="图 4-1：HRFormer 多分辨率并行架构（继承 HRNet）——高分辨率分支贯穿始终，分支间反复交换信息">
        {/* 高分辨率分支 */}
        <text x={60} y={30} fontSize={12} fontWeight={700} fill="#7c3aed">高分辨率分支（细节）</text>
        {[0, 1, 2, 3].map(i => (
          <FlowNode key={i} x={40 + i * 185} y={45} w={130} h={60} rx={10} fill="#8b5cf6"
            label={"HRFormer Block ×N"} sub={`1/4 分辨率`} fontSize={11.5} />
        ))}
        {[0, 1, 2].map(i => (
          <Arrow key={i} x1={170 + i * 185} y1={75} x2={223 + i * 185} y2={75} color="#8b5cf6" />
        ))}
        {/* 低分辨率分支 */}
        <text x={40} y={155} fontSize={12} fontWeight={700} fill="#0ea5e9">低分辨率分支（语义，通道数 ×2）</text>
        {[1, 2, 3].map(i => (
          <FlowNode key={i} x={40 + i * 185} y={170} w={130} h={60} rx={10} fill="#0ea5e9"
            label={"HRFormer Block ×N"} sub={`1/8 分辨率`} fontSize={11.5} />
        ))}
        {[1, 2].map(i => (
          <Arrow key={i} x1={170 + i * 185} y1={200} x2={223 + i * 185} y2={200} color="#0ea5e9" />
        ))}
        {/* 交换单元 */}
        {[1, 2].map(i => (
          <g key={i}>
            <ElbowArrow points={[[105 + i * 185, 105], [105 + i * 185, 130], [90 + i * 185, 130], [90 + i * 185, 168]]} color="#f59e0b" dashed label="下采样" labelPos={[60 + i * 185, 145]} />
            <ElbowArrow points={[[140 + i * 185, 170], [140 + i * 185, 140], [125 + i * 185, 140], [125 + i * 185, 107]]} color="#10b981" dashed label="上采样" labelPos={[168 + i * 185, 145]} />
          </g>
        ))}
        {/* stem 说明 */}
        <text x={400} y={270} textAnchor="middle" fontSize={11.5} fill="#64748b">
          Stem：两个 3×3 stride=2 卷积先把图像降到 1/4 分辨率；分支随 stage 推进逐级增加（1/4 → 1/8 → 1/16 → 1/32）
        </text>
      </DiagramFrame>

      <H3 accent="violet">核心块拆解：窗口自注意力 + 卷积 FFN</H3>
      <P>
        每个 HRFormer Block 内部是经典 Transformer 的两段式结构，但有两个关键改造：
      </P>
      <ul className="list-disc pl-6 space-y-2 text-slate-600 leading-7 text-[15px]">
        <li><b>局部窗口自注意力（Local-Window Self-Attention）：</b>全局自注意力让所有位置两两交互，计算量是分辨率的<b>平方级</b>，高分辨率下根本跑不动。HRFormer 借鉴 Swin Transformer 的做法：把特征图切成 7×7 的小窗口，<b>只在窗口内部</b>做自注意力，计算量降为<b>线性</b>——这正是它敢保持高分辨率的底气。</li>
        <li><b>夹了深度卷积的 FFN：</b>标准 Transformer 的 FFN 是两层全连接；HRFormer 在中间夹了一个 3×3 <b>深度卷积（Depthwise Conv）</b>，把 CNN 的局部感受野注入 Transformer，同时缓解窗口之间信息不通的问题。</li>
      </ul>

      <DiagramFrame viewBox="0 0 800 240" caption="图 4-2：HRFormer Block 内部结构 + 窗口自注意力示意">
        {/* Block 结构 */}
        <FlowNode x={10} y={30} w={90} h={56} rx={10} fill="#64748b" label={"输入\nB×C×H×W"} fontSize={11} />
        <Arrow x1={100} y1={58} x2={130} y2={58} />
        <FlowNode x={130} y={30} w={120} h={56} rx={10} fill="#8b5cf6" label={"LayerNorm +\n窗口自注意力"} sub="窗口内 Q·K·V 交互" fontSize={11} />
        <Arrow x1={250} y1={58} x2={280} y2={58} label="残差 +" />
        <FlowNode x={280} y={30} w={140} h={56} rx={10} fill="#f59e0b" label={"LayerNorm + ConvFFN"} sub="1×1 → DWConv3×3 → 1×1" fontSize={11} />
        <Arrow x1={420} y1={58} x2={450} y2={58} label="残差 +" />
        <FlowNode x={450} y={30} w={90} h={56} rx={10} fill="#10b981" label={"输出\n形状不变"} fontSize={11} />
        {/* 窗口示意 */}
        <text x={240} y={135} textAnchor="middle" fontSize={12} fontWeight={700} fill="#334155">窗口自注意力：把特征图切成 7×7 窗口，窗口内部互相「对话」</text>
        <g>
          {[0, 1, 2, 3].map(r => [0, 1, 2, 3].map(c => (
            <rect key={`${r}${c}`} x={60 + c * 60} y={150 + r * 22} width={58} height={20} rx={3}
              fill={r === 1 && c === 1 ? '#8b5cf6' : '#ede9fe'} stroke="#c4b5fd" strokeWidth={1} />
          )))}
          <text x={179} y={172} textAnchor="middle" fontSize={10} fill="#fff">窗口内 49 个位置两两交互</text>
        </g>
        {/* 注意力公式 */}
        <g>
          <rect x={420} y={140} width={360} height={80} rx={10} fill="#f5f3ff" stroke="#c4b5fd" />
          <text x={600} y={168} textAnchor="middle" fontSize={13} fill="#5b21b6" fontFamily="monospace">
            Attention(Q,K,V) = softmax( Q·Kᵀ/√d + B )·V
          </text>
          <text x={600} y={192} textAnchor="middle" fontSize={11} fill="#7c3aed">
            Q: 查询（我想找什么） K: 键（我是什么） V: 值（我携带的信息） B: 相对位置偏置
          </text>
          <text x={600} y={210} textAnchor="middle" fontSize={10.5} fill="#64748b">
            相似度高的位置获得更高权重 → 任意距离的特征直接融合
          </text>
        </g>
      </DiagramFrame>

      <H3 accent="violet">伪代码流程</H3>
      <PseudoCode accent="violet" lines={[
        { t: '输入: 图像 I' },
        { t: 'x = Stem(I)', note: '两个 3×3 stride=2 卷积 → 1/4 分辨率高分辨率分支' },
        { t: 'for 每个 stage:', indent: 0 },
        { t: '每条分支: 重复 N 次 HRFormerBlock', indent: 1 },
        { t: 'x = x + WindowAttn( LN(x) )', indent: 2, note: '切 7×7 窗口，窗内做多头自注意力（含相对位置偏置）' },
        { t: 'x = x + ConvFFN( LN(x) )', indent: 2, note: '1×1 升维 → 3×3 深度卷积 → 1×1 降维' },
        { t: '新增一条分辨率减半、通道翻倍的分支', indent: 1, note: '1/4 → 1/8 → 1/16 → 1/32' },
        { t: '交换单元: 高分支下采样加到低分支; 低分支上采样加到高分支', indent: 1, note: '双向信息交换' },
        { t: '输出: 高分辨率分支特征 → 关键点检测 / 分割头等密集预测任务' },
      ]} />

      <H3 accent="violet">PyTorch 教学版实现（核心组件全解析）</H3>
      <P>
        下面代码实现了 HRFormer 的三大核心组件：<b>窗口自注意力（含相对位置偏置）</b>、<b>夹深度卷积的 FFN</b>、<b>多分辨率交换单元</b>，
        可以直接运行观察形状变化。工程级完整实现（含 shift 操作、多 stage 堆叠、FFN 激活等细节）请见官方仓库。
      </P>
      <CodeBlock code={hrformerCode} title="hrformer_core.py —— HRFormer 核心组件教学实现" />

      <StepBox icon="idea" title="小白理解要点" accent="violet">
        <ul className="list-disc pl-5 space-y-1.5">
          <li><b>为什么 HRFormer 适合姿态估计/分割？</b>这类「密集预测」任务要求逐像素输出，分辨率就是生命线——HRFormer 全程不降高分辨率分支，细节从头保留到尾。</li>
          <li><b>窗口自注意力 vs 全局自注意力：</b>H×W 的特征图，全局自注意力要算 (HW)² 对关系；切成 7×7 窗口后只需算 (HW/49)×49² = 49×HW 对，从平方级降到线性级。</li>
          <li><b>效果（论文数据）：</b>COCO 姿态估计上 HRFormer-B 以更少参数超过 HRNet-W48 约 1.3 AP；ADE20K 语义分割、COCO 检测上同样全面领先同级别 Swin Transformer。</li>
          <li><b>和前三个模块的关系：</b>CBAM/ECA/CoordAtt 是「外挂插件」，插进现有网络即可；HRFormer 是「整机架构」，需要从骨干网络层面搭建，二者是不同层次的工具。</li>
        </ul>
      </StepBox>

      <RefBox items={[
        { type: 'paper', label: 'HRFormer: High-Resolution Transformer for Dense Prediction（arXiv:2110.09408）', desc: 'Yuan et al., NeurIPS 2021，本课架构与实验数据出处', url: 'https://arxiv.org/abs/2110.09408' },
        { type: 'code', label: '官方代码仓库 HRNet/HRFormer（微软开源）', desc: '完整工程实现，含姿态估计/分割/检测配置与预训练权重', url: 'https://github.com/HRNet/HRFormer' },
        { type: 'paper', label: 'Deep High-Resolution Representation Learning for Human Pose Estimation（HRNet，arXiv:1908.07919）', desc: '多分辨率并行架构的出处，HRFormer 的骨架', url: 'https://arxiv.org/abs/1908.07919' },
        { type: 'paper', label: 'Swin Transformer（arXiv:2103.14030，ICCV 2021 最佳论文）', desc: '局部窗口自注意力与相对位置偏置的出处', url: 'https://arxiv.org/abs/2103.14030' },
      ]} />
    </section>
  )
}
