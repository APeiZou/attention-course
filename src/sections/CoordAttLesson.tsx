import { LessonHeader, H3, P, StepBox, DiagramFrame, FlowNode, ElbowArrow, PseudoCode } from '../components/course/Lesson'
import CodeBlock from '../components/course/CodeBlock'
import RefBox from '../components/course/RefBox'

const caCode = `import torch
import torch.nn as nn

class CoordAtt(nn.Module):
    """Coordinate Attention (CVPR 2021)
    把「全局池化」拆成两个方向的「一维池化」:
      - 沿水平方向池化 -> 保留每一行的坐标信息
      - 沿垂直方向池化 -> 保留每一列的坐标信息
    因此注意力图既懂通道, 又知道目标「在哪一行哪一列」
    """
    def __init__(self, channels, reduction=32):
        super().__init__()
        mid = max(8, channels // reduction)          # 中间通道数, 至少为 8
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (B, C, H, W) -> (B, C, H, 1) 沿宽方向聚合
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (B, C, H, W) -> (B, C, 1, W) 沿高方向聚合
        self.conv1 = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish(inplace=True)        # 官方实现用的轻量激活 h-swish
        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x)                          # (B, C, H, 1) —— 每一行的摘要
        x_w = self.pool_w(x).permute(0, 1, 3, 2)      # (B, C, W, 1) —— 每一列的摘要(转置对齐)
        # 沿「行」方向拼接, 让横竖信息在同一个特征里交汇
        y = torch.cat([x_h, x_w], dim=2)              # (B, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))         # 1×1 卷积降维 + 非线性
        x_h, x_w = torch.split(y, [h, w], dim=2)      # 重新拆回两个方向
        x_w = x_w.permute(0, 1, 3, 2)                 # (B, mid, 1, W) 转回去
        a_h = self.conv_h(x_h).sigmoid()              # (B, C, H, 1) —— 行注意力
        a_w = self.conv_w(x_w).sigmoid()              # (B, C, 1, W) —— 列注意力
        return x * a_h * a_w                          # 双重加权: 行 × 列, 交点即目标


if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    ca = CoordAtt(64)
    print(ca(x).shape)  # torch.Size([2, 64, 32, 32])
    print('参数量:', sum(p.numel() for p in ca.parameters()))`

export default function CoordAttLesson() {
  return (
    <section className="mt-24">
      <LessonHeader id="coordatt" no="LESSON 3" title="CoordAtt：会看「坐标」的注意力" subtitle="Coordinate Attention · CVPR 2021" accent="sky" />

      <H3 accent="sky">一个类比：把地图揉成团 vs 按行按列做摘要</H3>
      <P>
        SE 和 ECA 的全局平均池化，相当于把整张地图<b>揉成一个纸团</b>：你知道这座城里「有猫」，但猫在哪个街区？不知道——位置信息被抹掉了。
        CBAM 的空间注意力能定位，但只靠 7×7 卷积看局部，看不到<b>长距离</b>的空间关系。
        CoordAtt（Coordinate Attention，坐标注意力）的做法像一位认真的档案管理员：<b>把地图按「行」和按「列」分别做摘要</b>——
        第 5 行有多活跃？第 12 列有多活跃？行摘要和列摘要一交汇，目标的坐标就被精确地「十字锁定」了。
      </P>

      <DiagramFrame viewBox="0 0 780 210" caption="图 3-1：行注意力 × 列注意力 = 十字锁定目标位置">
        {/* 热力图示意 */}
        <g>
          <rect x={40} y={40} width={130} height={130} rx={8} fill="#f1f5f9" stroke="#cbd5e1" />
          <ellipse cx={105} cy={105} rx={28} ry={20} fill="#0ea5e9" opacity={0.55} />
          <ellipse cx={105} cy={105} rx={14} ry={10} fill="#0369a1" opacity={0.8} />
          <text x={105} y={190} textAnchor="middle" fontSize={11.5} fill="#64748b">输入特征：目标在某个位置</text>
        </g>
        {/* 行注意力条 */}
        <g>
          <rect x={230} y={40} width={26} height={130} rx={4} fill="#e0f2fe" stroke="#7dd3fc" />
          <rect x={230} y={88} width={26} height={34} rx={4} fill="#0ea5e9" />
          <text x={243} y={190} textAnchor="middle" fontSize={11.5} fill="#64748b">行注意力 a_h</text>
          <text x={243} y={30} textAnchor="middle" fontSize={11} fill="#0369a1" fontWeight={700}>C×H×1</text>
        </g>
        <text x={285} y={110} fontSize={18} fill="#64748b" textAnchor="middle">×</text>
        {/* 列注意力条 */}
        <g>
          <rect x={310} y={40} width={130} height={26} rx={4} fill="#e0f2fe" stroke="#7dd3fc" />
          <rect x={358} y={40} width={34} height={26} rx={4} fill="#0ea5e9" />
          <text x={375} y={90} textAnchor="middle" fontSize={11.5} fill="#64748b">列注意力 a_w</text>
          <text x={375} y={30} textAnchor="middle" fontSize={11} fill="#0369a1" fontWeight={700}>C×1×W</text>
        </g>
        <text x={468} y={110} fontSize={18} fill="#64748b" textAnchor="middle">=</text>
        {/* 结果 */}
        <g>
          <rect x={500} y={40} width={130} height={130} rx={8} fill="#f1f5f9" stroke="#cbd5e1" />
          <rect x={500} y={88} width={130} height={34} fill="#7dd3fc" opacity={0.35} />
          <rect x={548} y={40} width={34} height={130} fill="#7dd3fc" opacity={0.35} />
          <rect x={548} y={88} width={34} height={34} fill="#0369a1" opacity={0.85} />
          <text x={565} y={190} textAnchor="middle" fontSize={11.5} fill="#64748b">行 × 列 → 精确坐标定位</text>
        </g>
        <text x={690} y={80} fontSize={11.5} fill="#0369a1" fontWeight={700}>「第 i 行且第 j 列」</text>
        <text x={690} y={100} fontSize={11.5} fill="#64748b">交叉处响应最强</text>
        <text x={690} y={120} fontSize={11.5} fill="#64748b">同时保留长距离依赖</text>
      </DiagramFrame>

      <H3 accent="sky">结构拆解：三步走</H3>
      <P>
        CoordAtt 的完整结构分三步：<b>① 坐标信息嵌入</b>——用 (H,1) 和 (1,W) 两个池化核沿两个方向聚合特征；
        <b>② 坐标注意力生成</b>——把两个方向的结果拼接，经 1×1 卷积降维、BatchNorm、非线性激活后重新拆开，各自用 1×1 卷积升回 C 通道并过 Sigmoid；
        <b>③ 加权</b>——行注意力与列注意力同时乘到输入上。整个模块只有 1×1 卷积，非常轻量，专门为移动端设计。
      </P>

      <DiagramFrame viewBox="0 0 800 320" caption="图 3-2：CoordAtt 完整数据流（与论文 Fig.2 对应）">
        <FlowNode x={10} y={125} w={100} h={70} rx={10} fill="#64748b" label={"输入 X\nC × H × W"} fontSize={12} />
        {/* 两个池化 */}
        <ElbowArrow points={[[110, 150], [140, 150], [140, 70], [170, 70]]} color="#0ea5e9" />
        <ElbowArrow points={[[110, 170], [140, 170], [140, 260], [170, 260]]} color="#38bdf8" />
        <FlowNode x={170} y={35} w={140} h={70} rx={10} fill="#0ea5e9" label={"X 方向池化 (H,1)"} sub="输出 C×H×1：每行一个摘要" fontSize={11.5} />
        <FlowNode x={170} y={225} w={140} h={70} rx={10} fill="#38bdf8" label={"Y 方向池化 (1,W)"} sub="输出 C×1×W：每列一个摘要" fontSize={11.5} />
        {/* 拼接 */}
        <ElbowArrow points={[[310, 70], [340, 70], [340, 145], [370, 145]]} color="#0ea5e9" />
        <ElbowArrow points={[[310, 260], [340, 260], [340, 175], [370, 175]]} color="#38bdf8" label="转置对齐" labelPos={[330, 205]} />
        <FlowNode x={370} y={125} w={130} h={70} rx={10} fill="#8b5cf6" label={"拼接 + 1×1 卷积\n降维 C → C/r"} sub="BatchNorm + h-swish" fontSize={11.5} />
        {/* 拆分 */}
        <ElbowArrow points={[[500, 145], [530, 145], [530, 70], [560, 70]]} color="#94a3b8" />
        <ElbowArrow points={[[500, 175], [530, 175], [530, 260], [560, 260]]} color="#94a3b8" label="Split 拆分" labelPos={[530, 205]} />
        <FlowNode x={560} y={35} w={110} h={70} rx={10} fill="#f59e0b" label={"1×1 卷积 + σ"} sub="行注意力 a_h: C×H×1" fontSize={11.5} />
        <FlowNode x={560} y={225} w={110} h={70} rx={10} fill="#f59e0b" label={"1×1 卷积 + σ"} sub="列注意力 a_w: C×1×W" fontSize={11.5} />
        {/* 加权 */}
        <ElbowArrow points={[[670, 70], [700, 70], [700, 130], [725, 130]]} color="#f59e0b" />
        <ElbowArrow points={[[670, 260], [700, 260], [700, 200], [725, 200]]} color="#f59e0b" />
        <FlowNode x={725} y={130} w={65} h={70} rx={10} fill="#10b981" label={"输出\nX·a_h·a_w"} fontSize={11} />
      </DiagramFrame>

      <H3 accent="sky">伪代码流程</H3>
      <PseudoCode accent="sky" lines={[
        { t: '输入: 特征图 X (C×H×W)' },
        { t: 'z_h = AvgPool_H(X)', note: '池化核 (H,1)：沿宽聚合 → C×H×1' },
        { t: 'z_w = AvgPool_W(X).转置', note: '池化核 (1,W)：沿高聚合 → C×W×1' },
        { t: 'f = δ( BN( Conv1×1( concat(z_h, z_w) ) ) )', note: '拼接后降维到 C/r，δ 为 h-swish' },
        { t: 'f_h, f_w = split(f, [H, W])', note: '重新拆回两个方向' },
        { t: 'a_h = σ( Conv1×1(f_h) )', note: '行注意力 C×H×1' },
        { t: 'a_w = σ( Conv1×1(f_w.转置) )', note: '列注意力 C×1×W' },
        { t: '输出: X × a_h × a_w', note: '行×列十字加权' },
      ]} />

      <H3 accent="sky">PyTorch 完整实现（与官方实现一致）</H3>
      <CodeBlock code={caCode} title="coord_att.py —— Coordinate Attention 完整实现" />

      <StepBox icon="idea" title="小白理解要点" accent="sky">
        <ul className="list-disc pl-5 space-y-1.5">
          <li><b>为什么先拼接再拆开？</b>拼接让行、列信息在共享的 1×1 卷积里「碰头交流」，一次卷积同时建模两个方向的通道关系，比分开做更高效。</li>
          <li><b>和 CBAM 空间注意力的区别：</b>CBAM 的 7×7 卷积只能看局部邻域；CoordAtt 的行/列摘要覆盖整条边，能捕获<b>长距离</b>依赖。</li>
          <li><b>效果：</b>论文报告 MobileNetV2 上，CoordAtt 以与 SE 几乎相同的计算量，ImageNet Top-1 提升 1.3%，且在检测、分割等下游任务上增益更明显（因为定位信息对它们更重要）。</li>
          <li><b>插入位置：</b>官方推荐插入 MobileNetV2 的倒残差块（Inverted Residual Block）中 SE 所在的位置，直接替换。</li>
        </ul>
      </StepBox>

      <RefBox items={[
        { type: 'paper', label: 'Coordinate Attention for Efficient Mobile Network Design（arXiv:2103.02907）', desc: 'Hou et al., CVPR 2021，结构与实验数据出处', url: 'https://arxiv.org/abs/2103.02907' },
        { type: 'code', label: '官方代码仓库 Andrew-Qibin/CoordAttention', desc: '论文作者开源实现（本课代码与其结构一致）', url: 'https://github.com/Andrew-Qibin/CoordAttention' },
        { type: 'paper', label: 'CVPR 2021 开放获取页面（openaccess.thecvf.com）', desc: '论文正式版 PDF', url: 'https://openaccess.thecvf.com/content/CVPR2021/html/Hou_Coordinate_Attention_for_Efficient_Mobile_Network_Design_CVPR_2021_paper.html' },
      ]} />
    </section>
  )
}
