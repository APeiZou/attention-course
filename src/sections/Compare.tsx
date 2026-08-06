import { LessonHeader, H3, P, StepBox } from '../components/course/Lesson'
import CodeBlock from '../components/course/CodeBlock'
import RefBox from '../components/course/RefBox'

const plugCode = `import torch
import torch.nn as nn
# from cbam import CBAM        # 第 1 课
# from eca import ECABlock     # 第 2 课
# from coord_att import CoordAtt  # 第 3 课

class BasicBlockWithAttention(nn.Module):
    """演示：把任意注意力模块插进 ResNet 的 BasicBlock
    原则: 注意力作用于「残差分支的输出」，再与 shortcut 相加
    """
    def __init__(self, channels, att_module):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.att = att_module            # <--- 想换哪个注意力就换哪个

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.att(out)              # 残差分支输出先做注意力加权
        return self.relu(out + identity)

# 用法（三选一，一行切换）：
# blk = BasicBlockWithAttention(64, CBAM(64))
# blk = BasicBlockWithAttention(64, ECABlock(64))
# blk = BasicBlockWithAttention(64, CoordAtt(64))
# blk(torch.randn(2, 64, 56, 56)).shape  ->  torch.Size([2, 64, 56, 56])`

const ROWS: [string, string, string, string, string][] = [
  ['发表', 'ECCV 2018', 'CVPR 2020', 'CVPR 2021', 'NeurIPS 2021'],
  ['模块性质', '即插即用插件', '即插即用插件', '即插即用插件', '完整骨干网络'],
  ['注意力类型', '通道 + 空间（串联）', '纯通道', '通道 + 坐标位置', '局部窗口自注意力'],
  ['核心操作', '双池化 + 共享MLP + 7×7卷积', 'GAP + 一维卷积(核自适应)', '双向一维池化 + 1×1卷积', '窗口内 Q·K·V + 深度卷积FFN'],
  ['额外参数量', '极少（约 C²/8 级）', '几乎为零（仅 3~9 个）', '很少（约 2C²/r 级）', '骨干网络本身'],
  ['位置信息', '有（7×7 局部视野）', '无', '有（长距离、十字定位）', '有（窗口内全局 + 位置偏置）'],
  ['长距离依赖', '弱', '无', '较强（沿行/列全局）', '强（窗口内任意两位置）'],
  ['典型场景', '通用增强：分类/检测/分割', '移动端、实时、极致轻量', '移动端 + 需要定位的任务', '姿态估计、分割等密集预测'],
]

export default function Compare() {
  return (
    <section className="mt-24">
      <LessonHeader id="compare" no="SUMMARY" title="总结对比与选型指南" subtitle="Comparison & How to Choose" accent="slate" />

      <H3>一表看懂四个模块</H3>
      <div className="my-6 rounded-xl border border-slate-200 bg-white shadow-sm overflow-x-auto">
        <table className="w-full text-[13px] min-w-[720px]">
          <thead>
            <tr className="bg-slate-900 text-white">
              <th className="px-4 py-3 text-left font-bold w-24">维度</th>
              <th className="px-4 py-3 text-left font-bold text-orange-300">CBAM</th>
              <th className="px-4 py-3 text-left font-bold text-emerald-300">ECA</th>
              <th className="px-4 py-3 text-left font-bold text-sky-300">CoordAtt</th>
              <th className="px-4 py-3 text-left font-bold text-violet-300">HRFormer</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((row, i) => (
              <tr key={i} className={i % 2 ? 'bg-slate-50' : 'bg-white'}>
                <td className="px-4 py-2.5 font-bold text-slate-700 border-t border-slate-100">{row[0]}</td>
                {row.slice(1).map((cell, j) => (
                  <td key={j} className="px-4 py-2.5 text-slate-600 border-t border-slate-100 leading-5">{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <H3>怎么选？一张决策思路</H3>
      <StepBox icon="map" title="按需求对号入座" accent="slate">
        <ul className="list-disc pl-5 space-y-2">
          <li><b>「我想给现有模型（ResNet/UNet/YOLO）免费涨点」</b>→ 先试 <b className="text-emerald-600">ECA</b>：成本几乎为零，改三行代码就能用。</li>
          <li><b>「任务里目标位置很关键，但模型对定位不敏感」</b>→ 用 <b className="text-sky-600">CoordAtt</b>：坐标信息直接编码进注意力。</li>
          <li><b>「想要稳定可靠的通用增益，不在乎多一点参数」</b>→ 用 <b className="text-orange-600">CBAM</b>：通道+空间双管齐下，论文验证最充分，YOLO 系列改进中最常用。</li>
          <li><b>「做的是关键点检测 / 语义分割等逐像素任务，且愿意换骨干」</b>→ 上 <b className="text-violet-600">HRFormer</b>：高分辨率全程保留 + Transformer 表征，密集预测的旗舰方案。</li>
        </ul>
      </StepBox>

      <H3>动手练习：30 秒给 ResNet 块装上注意力</H3>
      <P>
        三个即插即用模块的接口完全一致（输入输出形状不变），所以可以随意替换。下面是一个标准插入范式：
      </P>
      <CodeBlock code={plugCode} title="plug_in.py —— 注意力模块插入 ResNet BasicBlock 的标准姿势" />

      <StepBox icon="warn" title="常见踩坑提醒" accent="slate">
        <ul className="list-disc pl-5 space-y-1.5">
          <li>注意力要放在 <b>BN 之后、残差相加之前</b>，不要放在 shortcut 路径上。</li>
          <li>CBAM 的两个子模块<b>顺序不要乱</b>：先通道后空间，这是论文消融实验的结论。</li>
          <li>ECA 的一维卷积核必须取<b>奇数</b>，保证邻居对称。</li>
          <li>CoordAtt 的转置（permute）最容易写错：记住 Y 方向池化结果要先转成 (B, C, W, 1) 才能和 X 方向拼接。</li>
          <li>HRFormer 的窗口自注意力要求 H、W 能被窗口大小整除，工程实现里需要 padding 处理（官方仓库已封装）。</li>
        </ul>
      </StepBox>

      <H3>全部参考文献汇总</H3>
      <RefBox title="本课程全部参考依据（论文 + 官方代码）" items={[
        { type: 'paper', label: 'SENet: Squeeze-and-Excitation Networks', desc: 'arXiv:1709.01507 · CVPR 2018 · 通道注意力基础', url: 'https://arxiv.org/abs/1709.01507' },
        { type: 'paper', label: 'CBAM: Convolutional Block Attention Module', desc: 'arXiv:1807.06521 · ECCV 2018', url: 'https://arxiv.org/abs/1807.06521' },
        { type: 'code', label: 'CBAM 官方代码 Jongchan/attention-module', desc: 'GitHub', url: 'https://github.com/Jongchan/attention-module' },
        { type: 'paper', label: 'ECA-Net: Efficient Channel Attention', desc: 'arXiv:1910.03151 · CVPR 2020', url: 'https://arxiv.org/abs/1910.03151' },
        { type: 'code', label: 'ECA 官方代码 BangguWu/ECANet', desc: 'GitHub', url: 'https://github.com/BangguWu/ECANet' },
        { type: 'paper', label: 'Coordinate Attention for Efficient Mobile Network Design', desc: 'arXiv:2103.02907 · CVPR 2021', url: 'https://arxiv.org/abs/2103.02907' },
        { type: 'code', label: 'CoordAtt 官方代码 Andrew-Qibin/CoordAttention', desc: 'GitHub', url: 'https://github.com/Andrew-Qibin/CoordAttention' },
        { type: 'paper', label: 'HRFormer: High-Resolution Transformer for Dense Prediction', desc: 'arXiv:2110.09408 · NeurIPS 2021', url: 'https://arxiv.org/abs/2110.09408' },
        { type: 'code', label: 'HRFormer 官方代码 HRNet/HRFormer', desc: 'GitHub（微软开源，含预训练权重）', url: 'https://github.com/HRNet/HRFormer' },
        { type: 'paper', label: 'HRNet: Deep High-Resolution Representation Learning', desc: 'arXiv:1908.07919 · CVPR 2019', url: 'https://arxiv.org/abs/1908.07919' },
        { type: 'paper', label: 'Swin Transformer: Hierarchical Vision Transformer using Shifted Windows', desc: 'arXiv:2103.14030 · ICCV 2021', url: 'https://arxiv.org/abs/2103.14030' },
        { type: 'paper', label: 'Attention Is All You Need', desc: 'arXiv:1706.03762 · NeurIPS 2017 · 自注意力源头', url: 'https://arxiv.org/abs/1706.03762' },
      ]} />

      <div className="mt-8 rounded-2xl bg-slate-900 text-white p-6 text-center">
        <div className="font-bold text-lg">课程完结 🎓</div>
        <p className="text-slate-400 text-sm mt-2 leading-7">
          从 SE 的「全员大会」，到 ECA 的「邻居小会」，到 CoordAtt 的「十字定位」，再到 HRFormer 的「多机位直播团队」——<br />
          注意力的本质从未改变：<b className="text-orange-300">把有限的算力，花在最重要的信息上。</b>
        </p>
      </div>
    </section>
  )
}
