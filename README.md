# 注意力机制与高分辨率表征实战课

面向深度学习初学者的交互式教学网站，用「生活类比 → 原理图解 → 伪代码 → PyTorch 代码」四步讲讲透四个视觉注意力里程碑模块：

| 课程 | 模块 | 出处 | 参考 |
| --- | --- | --- | --- |
| 第 0 课 | SE（注意力入门） | CVPR 2018 | [arXiv:1709.01507](https://arxiv.org/abs/1709.01507) |
| 第 1 课 | CBAM | ECCV 2018 | [arXiv:1807.06521](https://arxiv.org/abs/1807.06521) · [官方代码](https://github.com/Jongchan/attention-module) |
| 第 2 课 | ECA | CVPR 2020 | [arXiv:1910.03151](https://arxiv.org/abs/1910.03151) · [官方代码](https://github.com/BangguWu/ECANet) |
| 第 3 课 | CoordAtt | CVPR 2021 | [arXiv:2103.02907](https://arxiv.org/abs/2103.02907) · [官方代码](https://github.com/Andrew-Qibin/CoordAttention) |
| 第 4 课 | HRFormer | NeurIPS 2021 | [arXiv:2110.09408](https://arxiv.org/abs/2110.09408) · [官方代码](https://github.com/HRNet/HRFormer) |

## 课程内容

- 生活化类比讲解原理（鸡尾酒会效应、舞台灯光师、邻居开小会、多机位直播团队）
- 手绘风格 SVG 原理结构图
- 逐行编号伪代码流程图
- 可直接运行的 PyTorch 完整实现（一键复制）
- 四模块八维对比表与选型指南
- 全部论文与官方仓库参考链接

## 技术栈

React 19 + TypeScript + Vite + Tailwind CSS + shadcn/ui + lucide-react

## 本地运行

```bash
npm install
npm run dev     # 开发预览
npm run build   # 构建到 dist/
```
