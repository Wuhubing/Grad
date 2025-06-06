# 🧠 Knowledge Coupling Analysis for LLaMA-2 Models

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-green.svg)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-orange.svg)

**多跳推理知识片段耦合分析系统**  
*Multi-hop Reasoning Knowledge Piece Coupling Analysis*

</div>

## 📝 项目概述

本项目实现了一个GPU优化的知识片段间耦合度计算和涟漪效应分析系统，专门用于研究大语言模型（LLaMA-2、GPT-2等）中知识的相互依赖关系。

### 🎯 核心功能

- **🔬 梯度相似度分析**: 计算知识片段间的参数空间重叠度量
- **🌊 涟漪效应预测**: 基于耦合强度预测知识编辑的连锁反应
- **🚀 GPU加速计算**: 15x性能提升的GPU优化实现
- **📊 多数据集支持**: 支持HotpotQA、MuSiQue、WikiHop、Bamboogle等多跳推理数据集
- **🤖 多模型兼容**: 支持LLaMA-2 (7B/13B/70B) 和 GPT-2 (small/medium/large/xl)

### 🧮 核心算法

```
GradSim(i,j) = cos(∇_θ log P(a_i|q_i), ∇_θ log P(a_j|q_j))
```

通过计算不同知识片段对应答案的梯度向量余弦相似度，量化知识间的耦合强度。

## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/Wuhubing/Grad.git
cd Grad

# 安装依赖
pip install -r requirements.txt
```

### 基础使用

```bash
# 使用测试数据运行知识耦合分析
python knowledge_coupling_mvp.py --hotpot_data test_hotpot_20.json --max_samples 20

# 下载多跳推理数据集
python multihop_dataset_downloader.py

# 多模型测试
python test_multi_model.py
```

## 📁 项目结构

```
Grad/
├── 📄 knowledge_coupling_mvp.py      # 核心分析系统
├── 📄 multihop_dataset_downloader.py # 多数据集下载器
├── 📄 test_multi_model.py           # 多模型测试脚本
├── 📄 demo_results.py               # 结果演示脚本
├── 📄 test_downloader.py            # 下载器测试
├── 📄 requirements.txt              # 项目依赖
├── 📄 test_hotpot_20.json          # 测试数据集
└── 📚 README_MVP.md                 # MVP详细文档
```

## 🔧 核心组件

### 1. 知识耦合分析器 (`knowledge_coupling_mvp.py`)

- **多模型支持**: 自动检测并适配LLaMA-2和GPT-2架构
- **GPU优化**: 全流程GPU加速，支持大规模分析
- **智能层选择**: 根据模型类型自动选择目标层

### 2. 多数据集下载器 (`multihop_dataset_downloader.py`)

支持的数据集:
- **HotpotQA**: 维基百科多跳推理 (~500MB)
- **MuSiQue**: 单一支持事实多跳问答 (~100MB) 
- **WikiHop**: 多文档阅读理解 (~200MB)
- **Bamboogle**: 合成多跳问答 (~50MB)

### 3. 实验验证脚本

- `test_multi_model.py`: 多模型兼容性测试
- `demo_results.py`: 详细结果展示
- `test_downloader.py`: 数据集下载测试

## 📊 实验结果

### 性能指标

- **处理速度**: 23.44 samples/sec (GPU加速)
- **内存使用**: ~1.4GB GPU内存 (NVIDIA A40)
- **梯度维度**: 28,320,768 参数
- **耦合计算**: 实时GPU矩阵运算

### 分析输出

系统生成9个维度的综合分析结果:

1. **metadata**: 模型信息和处理详情
2. **knowledge_pieces**: 知识片段详细信息
3. **gradient_analysis**: 梯度计算结果
4. **coupling_analysis**: 耦合强度统计
5. **high_coupling_pairs**: 高耦合对排名
6. **cross_pot_analysis**: 跨问题耦合模式
7. **generated_files**: 生成文件路径
8. **gpu_performance**: GPU性能指标
9. **experiment_readiness**: 实验准备状态

## 🔬 技术原理

### 梯度提取
针对每个知识片段的真实答案token计算梯度:
```python
∇_θ log P(answer|question)
```

### 耦合度量
使用余弦相似度量化知识片段间的参数空间重叠:
```python
coupling = cosine_similarity(grad_i, grad_j)
```

### 涟漪效应预测
基于耦合强度预测知识编辑对其他片段的影响:
```python
ripple_strength = coupling_strength × edit_magnitude
```

## 🎨 可视化

系统自动生成:
- 📈 耦合强度热图
- 📊 耦合分布统计
- 🌊 涟漪效应预测图表

## 🛠️ 高级用法

### 自定义模型路径
```bash
python knowledge_coupling_mvp.py \
    --hotpot_data your_data.json \
    --model_path "meta-llama/Llama-2-7b-hf" \
    --max_samples 100 \
    --device cuda
```

### 批量数据集处理
```bash
# 下载所有支持的数据集
python multihop_dataset_downloader.py
```

### 结果分析
```python
from demo_results import analyze_detailed_results
results = analyze_detailed_results("gpu_coupling_results/")
```

## 📈 性能优化

- **GPU内存管理**: 智能批处理和内存释放
- **张量操作优化**: 全流程GPU张量操作
- **并行计算**: 矩阵运算并行化
- **缓存策略**: 梯度和模型输出缓存

## 🤝 贡献指南

欢迎贡献代码! 请确保:

1. 遵循现有代码风格
2. 添加适当的测试
3. 更新相关文档
4. 提交前运行测试套件

## 📚 相关资源

- [MVP详细文档](README_MVP.md)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

感谢以下开源项目和数据集:
- HuggingFace Transformers
- HotpotQA Dataset
- MuSiQue Dataset
- PyTorch Team

---

<div align="center">

**🧠 探索大语言模型中的知识耦合奥秘**

Made with ❤️ for AI Research

</div> 