# 🤖 Multi-Model Knowledge Coupling MVP System

**专门用于GPT-2和LLaMA-2的知识耦合分析和涟漪效应验证系统**

验证核心假设：**「GradSim高 ↔ 涟漪强」**

## 🎯 系统概览

这是一个支持多种模型架构的最小可行实验(MVP)系统，用于验证梯度相似度(GradSim)是否能预测知识编辑中的涟漪效应。

### 🚀 支持的模型
- **GPT-2 系列**: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- **LLaMA-2 系列**: `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf`, `meta-llama/Llama-2-70b-hf`
- **自动检测**: 支持任何HuggingFace兼容的Causal LM模型

### 核心研究问题
1. **知识片段间的耦合度**：通过梯度相似度量化知识片段在参数空间的重叠程度
2. **涟漪效应预测**：高耦合度是否预示着更强的编辑涟漪效应
3. **跨模型验证**：不同模型架构下的耦合-涟漪关系是否一致

### 技术实现
- **多模型支持**：自动适配GPT-2和LLaMA-2的不同层结构
- **数据**：HotpotQA 2-hop推理链
- **方法**：梯度计算 + MEMIT编辑 + 涟漪效应测量
- **公式**：`GradSim(i,j) = cos(∇_θ log P(a_i|q_i), ∇_θ log P(a_j|q_j))`

## 🚀 快速开始

### 环境要求
```bash
pip install torch transformers datasets scikit-learn matplotlib seaborn pandas tqdm numpy
```

### 数据准备
下载HotpotQA数据：
```bash
# 下载HotpotQA开发集
wget https://hotpotqa.github.io/data/hotpot_dev_distractor_v1.json
```

### 🧪 测试多模型支持
```bash
# 测试模型兼容性
python test_multi_model.py
```

### 运行分析

#### GPT-2 版本（推荐入门）
```bash
# GPT-2 Small (CPU友好)
python run_complete_mvp.py \
    --hotpot_data hotpot_dev_distractor_v1.json \
    --model_path gpt2 \
    --max_samples 20 \
    --max_experiments 2 \
    --device cpu

# GPT-2 Medium (更好效果)
python run_complete_mvp.py \
    --hotpot_data hotpot_dev_distractor_v1.json \
    --model_path gpt2-medium \
    --max_samples 50 \
    --max_experiments 3 \
    --device cuda
```

#### LLaMA-2 版本（需要GPU）
```bash
# LLaMA-2 7B
python run_complete_mvp.py \
    --hotpot_data hotpot_dev_distractor_v1.json \
    --model_path meta-llama/Llama-2-7b-hf \
    --max_samples 50 \
    --max_experiments 3 \
    --device cuda

# LLaMA-2 13B (大内存GPU)
python run_complete_mvp.py \
    --hotpot_data hotpot_dev_distractor_v1.json \
    --model_path meta-llama/Llama-2-13b-hf \
    --max_samples 30 \
    --max_experiments 2 \
    --device cuda
```

## 📋 系统架构

### 🔗 阶段1：多模型知识耦合分析 (`knowledge_coupling_mvp.py`)
**目标**：在不同模型架构上计算知识片段间的真实耦合度

**多模型适配**：
- **自动检测**：根据模型路径和结构自动识别模型类型
- **层选择**：自适应选择目标层
  - GPT-2: `transformer.h.{i}.mlp.c_proj`
  - LLaMA-2: `model.layers.{i}.mlp.down_proj`
- **统一接口**：相同的API支持所有模型

**核心类**：
- `MultiModelKnowledgeCouplingMVP`：多模型主分析器
- `KnowledgePiece`：知识片段表示

### 🌊 阶段2：多模型MEMIT涟漪实验 (`memit_ripple_analyzer.py`)
**目标**：在不同模型上测量知识编辑的涟漪效应

**多模型MEMIT**：
- **层适配**：根据模型类型选择编辑层
- **权重恢复**：安全的多模型权重操作
- **统一评估**：标准化的ΔlogP和EM测量

**核心类**：
- `MultiModelMEMIT`：多模型MEMIT实现
- `MultiModelRippleAnalyzer`：涟漪效应分析器

### 📊 阶段3：跨模型相关性验证 (`run_complete_mvp.py`)
**目标**：验证GradSim与涟漪效应在不同模型上的一致性

**流程**：
1. **模型检测**：自动识别和适配模型架构
2. **统一分析**：相同的分析流程适用于所有模型
3. **结果对比**：支持跨模型结果比较

## 🔧 模型特定配置

### GPT-2 系列配置
```python
# GPT-2 目标层
'gpt2': {
    'layer_pattern': 'transformer.h.{}.mlp.c_proj',
    'embed_tokens': 'transformer.wte',
    'num_layers': 12  # gpt2-xl: 48层
}
```

### LLaMA-2 系列配置
```python
# LLaMA-2 目标层
'llama': {
    'layer_pattern': 'model.layers.{}.mlp.down_proj',
    'embed_tokens': 'model.embed_tokens',
    'num_layers': 32  # 7B: 32层, 13B: 40层, 70B: 80层
}
```

### 自定义模型支持
```python
# 添加新模型支持
SUPPORTED_MODELS = {
    'your_model': {
        'patterns': ['your_model_name'],
        'tokenizer_class': YourTokenizer,
        'model_class': YourModel,
        'target_layers': 'your_target_layer',
        'layer_pattern': 'your.layer.pattern.{}'
    }
}
```

## 📊 输出文件说明

所有模型生成相同的输出格式，便于比较：

### 核心数据文件
- `knowledge_pieces.json` - 提取的知识片段数据
- `coupling_matrix.npy` - 知识耦合矩阵（N×N）
- `coupling_analysis.json` - 耦合分析结果（包含模型类型）
- `ripple_experiments.json` - MEMIT实验和涟漪效应数据
- `correlation_analysis.json` - 相关性分析结果

### 可视化文件
- `coupling_heatmap.png` - 知识耦合热图
- `comprehensive_analysis.png` - 六面板综合分析图
- `final_mvp_report.md` - 完整分析报告（包含模型信息）

## 🔍 跨模型结果对比

### 假设验证标准（一致性）
不同模型应该显示相似的模式：
- **GPT-2**: 验证在Transformer架构上的GradSim-涟漪关系
- **LLaMA-2**: 验证在现代大模型架构上的一致性
- **跨模型**: 比较不同架构的耦合模式差异

### 关键对比指标
1. **耦合分布**: 不同模型的耦合强度分布
2. **涟漪模式**: 涟漪效应在不同架构上的表现
3. **相关性一致性**: GradSim-涟漪关系的跨模型稳定性

## 🚧 故障排除

### 常见问题

**1. 模型检测失败**
```bash
# 查看支持的模型
python -c "from knowledge_coupling_mvp import MultiModelKnowledgeCouplingMVP; print('Supported patterns:', MultiModelKnowledgeCouplingMVP.SUPPORTED_MODELS)"
```

**2. 内存不足（大模型）**
```bash
# 使用更小的样本数
--max_samples 10 --max_experiments 1

# 使用CPU（慢但稳定）
--device cpu
```

**3. 层名称不匹配**
```python
# 检查模型层结构
for name, param in model.named_parameters():
    if 'mlp' in name.lower():
        print(name)
```

**4. LLaMA-2 访问权限**
```bash
# 需要HuggingFace token
pip install huggingface_hub
huggingface-cli login
```

## 📈 性能对比

### 计算需求

| 模型 | 内存需求 | 推荐设备 | 处理速度 | 准确性 |
|------|----------|----------|----------|--------|
| GPT-2 Small | ~2GB | CPU/GPU | 快 | 基础 |
| GPT-2 Medium | ~4GB | GPU | 中等 | 良好 |
| GPT-2 Large | ~6GB | GPU | 中等 | 很好 |
| LLaMA-2 7B | ~14GB | GPU | 慢 | 优秀 |
| LLaMA-2 13B | ~26GB | 高端GPU | 很慢 | 出色 |

### 建议配置
- **入门学习**: GPT-2 Small (CPU)
- **研究验证**: GPT-2 Medium/Large (GPU)
- **论文实验**: LLaMA-2 7B (GPU)
- **完整对比**: 多模型组合实验

## 🎯 实际使用示例

### 快速验证（GPT-2）
```bash
# 5分钟快速测试
python run_complete_mvp.py \
    --hotpot_data hotpot_dev_distractor_v1.json \
    --model_path gpt2 \
    --max_samples 10 \
    --max_experiments 1 \
    --output_dir quick_test
```

### 完整实验（LLaMA-2）
```bash
# 完整研究实验
python run_complete_mvp.py \
    --hotpot_data hotpot_dev_distractor_v1.json \
    --model_path meta-llama/Llama-2-7b-hf \
    --max_samples 100 \
    --max_experiments 5 \
    --output_dir full_experiment
```

### 跨模型对比
```bash
# 运行多个模型并比较结果
for model in gpt2 gpt2-medium meta-llama/Llama-2-7b-hf; do
    python run_complete_mvp.py \
        --hotpot_data hotpot_dev_distractor_v1.json \
        --model_path $model \
        --output_dir results_$model \
        --max_samples 50
done
```

## 🔬 研究扩展

### 新增模型支持
1. 在`SUPPORTED_MODELS`中添加配置
2. 实现模型特定的层选择逻辑
3. 测试梯度计算和MEMIT编辑

### 新增数据集
1. 实现数据集特定的知识提取器
2. 适配cloze问题生成逻辑
3. 验证跨数据集一致性

## 📄 引用

```bibtex
@software{multi_model_knowledge_coupling_mvp,
  title={Multi-Model Knowledge Coupling MVP: Cross-Architecture Gradient Similarity Analysis},
  author={Your Name},
  year={2024},
  note={Supports GPT-2 and LLaMA-2 architectures},
  url={https://github.com/your_repo}
}
```

---

**🌟 新特性**: 现在支持GPT-2和LLaMA-2！选择最适合你研究需求的模型架构。 