# 🧠 Knowledge Coupling Analysis for LLaMA-2 Models

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-green.svg)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-orange.svg)

**Multi-hop Reasoning Knowledge Piece Coupling Analysis System**  
*Knowledge Fragment Coupling Analysis for Large Language Models*

</div>

## 📝 Project Overview

This project implements a GPU-optimized knowledge fragment coupling calculation and ripple effect analysis system, specifically designed to study the interdependencies of knowledge in large language models (LLaMA-2, GPT-2, etc.).

### 🎯 Core Features

- **🔬 Gradient Similarity Analysis**: Calculate parameter space overlap metrics between knowledge fragments
- **🌊 Ripple Effect Prediction**: Predict chain reactions of knowledge editing based on coupling strength
- **🚀 GPU Accelerated Computing**: 15x performance improvement with GPU optimization
- **📊 Multi-dataset Support**: Support for HotpotQA, MuSiQue, WikiHop, Bamboogle and other multi-hop reasoning datasets
- **🤖 Multi-model Compatibility**: Support for LLaMA-2 (7B/13B/70B) and GPT-2 (small/medium/large/xl)

### 🧮 Core Algorithm

```
GradSim(i,j) = cos(∇_θ log P(a_i|q_i), ∇_θ log P(a_j|q_j))
```

Quantify the coupling strength between knowledge by calculating the cosine similarity of gradient vectors corresponding to different knowledge fragment answers.

## 🚀 Quick Start

### Environment Setup

```bash
# Clone repository
git clone https://github.com/Wuhubing/Grad.git
cd Grad

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run knowledge coupling analysis with test data
python knowledge_coupling_mvp.py --hotpot_data test_hotpot_20.json --max_samples 20

# Download multi-hop reasoning datasets
python multihop_dataset_downloader.py

# Multi-model testing
python test_multi_model.py
```

## 📁 Project Structure

```
Grad/
├── 📄 knowledge_coupling_mvp.py      # Core analysis system
├── 📄 multihop_dataset_downloader.py # Multi-dataset downloader
├── 📄 test_multi_model.py           # Multi-model testing script
├── 📄 demo_results.py               # Results demonstration script
├── 📄 test_downloader.py            # Downloader testing
├── 📄 requirements.txt              # Project dependencies
├── 📄 test_hotpot_20.json          # Test dataset
└── 📚 README_MVP.md                 # MVP detailed documentation
```

## 🔧 Core Components

### 1. Knowledge Coupling Analyzer (`knowledge_coupling_mvp.py`)

- **Multi-model Support**: Automatically detect and adapt to LLaMA-2 and GPT-2 architectures
- **GPU Optimization**: Full pipeline GPU acceleration, supporting large-scale analysis
- **Intelligent Layer Selection**: Automatically select target layers based on model type

### 2. Multi-dataset Downloader (`multihop_dataset_downloader.py`)

Supported datasets:
- **HotpotQA**: Wikipedia multi-hop reasoning (~500MB)
- **MuSiQue**: Single supporting fact multi-hop QA (~100MB) 
- **WikiHop**: Multi-document reading comprehension (~200MB)
- **Bamboogle**: Synthetic multi-hop QA (~50MB)

### 3. Experimental Validation Scripts

- `test_multi_model.py`: Multi-model compatibility testing
- `demo_results.py`: Detailed results display
- `test_downloader.py`: Dataset download testing

## 📊 Experimental Results

### Performance Metrics

- **Processing Speed**: 23.44 samples/sec (GPU accelerated)
- **Memory Usage**: ~1.4GB GPU memory (NVIDIA A40)
- **Gradient Dimensions**: 28,320,768 parameters
- **Coupling Computation**: Real-time GPU matrix operations

### Analysis Output

The system generates comprehensive analysis results across 9 dimensions:

1. **metadata**: Model information and processing details
2. **knowledge_pieces**: Detailed knowledge fragment information
3. **gradient_analysis**: Gradient computation results
4. **coupling_analysis**: Coupling strength statistics
5. **high_coupling_pairs**: High coupling pair rankings
6. **cross_pot_analysis**: Cross-question coupling patterns
7. **generated_files**: Generated file paths
8. **gpu_performance**: GPU performance metrics
9. **experiment_readiness**: Experiment preparation status

## 🔬 Technical Principles

### Gradient Extraction
Calculate gradients for the ground truth answer tokens of each knowledge fragment:
```python
∇_θ log P(answer|question)
```

### Coupling Measurement
Use cosine similarity to quantify parameter space overlap between knowledge fragments:
```python
coupling = cosine_similarity(grad_i, grad_j)
```

### Ripple Effect Prediction
Predict the impact of knowledge editing on other fragments based on coupling strength:
```python
ripple_strength = coupling_strength × edit_magnitude
```

## 🎨 Visualization

The system automatically generates:
- 📈 Coupling strength heatmaps
- 📊 Coupling distribution statistics
- 🌊 Ripple effect prediction charts

## 🛠️ Advanced Usage

### Custom Model Path
```bash
python knowledge_coupling_mvp.py \
    --hotpot_data your_data.json \
    --model_path "meta-llama/Llama-2-7b-hf" \
    --max_samples 100 \
    --device cuda
```

### Batch Dataset Processing
```bash
# Download all supported datasets
python multihop_dataset_downloader.py
```

### Results Analysis
```python
from demo_results import analyze_detailed_results
results = analyze_detailed_results("gpu_coupling_results/")
```

## 📈 Performance Optimization

- **GPU Memory Management**: Intelligent batching and memory release
- **Tensor Operation Optimization**: Full pipeline GPU tensor operations
- **Parallel Computing**: Matrix operation parallelization
- **Caching Strategy**: Gradient and model output caching

## 🤝 Contributing

Welcome to contribute code! Please ensure:

1. Follow existing code style
2. Add appropriate tests
3. Update relevant documentation
4. Run test suite before submission

## 📚 Related Resources

- [MVP Detailed Documentation](README_MVP.md)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)

</div>
