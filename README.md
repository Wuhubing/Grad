# Knowledge Coupling Analysis Project

## 📋 Project Overview

This project implements a knowledge coupling analysis system based on large language models, specifically designed to analyze coupling relationships between knowledge pieces in multi-hop reasoning tasks. We conduct experiments using the HotpotQA dataset and LLaMA2-7B model, quantifying knowledge coupling strength through gradient analysis techniques.

## 🎯 Key Findings

### 🧠 Model Layer Selection Optimization
- **Optimal Layer Range:** Layers 28-31 (last 4 layers of LLaMA2-7B)
- **Memory Optimization:** Reduced GPU usage from 44GB to 21GB (52% reduction)
- **Semantic Representation:** Higher layers better capture semantic-level knowledge relationships

### 📊 Knowledge Coupling Analysis Results
- **Dataset Scale:** 97,852 samples (90,447 train + 7,405 dev)
- **Coupling Computation:** Cosine similarity-based gradient vector analysis
- **High Coupling Threshold:** ≥0.4 considered strong coupling relationship
- **Key Discovery:** Knowledge pieces within the same reasoning chain show highest coupling strength (~0.8)

### 🎮 Hardware Requirements & Performance
- **GPU Memory:** Minimum 20GB (A40 or higher recommended)
- **System Memory:** Minimum 50GB available
- **Processing Speed:** ~10 samples/minute (layers 28-31 configuration)

## 🚀 Quick Start

### Method 1: One-Click Automated Run (Recommended)

```bash
# 1. Run complete pipeline with HuggingFace token
./run_full_analysis.sh --hf-token hf_your_token_here

# 2. Small-scale test run
./run_full_analysis.sh --batch-size 100 --sub-batch 5

# 3. Environment setup only
./run_full_analysis.sh --env-only

# 4. Data preparation only
./run_full_analysis.sh --data-only
```

### Method 2: Manual Step-by-Step Run

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Login to HuggingFace
huggingface-cli login

# 4. Download and convert data
python multihop_dataset_downloader.py
python convert_hotpot_to_coupling_format.py

# 5. Run knowledge coupling analysis
python knowledge_coupling_mvp.py --samples 20

# 6. Large-scale batch processing
python knowledge_coupling_batch_processor.py --batch_size 1000
```

## 📁 Project Structure

```
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── run_full_analysis.sh               # One-click automation script
├── monitor.sh                         # Monitoring script
│
├── 📊 Core Analysis Modules
│   ├── knowledge_coupling_mvp.py              # Core coupling analysis engine
│   ├── knowledge_coupling_batch_processor.py  # Batch processing system
│   └── demo_knowledge_coupling.py             # Demo script
│
├── 🗂️ Data Processing Modules
│   ├── multihop_dataset_downloader.py         # Dataset downloader
│   ├── convert_hotpot_to_coupling_format.py   # Format converter
│   └── export_report_data.py                  # Result export tool
│
├── 📁 Data Directory
│   ├── datasets/                      # Raw data
│   │   ├── hotpot_train_v1.1.json    # HotpotQA training set (540MB)
│   │   ├── hotpot_dev_distractor_v1.json  # HotpotQA dev set (44MB)
│   │   └── processed/                 # Converted data
│   │       ├── hotpotqa_all_converted.json    # Complete dataset (674MB)
│   │       ├── hotpotqa_sample_20.json        # 20-sample test set
│   │       ├── hotpotqa_sample_100.json       # 100-sample test set
│   │       └── ...
│
├── 📈 Output Directory
│   ├── results/                       # Analysis results
│   ├── logs/                         # Runtime logs
│   └── checkpoints/                  # Batch processing checkpoints
│
└── 🔧 Configuration Files
    ├── .gitignore                    # Git ignore file
    └── venv/                         # Python virtual environment
```

## 🛠️ Environment Setup

### System Requirements
- **Operating System:** Linux (Ubuntu 18.04+ recommended)
- **Python:** 3.8+
- **CUDA:** 11.0+ (PyTorch compatible)
- **GPU:** NVIDIA A40 or equivalent (minimum 20GB VRAM)

### Core Dependencies
```bash
# Deep learning frameworks
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.23.0

# LLaMA2 model support
sentencepiece>=0.1.99
protobuf>=3.20.0
tokenizers>=0.14.0

# Data science
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Visualization & monitoring
matplotlib>=3.7.0
seaborn>=0.12.0
psutil>=5.9.0
gpustat>=1.1.0
```

## 🔑 HuggingFace Configuration

LLaMA2 model requires HuggingFace access permissions:

```bash
# Method 1: Command line login
huggingface-cli login

# Method 2: Using token
export HUGGINGFACE_HUB_TOKEN="hf_your_token_here"

# Method 3: Script parameter
./run_full_analysis.sh --hf-token hf_your_token_here
```

Get token: https://huggingface.co/settings/tokens

## 📊 Monitoring & Management

### Real-time Monitoring
```bash
# Launch real-time monitoring panel
./monitor.sh watch

# Check current status
./monitor.sh status

# View detailed logs
./monitor.sh detailed-log

# Stop batch processing
./monitor.sh stop
```

### Monitoring Interface Example
```
===============================================
     Knowledge Coupling Analysis Monitor v2.0
===============================================
Time: 2025-06-11 12:30:45

✅ Batch process running (PID: 12345)
   Runtime: 02:15:30

💾 Memory Usage:
   Mem: 156G/503G (31%)
   Swap: 0G/16G (0%)

🎮 GPU Status:
   GPU0: NVIDIA A40
     Utilization: 95%  Memory: 21GB/46GB (46%)  Temperature: 65°C

📊 Processing Progress:
   Completed batches: 15
   Latest checkpoint: batch_0014
   Last update: 2025-06-11 12:28:15

📁 Output Files:
   Result file count: 127
   -rw-r--r-- 1 root root 2.3G coupling_results.json
   -rw-r--r-- 1 root root 1.1G coupling_matrix.npy
```

## 🧪 Testing & Validation

### Small-scale Testing
```bash
# Test with 20 samples
python knowledge_coupling_mvp.py --samples 20 --layer_range 28-31

# Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Performance Benchmarks
| Configuration | Samples | GPU Memory | Processing Time | Coupling Strength Range |
|---------------|---------|------------|-----------------|------------------------|
| Layers 28-31 | 10 | 21GB | 2 minutes | 0.02-0.81 |
| Layers 24-27 | 10 | 19GB | 2 minutes | 0.03-0.76 |
| All layers 0-31 | 10 | 44GB | 3 minutes | 0.01-0.85 |

## 📈 Result Interpretation

### Output File Description
```bash
results/
├── coupling_results_for_validation.json  # Complete coupling analysis results
├── coupling_pairs.csv                   # Coupling pairs in CSV format
├── high_coupling_pairs.json             # High coupling pairs (≥0.4)
├── coupling_matrix.npy                  # Raw coupling matrix
└── analysis_report.md                   # Detailed analysis report
```

### Coupling Strength Interpretation
- **0.8-1.0:** Extremely strong coupling (usually knowledge within same reasoning chain)
- **0.4-0.8:** Strong coupling (related knowledge pieces)
- **0.2-0.4:** Medium coupling (indirectly related)
- **0.0-0.2:** Weak coupling (essentially independent)
- **Negative values:** Inverse relationships (rare)

## ⚠️ Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Solution: Reduce batch size
./run_full_analysis.sh --batch-size 500 --sub-batch 5

# Or use fewer layers
./run_full_analysis.sh --layer-start 30 --layer-end 31
```

#### 2. HuggingFace Token Error
```bash
# Re-login
huggingface-cli logout
huggingface-cli login

# Check token permissions
huggingface-cli whoami
```

#### 3. Data Download Failed
```bash
# Manual download
cd datasets
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

#### 4. Process Stuck
```bash
# Check process status
./monitor.sh status

# Force stop
./monitor.sh stop

# Clean checkpoints and restart
rm -rf checkpoints/*
./run_full_analysis.sh --fresh
```

### Log Analysis
```bash
# View error logs
tail -50 logs/batch_processing.log | grep ERROR

# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor memory
free -h && cat /proc/meminfo | grep Available
```

## 📚 Technical Details

### Algorithm Principles
1. **Gradient Extraction:** Extract gradient vectors from LLaMA2's down_proj layers
2. **Similarity Computation:** Use cosine similarity to quantify coupling strength
3. **Batch Processing Optimization:** Process in batches to avoid memory overflow
4. **Checkpoint Mechanism:** Support interruption recovery

### Key Parameters
- `--layer_start/--layer_end`: Model layer range for analysis
- `--batch_size`: Main batch size (recommended 1000-2000)
- `--sub_batch_size`: Sub-batch size (recommended 5-10)
- `--threshold`: High coupling threshold (default 0.4)

## 🤝 Contributing

### Development Environment
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python -m pytest tests/

# Code formatting
black *.py
```

### Extension Features
- Support for more models (GPT-4, Claude, etc.)
- Add visualization interface
- Optimize memory usage
- Support distributed computing

## 📜 License

This project follows the MIT License - see LICENSE file for details

## 📞 Contact

- **Project Maintainer:** [Your Name]
- **Email:** [Your Email]
- **GitHub:** [Project URL]

## 🙏 Acknowledgments

- HotpotQA dataset providers
- HuggingFace Transformers team
- Meta LLaMA team
- Open source community contributors

---

**Last Updated:** June 11, 2025  
**Version:** v2.0

> 💡 **Tip:** If you're running for the first time, we recommend starting with `--data-only` to prepare data, then testing with small batches, and finally conducting full analysis. 