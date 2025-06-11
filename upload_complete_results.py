#!/usr/bin/env python3
"""
Upload Complete Knowledge Coupling Analysis Results to HuggingFace
Including all batch results and final merged data
"""

import os
import json
import glob
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
from datetime import datetime

def create_complete_dataset_card():
    """Create a comprehensive dataset card for the complete knowledge coupling results."""
    
    card_content = """
---
language:
- en
license: mit
size_categories:
- 100K<n<1M
task_categories:
- question-answering
- text-analysis
tags:
- knowledge-coupling
- llama2
- hotpotqa
- multi-hop-reasoning
- gradient-analysis
- ripple-effects
- batch-processing
---

# Complete Knowledge Coupling Analysis on HotpotQA Dataset

## Dataset Description

This dataset contains the **complete results** of a comprehensive knowledge coupling analysis performed on the HotpotQA dataset using LLaMA2-7B model. The analysis investigates how different pieces of knowledge interact within the model's parameter space through gradient-based coupling measurements.

**This is the full dataset including all batch processing results and intermediate data.**

## Research Overview

- **Model**: meta-llama/Llama-2-7b-hf (layers 28-31 focused analysis)
- **Dataset**: HotpotQA (train + dev splits, 97,852 total samples)
- **Method**: Gradient-based knowledge coupling via cosine similarity
- **Target Layers**: model.layers.28-31.mlp.down_proj (semantically rich layers)
- **Processing**: Batch processing with 2000 samples per batch (49 total batches)

## Key Findings

The analysis revealed:
- Mean coupling score: 0.0222 across all knowledge piece pairs
- High coupling pairs (≥0.4 threshold): Critical for ripple effect prediction
- Layer-specific analysis focusing on MLP down-projection layers
- Comprehensive gradient analysis with 180,355,072 dimensions per knowledge piece
- Batch-wise processing enabled full dataset coverage with memory optimization

## Dataset Structure

### Final Merged Results
- `final_merged_results/global_analysis_results.json`: Comprehensive analysis summary
- `final_merged_results/all_knowledge_pieces.json`: Complete knowledge pieces (92MB)
- `final_merged_results/all_coupling_pairs.csv`: All coupling measurements (245MB)

### Batch Results (batch_0000 to batch_0048)
Each batch directory contains:
- `batch_metadata.json`: Batch processing metadata and statistics
- `knowledge_pieces.json`: Knowledge pieces processed in this batch
- `coupling_pairs.csv`: Coupling measurements for this batch
- `high_coupling_pairs.json`: High coupling pairs (≥0.4) in this batch

### Supporting Files
- `dataset_info.json`: Complete dataset statistics and conversion details
- `coupling_analysis_config.json`: Analysis configuration and parameters
- `batch_summary.json`: Summary of all batch processing results

## Usage Examples

### Load Complete Results
```python
from datasets import load_dataset

# Load the complete knowledge coupling results
dataset = load_dataset("Wuhuwill/hotpotqa-knowledge-coupling-complete")

# Access final merged results
global_results = dataset["final_merged_results/global_analysis_results.json"]
all_knowledge_pieces = dataset["final_merged_results/all_knowledge_pieces.json"]
all_coupling_pairs = dataset["final_merged_results/all_coupling_pairs.csv"]
```

### Access Specific Batch Results
```python
# Access specific batch results
batch_0 = dataset["batch_0000/knowledge_pieces.json"]
batch_0_coupling = dataset["batch_0000/coupling_pairs.csv"]
batch_0_metadata = dataset["batch_0000/batch_metadata.json"]

# High coupling pairs from a specific batch
high_coupling_batch_0 = dataset["batch_0000/high_coupling_pairs.json"]
```

### Analyze Batch Processing Statistics
```python
import json

# Load batch summary
batch_summary = json.loads(dataset["batch_summary.json"])

# Analyze per-batch statistics
for batch_id, stats in batch_summary["batch_statistics"].items():
    print(f"Batch {batch_id}: {stats['knowledge_pieces']} pieces, "
          f"Mean coupling: {stats['mean_coupling']:.4f}")
```

## Research Applications

This complete dataset enables:

1. **Full-Scale Knowledge Coupling Analysis**: Access to all 97,852 samples with complete coupling measurements
2. **Batch-wise Analysis**: Study how coupling patterns vary across different data subsets
3. **Incremental Processing Research**: Understand how results accumulate during batch processing
4. **Memory-Efficient Model Analysis**: Learn from the batch processing approach for large-scale analyses
5. **Ripple Effect Prediction**: Use high coupling pairs for knowledge editing impact prediction

## Technical Specifications

- **Total Knowledge Pieces**: 97,852
- **Total Coupling Pairs**: ~4.8 billion measurements
- **Batch Size**: 2,000 samples per batch
- **Total Batches**: 49 (batch_0000 to batch_0048)
- **Memory Optimization**: Layer-focused analysis (28-31) for GPU efficiency
- **Processing Time**: Complete analysis across multiple batch runs
- **Storage**: ~350MB total compressed data

## Hardware Requirements

- **GPU**: NVIDIA A40 (46GB VRAM) or equivalent
- **Memory**: ~21GB GPU memory during processing
- **Storage**: ~2GB for complete dataset download

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{hotpotqa_knowledge_coupling_complete,
  title={Complete Knowledge Coupling Analysis on HotpotQA Dataset using LLaMA2-7B},
  author={Wuhuwill},
  year={2024},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/Wuhuwill/hotpotqa-knowledge-coupling-complete},
  note={Full dataset including all batch processing results}
}
```

## Technical Details

- **Gradient Computation**: ∇_θ log P(answer|question) for cloze-style questions
- **Coupling Measurement**: Cosine similarity between L2-normalized gradients
- **Memory Optimization**: Focused on layers 28-31 to handle GPU memory constraints
- **Batch Processing**: 2000 samples per batch for memory efficiency
- **Hardware**: NVIDIA A40 GPU (46GB VRAM)
- **Processing Framework**: Custom PyTorch implementation with HuggingFace Transformers

## License

This dataset is released under the MIT License. The original HotpotQA dataset follows its respective licensing terms.

## Acknowledgments

This research was conducted using advanced GPU resources and represents a comprehensive analysis of knowledge interactions in large language models.
"""
    
    return card_content

def create_batch_summary():
    """Create a summary of all batch processing results."""
    
    results_dir = "results/full_hotpotqa_analysis"
    batch_dirs = glob.glob(f"{results_dir}/batch_*")
    batch_dirs.sort()
    
    batch_statistics = {}
    total_knowledge_pieces = 0
    total_coupling_pairs = 0
    all_mean_couplings = []
    
    for batch_dir in batch_dirs:
        batch_name = os.path.basename(batch_dir)
        metadata_file = os.path.join(batch_dir, "batch_metadata.json")
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                batch_stats = {
                    "knowledge_pieces": metadata.get("total_knowledge_pieces", 0),
                    "coupling_pairs": metadata.get("total_coupling_pairs", 0),
                    "mean_coupling": metadata.get("mean_coupling", 0.0),
                    "high_coupling_pairs": metadata.get("high_coupling_pairs", 0),
                    "processing_time": metadata.get("total_time", "Unknown")
                }
                
                batch_statistics[batch_name] = batch_stats
                total_knowledge_pieces += batch_stats["knowledge_pieces"]
                total_coupling_pairs += batch_stats["coupling_pairs"]
                if batch_stats["mean_coupling"]:
                    all_mean_couplings.append(batch_stats["mean_coupling"])
                    
            except Exception as e:
                print(f"Warning: Could not read metadata for {batch_name}: {e}")
    
    summary = {
        "dataset_name": "hotpotqa-knowledge-coupling-complete",
        "description": "Complete batch processing summary for knowledge coupling analysis",
        "creation_date": datetime.now().isoformat(),
        "total_batches": len(batch_dirs),
        "batch_range": f"batch_0000 to batch_{len(batch_dirs)-1:04d}",
        "aggregate_statistics": {
            "total_knowledge_pieces": total_knowledge_pieces,
            "total_coupling_pairs": total_coupling_pairs,
            "overall_mean_coupling": sum(all_mean_couplings) / len(all_mean_couplings) if all_mean_couplings else 0.0,
            "batch_count": len(batch_dirs)
        },
        "batch_statistics": batch_statistics,
        "file_structure": {
            "final_merged_results/": "Complete merged analysis results",
            "batch_XXXX/": "Individual batch processing results",
            "batch_metadata.json": "Metadata for each batch",
            "knowledge_pieces.json": "Knowledge pieces per batch",
            "coupling_pairs.csv": "Coupling measurements per batch",
            "high_coupling_pairs.json": "High coupling pairs per batch"
        }
    }
    
    return summary

def upload_complete_results_to_huggingface(repo_name, username=None, private=False):
    """Upload all results including batch data to HuggingFace dataset."""
    
    # Initialize HuggingFace API
    api = HfApi()
    
    # Create the repository name
    if username:
        full_repo_name = f"{username}/{repo_name}"
    else:
        # Try to get username from token
        try:
            user_info = api.whoami()
            username = user_info["name"]
            full_repo_name = f"{username}/{repo_name}"
            print(f"Using username: {username}")
        except Exception as e:
            print(f"Could not get username: {e}")
            full_repo_name = repo_name
    
    print(f"Creating repository: {full_repo_name}")
    
    # Create the repository
    try:
        create_repo(
            repo_id=full_repo_name,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
        print(f"✓ Repository created/confirmed: {full_repo_name}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return False
    
    # Create supporting files
    print("Creating complete dataset metadata files...")
    
    # Create README.md (dataset card)
    with open("temp_complete_README.md", "w", encoding="utf-8") as f:
        f.write(create_complete_dataset_card())
    
    # Create batch summary
    batch_summary = create_batch_summary()
    with open("temp_batch_summary.json", "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)
    
    # Read existing analysis config and dataset info
    results_dir = "results/full_hotpotqa_analysis/final_merged_results"
    
    # Create enhanced dataset info
    if os.path.exists(f"{results_dir}/global_analysis_results.json"):
        with open(f"{results_dir}/global_analysis_results.json", 'r', encoding='utf-8') as f:
            global_results = json.load(f)
    else:
        global_results = {}
    
    enhanced_dataset_info = {
        "dataset_name": "hotpotqa-knowledge-coupling-complete",
        "description": "Complete knowledge coupling analysis results including all batch data",
        "creation_date": datetime.now().isoformat(),
        "model_info": {
            "model_name": "meta-llama/Llama-2-7b-hf",
            "target_layers": "model.layers.28-31.mlp.down_proj",
            "total_parameters_analyzed": "180,355,072 per knowledge piece"
        },
        "dataset_statistics": global_results.get("dataset_stats", {}),
        "analysis_results": {
            "total_knowledge_pieces": global_results.get("total_knowledge_pieces", batch_summary["aggregate_statistics"]["total_knowledge_pieces"]),
            "total_coupling_pairs": global_results.get("total_coupling_pairs", batch_summary["aggregate_statistics"]["total_coupling_pairs"]),
            "mean_coupling_score": global_results.get("mean_coupling", batch_summary["aggregate_statistics"]["overall_mean_coupling"]),
            "high_coupling_threshold": 0.4,
            "total_batches": batch_summary["total_batches"]
        },
        "batch_processing": batch_summary,
        "files": {
            "README.md": "Complete dataset documentation",
            "batch_summary.json": "Summary of all batch processing results",
            "dataset_info.json": "Enhanced dataset metadata",
            "coupling_analysis_config.json": "Analysis configuration parameters",
            "final_merged_results/": "Complete merged analysis results (~350MB)",
            "batch_XXXX/": f"Individual batch results ({batch_summary['total_batches']} batches)"
        }
    }
    
    with open("temp_complete_dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(enhanced_dataset_info, f, indent=2, ensure_ascii=False)
    
    # Create analysis config
    config = {
        "model_config": {
            "model_name": "meta-llama/Llama-2-7b-hf",
            "target_layers": ["model.layers.28", "model.layers.29", "model.layers.30", "model.layers.31"],
            "target_component": "mlp.down_proj",
            "layer_selection_reason": "Last 4 layers chosen for semantic richness and memory optimization"
        },
        "coupling_analysis": {
            "method": "gradient_cosine_similarity",
            "gradient_computation": "∇_θ log P(answer|question)",
            "normalization": "L2 normalization",
            "high_coupling_threshold": 0.4,
            "batch_size": 2000,
            "memory_optimization": True
        },
        "batch_processing": {
            "total_batches": batch_summary["total_batches"],
            "batch_size": 2000,
            "samples_per_batch": "~2000 (variable for last batch)",
            "processing_approach": "Sequential batch processing for memory efficiency"
        },
        "dataset_processing": {
            "source_dataset": "hotpotqa",
            "total_samples": 97852,
            "format": "cloze_style_questions",
            "question_template": "Given the context: {context}, the answer to '{question}' is [MASK]."
        },
        "hardware_specs": {
            "gpu": "NVIDIA A40",
            "vram": "46GB",
            "gpu_memory_allocated": "~21GB during analysis",
            "gpu_memory_reserved": "~43GB during analysis"
        }
    }
    
    with open("temp_complete_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # Upload supporting files first
    supporting_files = [
        {
            "local_path": "temp_complete_README.md",
            "repo_path": "README.md",
            "description": "Complete dataset documentation"
        },
        {
            "local_path": "temp_batch_summary.json",
            "repo_path": "batch_summary.json",
            "description": "Batch processing summary"
        },
        {
            "local_path": "temp_complete_dataset_info.json",
            "repo_path": "dataset_info.json",
            "description": "Enhanced dataset metadata"
        },
        {
            "local_path": "temp_complete_config.json",
            "repo_path": "coupling_analysis_config.json",
            "description": "Complete analysis configuration"
        }
    ]
    
    successful_uploads = []
    failed_uploads = []
    
    # Upload supporting files
    for file_info in supporting_files:
        local_path = file_info["local_path"]
        repo_path = file_info["repo_path"]
        description = file_info["description"]
        
        try:
            print(f"Uploading {repo_path}...")
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=full_repo_name,
                repo_type="dataset",
                commit_message=f"Add {description}"
            )
            print(f"✓ Successfully uploaded: {repo_path}")
            successful_uploads.append(repo_path)
            
        except Exception as e:
            print(f"✗ Failed to upload {repo_path}: {e}")
            failed_uploads.append(repo_path)
    
    # Upload the entire results directory
    print(f"\n📁 Uploading complete results directory...")
    print(f"   This includes all batch results and final merged data")
    
    try:
        # Upload the entire results folder
        upload_folder(
            folder_path="results/full_hotpotqa_analysis",
            repo_id=full_repo_name,
            repo_type="dataset",
            commit_message="Add complete knowledge coupling analysis results with all batch data"
        )
        print(f"✓ Successfully uploaded complete results directory")
        successful_uploads.append("results/full_hotpotqa_analysis/")
        
    except Exception as e:
        print(f"✗ Failed to upload results directory: {e}")
        failed_uploads.append("results/full_hotpotqa_analysis/")
    
    # Clean up temporary files
    for temp_file in ["temp_complete_README.md", "temp_batch_summary.json", 
                      "temp_complete_dataset_info.json", "temp_complete_config.json"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    # Summary
    print(f"\n📊 Upload Summary:")
    print(f"✓ Successfully uploaded: {len(successful_uploads)} items")
    if successful_uploads:
        for item in successful_uploads:
            print(f"  - {item}")
    
    if failed_uploads:
        print(f"✗ Failed uploads: {len(failed_uploads)} items")
        for item in failed_uploads:
            print(f"  - {item}")
    
    if successful_uploads:
        # Calculate approximate total size
        total_size_info = f"""
📈 Dataset Statistics:
   - Total Batches: {batch_summary['total_batches']}
   - Knowledge Pieces: {batch_summary['aggregate_statistics']['total_knowledge_pieces']:,}
   - Coupling Pairs: {batch_summary['aggregate_statistics']['total_coupling_pairs']:,}
   - Mean Coupling: {batch_summary['aggregate_statistics']['overall_mean_coupling']:.4f}
        """
        print(total_size_info)
        
        print(f"\n🎉 Complete dataset is now available at:")
        print(f"   https://huggingface.co/datasets/{full_repo_name}")
        return True
    else:
        print(f"\n❌ No files were successfully uploaded.")
        return False

if __name__ == "__main__":
    print("🚀 Starting Complete HuggingFace Upload Process...")
    print("   This will upload ALL batch results and final merged data")
    print("=" * 70)
    
    # Configuration
    repo_name = "hotpotqa-knowledge-coupling-complete"
    private_repo = False  # Set to True if you want a private dataset
    
    # Check if we're logged in
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"✓ Logged in as: {user_info['name']}")
    except Exception as e:
        print(f"❌ Not logged in to HuggingFace: {e}")
        print("Please run: huggingface-cli login")
        exit(1)
    
    # Check if results exist
    if not os.path.exists("results/full_hotpotqa_analysis"):
        print(f"❌ Results directory not found: results/full_hotpotqa_analysis")
        print("Please ensure the analysis has been completed first.")
        exit(1)
    
    # Count batches
    batch_dirs = glob.glob("results/full_hotpotqa_analysis/batch_*")
    print(f"📊 Found {len(batch_dirs)} batch directories to upload")
    
    # Confirm with user
    print(f"\nThis will upload approximately 2-3GB of data including:")
    print(f"  - {len(batch_dirs)} batch directories")
    print(f"  - Final merged results (~350MB)")
    print(f"  - Complete metadata and documentation")
    
    # Upload results
    success = upload_complete_results_to_huggingface(
        repo_name=repo_name,
        private=private_repo
    )
    
    if success:
        print("\n🎉 Complete upload finished successfully!")
        print("Your complete knowledge coupling analysis dataset is now available! 🤗")
        print("\nThis includes:")
        print("  ✓ All batch processing results")
        print("  ✓ Final merged analysis data")
        print("  ✓ Complete documentation and metadata")
        print("  ✓ Batch processing summaries")
    else:
        print("\n❌ Upload failed. Please check the errors above.") 