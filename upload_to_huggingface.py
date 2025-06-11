#!/usr/bin/env python3
"""
Upload Knowledge Coupling Analysis Results to HuggingFace
"""

import os
import json
from huggingface_hub import HfApi, create_repo, upload_file
from datetime import datetime

def create_dataset_card():
    """Create a comprehensive dataset card for the knowledge coupling results."""
    
    card_content = """
---
language:
- en
license: mit
size_categories:
- 10K<n<100K
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
---

# Knowledge Coupling Analysis on HotpotQA Dataset

## Dataset Description

This dataset contains the results of a comprehensive knowledge coupling analysis performed on the HotpotQA dataset using LLaMA2-7B model. The analysis investigates how different pieces of knowledge interact within the model's parameter space through gradient-based coupling measurements.

## Research Overview

- **Model**: meta-llama/Llama-2-7b-hf (layers 28-31 focused analysis)
- **Dataset**: HotpotQA (train + dev splits, 97,852 total samples)
- **Method**: Gradient-based knowledge coupling via cosine similarity
- **Target Layers**: model.layers.28-31.mlp.down_proj (semantically rich layers)

## Key Findings

The analysis revealed:
- Mean coupling score: 0.0222 across all knowledge piece pairs
- High coupling pairs (≥0.4 threshold): Critical for ripple effect prediction
- Layer-specific analysis focusing on MLP down-projection layers
- Comprehensive gradient analysis with 180,355,072 dimensions per knowledge piece

## Files Description

### Core Results
- `global_analysis_results.json`: Comprehensive analysis summary with statistics
- `all_knowledge_pieces.json`: Complete set of processed knowledge pieces (92MB)
- `all_coupling_pairs.csv`: All pairwise coupling measurements (245MB)

### Supporting Files
- `dataset_info.json`: Dataset statistics and conversion details
- `coupling_analysis_config.json`: Analysis configuration and parameters

## Usage

```python
from datasets import load_dataset

# Load the knowledge coupling results
dataset = load_dataset("your-username/hotpotqa-knowledge-coupling")

# Access global analysis results
global_results = dataset["global_analysis"]

# Access knowledge pieces
knowledge_pieces = dataset["knowledge_pieces"]

# Access coupling pairs
coupling_pairs = dataset["coupling_pairs"]
```

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{hotpotqa_knowledge_coupling,
  title={Knowledge Coupling Analysis on HotpotQA Dataset using LLaMA2-7B},
  author={[Your Name]},
  year={2024},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/your-username/hotpotqa-knowledge-coupling}
}
```

## Technical Details

- **Gradient Computation**: ∇_θ log P(answer|question) for cloze-style questions
- **Coupling Measurement**: Cosine similarity between L2-normalized gradients
- **Memory Optimization**: Focused on layers 28-31 to handle GPU memory constraints
- **Hardware**: NVIDIA A40 GPU (46GB VRAM)

## License

This dataset is released under the MIT License. The original HotpotQA dataset follows its respective licensing terms.
"""
    
    return card_content

def create_dataset_info():
    """Create dataset information file."""
    
    # Read the global analysis results to get statistics
    results_path = "results/full_hotpotqa_analysis/final_merged_results/global_analysis_results.json"
    
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            global_results = json.load(f)
    else:
        global_results = {}
    
    dataset_info = {
        "dataset_name": "hotpotqa-knowledge-coupling",
        "description": "Knowledge coupling analysis results on HotpotQA dataset using LLaMA2-7B",
        "creation_date": datetime.now().isoformat(),
        "model_info": {
            "model_name": "meta-llama/Llama-2-7b-hf",
            "target_layers": "model.layers.28-31.mlp.down_proj",
            "total_parameters_analyzed": "180,355,072 per knowledge piece"
        },
        "dataset_statistics": global_results.get("dataset_stats", {}),
        "analysis_results": {
            "total_knowledge_pieces": global_results.get("total_knowledge_pieces", "Unknown"),
            "total_coupling_pairs": global_results.get("total_coupling_pairs", "Unknown"),
            "mean_coupling_score": global_results.get("mean_coupling", "Unknown"),
            "high_coupling_threshold": 0.4
        },
        "files": {
            "global_analysis_results.json": "Comprehensive analysis summary",
            "all_knowledge_pieces.json": "Complete knowledge pieces data (~92MB)",
            "all_coupling_pairs.csv": "All pairwise coupling measurements (~245MB)",
            "dataset_info.json": "This file - dataset metadata",
            "coupling_analysis_config.json": "Analysis configuration parameters"
        }
    }
    
    return dataset_info

def create_analysis_config():
    """Create analysis configuration file."""
    
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
    
    return config

def upload_results_to_huggingface(repo_name, username=None, private=False):
    """Upload all results to HuggingFace dataset."""
    
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
    
    # Prepare files to upload
    results_dir = "results/full_hotpotqa_analysis/final_merged_results"
    
    files_to_upload = [
        {
            "local_path": f"{results_dir}/global_analysis_results.json",
            "repo_path": "global_analysis_results.json",
            "description": "Global analysis results"
        },
        {
            "local_path": f"{results_dir}/all_knowledge_pieces.json", 
            "repo_path": "all_knowledge_pieces.json",
            "description": "All knowledge pieces data"
        },
        {
            "local_path": f"{results_dir}/all_coupling_pairs.csv",
            "repo_path": "all_coupling_pairs.csv", 
            "description": "All coupling pairs measurements"
        }
    ]
    
    # Create and upload supporting files
    print("Creating dataset metadata files...")
    
    # Create README.md (dataset card)
    with open("temp_README.md", "w", encoding="utf-8") as f:
        f.write(create_dataset_card())
    
    # Create dataset_info.json
    with open("temp_dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(create_dataset_info(), f, indent=2, ensure_ascii=False)
    
    # Create coupling_analysis_config.json
    with open("temp_coupling_analysis_config.json", "w", encoding="utf-8") as f:
        json.dump(create_analysis_config(), f, indent=2, ensure_ascii=False)
    
    # Add supporting files to upload list
    files_to_upload.extend([
        {
            "local_path": "temp_README.md",
            "repo_path": "README.md",
            "description": "Dataset card and documentation"
        },
        {
            "local_path": "temp_dataset_info.json",
            "repo_path": "dataset_info.json", 
            "description": "Dataset metadata and statistics"
        },
        {
            "local_path": "temp_coupling_analysis_config.json",
            "repo_path": "coupling_analysis_config.json",
            "description": "Analysis configuration parameters"
        }
    ])
    
    # Upload files
    successful_uploads = []
    failed_uploads = []
    
    for file_info in files_to_upload:
        local_path = file_info["local_path"]
        repo_path = file_info["repo_path"]
        description = file_info["description"]
        
        if not os.path.exists(local_path):
            print(f"⚠️  File not found: {local_path}")
            failed_uploads.append(local_path)
            continue
        
        try:
            print(f"Uploading {repo_path}...")
            file_size = os.path.getsize(local_path)
            print(f"  File size: {file_size / (1024*1024):.1f} MB")
            
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
    
    # Clean up temporary files
    for temp_file in ["temp_README.md", "temp_dataset_info.json", "temp_coupling_analysis_config.json"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    # Summary
    print(f"\n📊 Upload Summary:")
    print(f"✓ Successfully uploaded: {len(successful_uploads)} files")
    if successful_uploads:
        for file in successful_uploads:
            print(f"  - {file}")
    
    if failed_uploads:
        print(f"✗ Failed uploads: {len(failed_uploads)} files")
        for file in failed_uploads:
            print(f"  - {file}")
    
    if successful_uploads:
        print(f"\n🎉 Dataset is now available at:")
        print(f"   https://huggingface.co/datasets/{full_repo_name}")
        return True
    else:
        print(f"\n❌ No files were successfully uploaded.")
        return False

if __name__ == "__main__":
    print("🚀 Starting HuggingFace upload process...")
    print("=" * 60)
    
    # Configuration
    repo_name = "hotpotqa-knowledge-coupling"
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
    
    # Upload results
    success = upload_results_to_huggingface(
        repo_name=repo_name,
        private=private_repo
    )
    
    if success:
        print("\n🎉 Upload completed successfully!")
        print("Your knowledge coupling analysis is now available on HuggingFace! 🤗")
    else:
        print("\n❌ Upload failed. Please check the errors above.") 