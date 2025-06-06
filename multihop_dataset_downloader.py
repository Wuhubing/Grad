#!/usr/bin/env python3
"""
Multi-hop Reasoning Dataset Downloader and Converter

支持下载和转换多个多跳推理数据集：
- HotpotQA
- Musique
- WikiHop
- Bamboogle

统一转换为知识耦合分析系统需要的格式
"""

import json
import os
import requests
import zipfile
import tarfile
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import random

class MultihopDatasetDownloader:
    """多跳推理数据集下载器和转换器"""
    
    # 数据集配置
    DATASETS = {
        'hotpotqa': {
            'name': 'HotpotQA',
            'urls': {
                'train': 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json',
                'dev': 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json'
            },
            'format': 'json',
            'description': 'Multi-hop reasoning over Wikipedia'
        },
        'musique': {
            'name': 'MuSiQue',
            'urls': {
                'main': 'https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing'
            },
            'format': 'gdrive_zip',
            'description': 'Multi-hop questions with single supporting facts'
        },
        'wikihop': {
            'name': 'WikiHop',
            'urls': {
                'huggingface': 'huggingface://wiki_hop'
            },
            'format': 'huggingface_trust',
            'description': 'Reading comprehension with multiple documents'
        },
        'bamboogle': {
            'name': 'Bamboogle',
            'urls': {
                'google_sheets': 'https://docs.google.com/spreadsheets/d/1jwcsA5kE4TObr9YHn9Gc-wQHYjTbLhDGx6tmIzMhl_U/export?format=csv'
            },
            'format': 'csv',
            'description': 'Multi-hop QA with synthetic data'
        }
    }
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        print(f"🗂️  Multi-hop Dataset Downloader initialized")
        print(f"   Data directory: {self.data_dir.absolute()}")
        print(f"   Supported datasets: {', '.join(self.DATASETS.keys())}")
    
    def download_dataset(self, dataset_name: str, force_download: bool = False) -> bool:
        """下载指定数据集"""
        if dataset_name not in self.DATASETS:
            print(f"❌ Unknown dataset: {dataset_name}")
            print(f"   Available: {', '.join(self.DATASETS.keys())}")
            return False
        
        dataset_config = self.DATASETS[dataset_name]
        dataset_dir = self.data_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        print(f"\n📥 Downloading {dataset_config['name']}...")
        print(f"   Description: {dataset_config['description']}")
        
        success = True
        for split, url in dataset_config['urls'].items():
            # 处理HuggingFace数据集
            if dataset_config['format'] == 'huggingface':
                try:
                    print(f"   🔄 Loading {split} from HuggingFace")
                    # 安装datasets库
                    import subprocess
                    subprocess.run(['pip', 'install', 'datasets'], check=True, capture_output=True)
                    
                    from datasets import load_dataset
                    
                    # 提取数据集名称
                    dataset_name_hf = url.replace('huggingface://', '')
                    
                    # 加载数据集
                    dataset = load_dataset(dataset_name_hf)
                    
                    # 保存为JSON文件
                    for split_name, split_data in dataset.items():
                        output_file = dataset_dir / f"{dataset_name_hf}_{split_name}.json"
                        split_data.to_json(output_file)
                        print(f"   ✅ Saved {split_name}: {output_file}")
                    
                except Exception as e:
                    print(f"   ❌ Failed to load from HuggingFace: {e}")
                    success = False
                continue
                
            # 处理HuggingFace数据集（需要信任远程代码）
            if dataset_config['format'] == 'huggingface_trust':
                try:
                    print(f"   🔄 Loading {split} from HuggingFace (trust_remote_code=True)")
                    # 安装datasets库
                    import subprocess
                    subprocess.run(['pip', 'install', 'datasets'], check=True, capture_output=True)
                    
                    from datasets import load_dataset
                    
                    # 提取数据集名称
                    dataset_name_hf = url.replace('huggingface://', '')
                    
                    # 加载数据集，信任远程代码
                    dataset = load_dataset(dataset_name_hf, trust_remote_code=True)
                    
                    # 保存为JSON文件
                    for split_name, split_data in dataset.items():
                        output_file = dataset_dir / f"{dataset_name_hf}_{split_name}.json"
                        split_data.to_json(output_file)
                        print(f"   ✅ Saved {split_name}: {output_file}")
                    
                except Exception as e:
                    print(f"   ❌ Failed to load from HuggingFace: {e}")
                    success = False
                continue
                
            # 处理Google Drive链接
            if dataset_config['format'] == 'gdrive_zip':
                # 对于Google Drive，使用gdown下载
                try:
                    import subprocess
                    print(f"   🔄 Downloading {split} via Google Drive")
                    
                    # 提取Google Drive文件ID
                    if 'drive.google.com' in url:
                        if '/file/d/' in url:
                            file_id = url.split('/file/d/')[1].split('/')[0]
                        else:
                            print(f"   ❌ Cannot extract file ID from: {url}")
                            success = False
                            continue
                    
                    # 设置输出文件名
                    output_filename = f"musique_v1.0.zip"
                    output_path = dataset_dir / output_filename
                    
                    if output_path.exists() and not force_download:
                        print(f"   ✅ {split} already exists: {output_path}")
                    else:
                        # 确保安装gdown
                        subprocess.run(['pip', 'install', 'gdown'], check=True, capture_output=True)
                        
                        # 使用gdown下载
                        cmd = ['gdown', '--id', file_id, '--output', str(output_path)]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if result.returncode != 0:
                            print(f"   ❌ gdown failed: {result.stderr}")
                            success = False
                            continue
                        
                        print(f"   ✅ Downloaded {split}: {output_path}")
                    
                    # 解压ZIP文件
                    if output_path.exists():
                        self._extract_zip(output_path, dataset_dir)
                        # 清理下载的zip文件
                        output_path.unlink()
                    
                except Exception as e:
                    print(f"   ❌ Failed to download {split} via Google Drive: {e}")
                    success = False
                continue
                
            # 原有的下载逻辑
            filename = self._get_filename_from_url(url)
            file_path = dataset_dir / filename
            
            if file_path.exists() and not force_download:
                print(f"   ✅ {split} already exists: {file_path}")
                continue
            
            try:
                print(f"   🔄 Downloading {split} from {url}")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(file_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"   {split}") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                # 解压缩文件
                if dataset_config['format'] == 'zip':
                    self._extract_zip(file_path, dataset_dir)
                elif dataset_config['format'] == 'tar.gz':
                    self._extract_tar(file_path, dataset_dir)
                
                print(f"   ✅ Downloaded {split}: {file_path}")
                
            except Exception as e:
                print(f"   ❌ Failed to download {split}: {e}")
                success = False
        
        return success
    
    def _get_filename_from_url(self, url: str) -> str:
        """从URL提取文件名"""
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        if not filename:
            filename = url.split('/')[-1]
        return filename or "data.json"
    
    def _extract_zip(self, zip_path: Path, extract_dir: Path):
        """解压ZIP文件"""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"   📦 Extracted: {zip_path}")
    
    def _extract_tar(self, tar_path: Path, extract_dir: Path):
        """解压TAR文件"""
        with tarfile.open(tar_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_dir)
        print(f"   📦 Extracted: {tar_path}")
    
    def convert_hotpotqa(self, file_path: Path, max_samples: int = None) -> List[Dict]:
        """转换HotpotQA格式"""
        print(f"🔄 Converting HotpotQA: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        converted = []
        for i, item in enumerate(data):
            # HotpotQA已经是我们需要的格式
            converted_item = {
                '_id': item.get('_id', f'hotpotqa_{i}'),
                'question': item['question'],
                'answer': item['answer'],
                'supporting_facts': item.get('supporting_facts', []),
                'context': item.get('context', []),
                'dataset': 'hotpotqa',
                'level': item.get('level', 'hard'),
                'type': item.get('type', 'bridge')
            }
            converted.append(converted_item)
        
        print(f"   ✅ Converted {len(converted)} HotpotQA samples")
        return converted
    
    def convert_musique(self, file_path: Path, max_samples: int = None) -> List[Dict]:
        """转换MuSiQue格式"""
        print(f"🔄 Converting MuSiQue: {file_path}")
        
        # MuSiQue数据集可能是JSON或JSONL格式
        converted = []
        
        try:
            # 尝试作为JSON文件读取
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.jsonl':
                    # JSONL格式
                    for i, line in enumerate(f):
                        if max_samples and i >= max_samples:
                            break
                        
                        try:
                            item = json.loads(line.strip())
                            converted_item = self._convert_musique_item(item, i)
                            if converted_item:
                                converted.append(converted_item)
                        except Exception as e:
                            print(f"   ⚠️  Error processing line {i}: {e}")
                            continue
                else:
                    # JSON格式
                    data = json.load(f)
                    if isinstance(data, list):
                        # 数据是列表
                        for i, item in enumerate(data):
                            if max_samples and i >= max_samples:
                                break
                            converted_item = self._convert_musique_item(item, i)
                            if converted_item:
                                converted.append(converted_item)
                    else:
                        # 数据可能有其他结构
                        print(f"   ⚠️  Unknown JSON structure in {file_path}")
                        return []
                    
        except Exception as e:
            print(f"   ❌ Error reading {file_path}: {e}")
            return []
        
        print(f"   ✅ Converted {len(converted)} MuSiQue samples")
        return converted
    
    def _convert_musique_item(self, item: Dict, index: int) -> Dict:
        """转换单个MuSiQue数据项"""
        try:
            # 提取supporting facts
            supporting_facts = []
            if 'supporting_facts' in item:
                for fact in item['supporting_facts']:
                    if isinstance(fact, dict):
                        supporting_facts.append([fact.get('title', ''), fact.get('sent_id', 0)])
                    elif isinstance(fact, list) and len(fact) >= 2:
                        supporting_facts.append([fact[0], fact[1]])
            
            # 构建context
            context = []
            if 'paragraphs' in item:
                for para in item['paragraphs']:
                    title = para.get('title', f'para_{len(context)}')
                    if 'paragraph_text' in para:
                        sentences = para['paragraph_text'].split('. ')
                    else:
                        sentences = para.get('sentences', [])
                    context.append([title, sentences])
            elif 'context' in item:
                # 如果context已经是我们需要的格式
                context = item['context']
            
            converted_item = {
                '_id': item.get('id', item.get('_id', f'musique_{index}')),
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
                'supporting_facts': supporting_facts,
                'context': context,
                'dataset': 'musique',
                'question_type': item.get('question_type', 'multihop'),
                'answerable': item.get('answerable', True)
            }
            return converted_item
            
        except Exception as e:
            print(f"   ⚠️  Error converting item {index}: {e}")
            return None
    
    def convert_wikihop(self, file_path: Path, max_samples: int = None) -> List[Dict]:
        """转换WikiHop格式"""
        print(f"🔄 Converting WikiHop: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        converted = []
        for i, item in enumerate(data):
            try:
                # WikiHop格式转换
                supports = item.get('supports', [])
                supporting_facts = []
                
                # 创建supporting facts
                for j, support_id in enumerate(supports[:2]):  # 只取前2个
                    supporting_facts.append([f'support_{j}', 0])
                
                # 构建context
                context = []
                candidates = item.get('candidates', [])
                documents = item.get('supports', [])
                
                # 添加候选答案作为context
                if candidates:
                    context.append(['candidates', candidates[:5]])  # 只取前5个候选
                
                # 添加支持文档
                for j, doc in enumerate(documents[:3]):  # 只取前3个文档
                    sentences = doc.split('. ')[:3]  # 每个文档只取前3句
                    context.append([f'document_{j}', sentences])
                
                converted_item = {
                    '_id': item.get('id', f'wikihop_{i}'),
                    'question': item.get('query', ''),
                    'answer': item.get('answer', ''),
                    'supporting_facts': supporting_facts,
                    'context': context,
                    'dataset': 'wikihop',
                    'candidates': candidates,
                    'query_type': 'bridge'
                }
                converted.append(converted_item)
                
            except Exception as e:
                print(f"   ⚠️  Error processing item {i}: {e}")
                continue
        
        print(f"   ✅ Converted {len(converted)} WikiHop samples")
        return converted
    
    def convert_bamboogle(self, file_path: Path, max_samples: int = None) -> List[Dict]:
        """转换Bamboogle格式"""
        print(f"🔄 Converting Bamboogle: {file_path}")
        
        # Bamboogle可能是CSV或JSON格式
        converted = []
        
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
                if max_samples:
                    df = df.head(max_samples)
                
                for i, row in df.iterrows():
                    # 基于CSV列构建数据
                    converted_item = {
                        '_id': f'bamboogle_{i}',
                        'question': row.get('question', ''),
                        'answer': row.get('answer', ''),
                        'supporting_facts': [['context', 0]],  # 简化的supporting facts
                        'context': [['context', [row.get('context', '')]]],
                        'dataset': 'bamboogle',
                        'difficulty': row.get('difficulty', 'medium')
                    }
                    converted.append(converted_item)
            else:
                # JSON格式
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if max_samples:
                    data = data[:max_samples]
                
                for i, item in enumerate(data):
                    converted_item = {
                        '_id': item.get('id', f'bamboogle_{i}'),
                        'question': item.get('question', ''),
                        'answer': item.get('answer', ''),
                        'supporting_facts': item.get('supporting_facts', [['context', 0]]),
                        'context': item.get('context', [['context', ['No context available']]]),
                        'dataset': 'bamboogle',
                        'type': item.get('type', 'synthetic')
                    }
                    converted.append(converted_item)
                    
        except Exception as e:
            print(f"   ❌ Error converting Bamboogle: {e}")
            return []
        
        print(f"   ✅ Converted {len(converted)} Bamboogle samples")
        return converted
    
    def convert_dataset(self, dataset_name: str, split: str = 'train', 
                       max_samples: int = None) -> List[Dict]:
        """转换指定数据集的指定分割"""
        dataset_dir = self.data_dir / dataset_name
        
        if dataset_name == 'hotpotqa':
            if split == 'train':
                file_path = dataset_dir / "hotpot_train_v1.1.json"
            else:  # dev
                file_path = dataset_dir / "hotpot_dev_distractor_v1.json"
            
            if file_path.exists():
                return self.convert_hotpotqa(file_path, max_samples)
        
        elif dataset_name == 'musique':
            # MuSiQue文件在data子目录中
            data_dir = dataset_dir / "data"
            if split == 'train':
                file_path = data_dir / "musique_ans_v1.0_train.jsonl"
            elif split == 'dev':
                file_path = data_dir / "musique_ans_v1.0_dev.jsonl"
            else:  # test
                file_path = data_dir / "musique_ans_v1.0_test.jsonl"
            
            if file_path.exists():
                return self.convert_musique(file_path, max_samples)
        
        elif dataset_name == 'wikihop':
            # WikiHop从HuggingFace下载后的文件
            train_file = dataset_dir / "wiki_hop_train.json"
            dev_file = dataset_dir / "wiki_hop_validation.json"
            
            if split == 'train' and train_file.exists():
                return self.convert_wikihop(train_file, max_samples)
            elif split == 'dev' and dev_file.exists():
                return self.convert_wikihop(dev_file, max_samples)
            else:
                # 备用：查找任何JSON文件
                possible_files = list(dataset_dir.glob("**/*.json"))
                if possible_files:
                    return self.convert_wikihop(possible_files[0], max_samples)
        
        elif dataset_name == 'bamboogle':
            # Bamboogle从Google Sheets下载的CSV文件
            bamboogle_file = dataset_dir / "bamboogle.csv"
            if bamboogle_file.exists():
                return self.convert_bamboogle(bamboogle_file, max_samples)
            else:
                # 备用：查找任何CSV文件
                possible_files = list(dataset_dir.glob("**/*.csv"))
                if possible_files:
                    return self.convert_bamboogle(possible_files[0], max_samples)
        
        print(f"❌ No data file found for {dataset_name} ({split})")
        return []
    
    def create_mixed_dataset(self, datasets: List[str], samples_per_dataset: int = 50,
                           output_file: str = "mixed_multihop_dataset.json") -> List[Dict]:
        """创建混合数据集"""
        print(f"\n🔀 Creating mixed dataset from: {', '.join(datasets)}")
        
        all_data = []
        
        for dataset_name in datasets:
            print(f"\n📚 Processing {dataset_name}...")
            converted_data = self.convert_dataset(dataset_name, max_samples=samples_per_dataset)
            all_data.extend(converted_data)
        
        # 随机打乱
        random.shuffle(all_data)
        
        # 保存混合数据集
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Mixed dataset created!")
        print(f"   Total samples: {len(all_data)}")
        print(f"   Saved to: {output_file}")
        
        # 统计各数据集的样本数
        dataset_counts = {}
        for item in all_data:
            dataset = item.get('dataset', 'unknown')
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        
        print(f"   Dataset distribution:")
        for dataset, count in dataset_counts.items():
            print(f"     {dataset}: {count} samples")
        
        return all_data
    
    def download_all_datasets(self, force_download: bool = False):
        """下载所有支持的数据集"""
        print(f"\n🚀 Downloading all multihop datasets...")
        
        for dataset_name in self.DATASETS:
            print(f"\n" + "="*50)
            success = self.download_dataset(dataset_name, force_download)
            if success:
                print(f"✅ {dataset_name} downloaded successfully")
            else:
                print(f"❌ {dataset_name} download failed")

def main():
    """主函数 - 演示用法"""
    print("🚀 Multi-hop Dataset Downloader and Converter")
    print("=" * 60)
    
    downloader = MultihopDatasetDownloader()
    
    # 下载所有四个数据集
    all_datasets = ['hotpotqa', 'musique', 'wikihop', 'bamboogle']
    
    print(f"\n📥 Step 1: Downloading all datasets ({', '.join(all_datasets)})...")
    for dataset in all_datasets:
        print(f"\n{'='*30}")
        success = downloader.download_dataset(dataset)
        if success:
            print(f"✅ {dataset} downloaded successfully")
        else:
            print(f"❌ {dataset} download failed - continuing with next dataset")
    
    print(f"\n🔄 Step 2: Creating mixed dataset from all available datasets...")
    # 只使用成功下载的数据集
    available_datasets = []
    for dataset in all_datasets:
        dataset_dir = downloader.data_dir / dataset
        if dataset_dir.exists():
            available_datasets.append(dataset)
    
    if available_datasets:
        mixed_data = downloader.create_mixed_dataset(
            datasets=available_datasets,
            samples_per_dataset=25,  # 每个数据集25个样本，4个数据集总共100个样本
            output_file="mixed_multihop_100.json"
        )
        
        print(f"\n🎯 Step 3: Dataset ready for knowledge coupling analysis!")
        print(f"   Use: python knowledge_coupling_mvp.py --hotpot_data mixed_multihop_100.json")
    else:
        print(f"\n❌ No datasets available for mixed dataset creation")
    
    print(f"\n📊 Final download summary:")
    for dataset in all_datasets:
        dataset_dir = downloader.data_dir / dataset
        if dataset_dir.exists():
            print(f"   ✅ {dataset}: Downloaded")
        else:
            print(f"   ❌ {dataset}: Failed or not available")

if __name__ == "__main__":
    main() 