#!/usr/bin/env python3
"""
转换HotpotQA数据集为知识耦合分析格式

将下载的HotpotQA数据集转换为我们系统期望的格式
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import os

class HotpotQAConverter:
    """HotpotQA数据集转换器"""
    
    def __init__(self, data_dir: str = "datasets/hotpotqa", output_dir: str = "datasets/processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🗂️  HotpotQA Converter initialized")
        print(f"   Input directory: {self.data_dir.absolute()}")
        print(f"   Output directory: {self.output_dir.absolute()}")
    
    def load_hotpotqa_file(self, file_path: Path) -> List[Dict]:
        """加载HotpotQA JSON文件"""
        print(f"📚 Loading {file_path.name}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"   ✅ Loaded {len(data)} samples")
        return data
    
    def convert_hotpotqa_to_coupling_format(self, hotpot_data: List[Dict], 
                                          max_samples: Optional[int] = None) -> List[Dict]:
        """转换HotpotQA格式为知识耦合分析格式"""
        print(f"🔄 Converting HotpotQA to coupling format...")
        
        if max_samples:
            hotpot_data = hotpot_data[:max_samples]
            print(f"   限制到前 {max_samples} 个样本")
        
        converted_data = []
        
        for i, item in enumerate(tqdm(hotpot_data, desc="Converting")):
            try:
                # 提取基本信息
                question = item.get('question', '')
                answer = item.get('answer', '')
                supporting_facts = item.get('supporting_facts', [])
                context = item.get('context', [])
                
                # 确定类别 - 基于HotpotQA的level和type
                category = self._determine_category(item)
                
                # 计算hop数量
                hop_count = len(supporting_facts) if supporting_facts else 2
                
                # 创建标准格式
                converted_item = {
                    "_id": item.get('_id', f"hotpotqa_{i}"),
                    "question": question,
                    "answer": answer,
                    "supporting_facts": supporting_facts,
                    "context": context,
                    "category": category,
                    "hop_count": hop_count,
                    "type": item.get('type', 'bridge'),
                    "level": item.get('level', 'hard'),
                    "dataset": "hotpotqa"
                }
                
                # 验证数据完整性
                if self._validate_item(converted_item):
                    converted_data.append(converted_item)
                else:
                    print(f"   ⚠️  Skipping invalid item {i}: {item.get('_id', 'unknown')}")
                
            except Exception as e:
                print(f"   ❌ Error processing item {i}: {e}")
                continue
        
        print(f"✅ Successfully converted {len(converted_data)} samples")
        return converted_data
    
    def _determine_category(self, item: Dict) -> str:
        """确定样本的类别"""
        # 基于HotpotQA的type和level来确定类别
        item_type = item.get('type', 'bridge').lower()
        level = item.get('level', 'hard').lower()
        
        if item_type == 'bridge':
            if level == 'easy':
                return 'Bridge_Easy'
            elif level == 'medium':
                return 'Bridge_Medium'
            else:
                return 'Bridge_Hard'
        elif item_type == 'comparison':
            if level == 'easy':
                return 'Comparison_Easy'
            elif level == 'medium':
                return 'Comparison_Medium'
            else:
                return 'Comparison_Hard'
        else:
            return f"Unknown_{level.capitalize()}"
    
    def _validate_item(self, item: Dict) -> bool:
        """验证转换后的数据项是否有效"""
        required_fields = ['_id', 'question', 'answer', 'supporting_facts', 'context']
        
        for field in required_fields:
            if field not in item or not item[field]:
                if field in ['supporting_facts', 'context']:
                    # supporting_facts和context可以为空列表，但不能为None
                    if item[field] is None:
                        return False
                else:
                    # 其他字段不能为空
                    return False
        
        # 检查question和answer不能为空字符串
        if not item['question'].strip() or not item['answer'].strip():
            return False
        
        return True
    
    def analyze_converted_data(self, data: List[Dict]) -> Dict[str, Any]:
        """分析转换后的数据统计信息"""
        print(f"\n📊 数据集分析:")
        
        total_samples = len(data)
        
        # 分析类别分布
        category_counts = {}
        for item in data:
            category = item.get('category', 'Unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # 分析hop数量分布
        hop_counts = {}
        for item in data:
            hop_count = item.get('hop_count', 0)
            hop_counts[hop_count] = hop_counts.get(hop_count, 0) + 1
        
        # 分析类型分布
        type_counts = {}
        for item in data:
            item_type = item.get('type', 'unknown')
            type_counts[item_type] = type_counts.get(item_type, 0) + 1
        
        # 分析难度分布
        level_counts = {}
        for item in data:
            level = item.get('level', 'unknown')
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # 分析context长度
        context_lengths = [len(item.get('context', [])) for item in data]
        avg_context_length = sum(context_lengths) / len(context_lengths) if context_lengths else 0
        
        # 分析supporting_facts长度
        sf_lengths = [len(item.get('supporting_facts', [])) for item in data]
        avg_sf_length = sum(sf_lengths) / len(sf_lengths) if sf_lengths else 0
        
        stats = {
            'total_samples': total_samples,
            'category_distribution': category_counts,
            'hop_distribution': hop_counts,
            'type_distribution': type_counts,
            'level_distribution': level_counts,
            'avg_context_length': avg_context_length,
            'avg_supporting_facts_length': avg_sf_length
        }
        
        print(f"   总样本数: {total_samples}")
        print(f"   类别分布: {category_counts}")
        print(f"   Hop分布: {hop_counts}")
        print(f"   类型分布: {type_counts}")
        print(f"   难度分布: {level_counts}")
        print(f"   平均context长度: {avg_context_length:.2f}")
        print(f"   平均supporting facts长度: {avg_sf_length:.2f}")
        
        return stats
    
    def save_converted_data(self, data: List[Dict], output_file: str):
        """保存转换后的数据"""
        # 确保输出文件在正确的目录中
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 保存到: {output_path}")
        print(f"   样本数: {len(data)}")
    
    def create_sample_files(self, data: List[Dict], 
                           sizes: List[int] = [20, 50, 100, 500, 1000]):
        """创建不同大小的样本文件用于测试"""
        print(f"\n📦 创建不同大小的样本文件...")
        
        # 随机打乱数据以确保样本的多样性
        random.shuffle(data)
        
        for size in sizes:
            if size <= len(data):
                sample_data = data[:size]
                filename = f"hotpotqa_sample_{size}.json"
                self.save_converted_data(sample_data, filename)
            else:
                print(f"   ⚠️  跳过大小 {size} (数据总量: {len(data)})")
    
    def convert_all_splits(self, max_samples_per_split: Optional[int] = None):
        """转换所有数据分割"""
        print(f"\n🚀 开始转换所有数据分割...")
        
        all_converted_data = []
        
        # 转换train数据
        train_file = self.data_dir / "hotpot_train_v1.1.json"
        if train_file.exists():
            print(f"\n📂 处理训练集...")
            train_data = self.load_hotpotqa_file(train_file)
            
            # 限制样本数量（如果指定）
            if max_samples_per_split:
                train_data = train_data[:max_samples_per_split]
                print(f"   限制训练集到 {max_samples_per_split} 个样本")
            
            converted_train = self.convert_hotpotqa_to_coupling_format(train_data)
            
            # 保存训练集
            self.save_converted_data(converted_train, "hotpotqa_train_converted.json")
            
            # 分析数据
            train_stats = self.analyze_converted_data(converted_train)
            
            all_converted_data.extend(converted_train)
        
        # 转换dev数据
        dev_file = self.data_dir / "hotpot_dev_distractor_v1.json"
        if dev_file.exists():
            print(f"\n📂 处理验证集...")
            dev_data = self.load_hotpotqa_file(dev_file)
            
            # 限制样本数量（如果指定）
            if max_samples_per_split:
                dev_data = dev_data[:max_samples_per_split]
                print(f"   限制验证集到 {max_samples_per_split} 个样本")
            
            converted_dev = self.convert_hotpotqa_to_coupling_format(dev_data)
            
            # 保存验证集
            self.save_converted_data(converted_dev, "hotpotqa_dev_converted.json")
            
            # 分析数据
            dev_stats = self.analyze_converted_data(converted_dev)
            
            all_converted_data.extend(converted_dev)
        
        # 保存合并的数据
        if all_converted_data:
            print(f"\n📦 保存合并数据...")
            self.save_converted_data(all_converted_data, "hotpotqa_all_converted.json")
            
            # 创建不同大小的样本文件
            self.create_sample_files(all_converted_data)
            
            # 总体分析
            print(f"\n📊 总体数据分析:")
            overall_stats = self.analyze_converted_data(all_converted_data)
            
            # 保存统计信息
            stats_file = self.output_dir / "conversion_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(overall_stats, f, indent=2, ensure_ascii=False)
            print(f"📄 统计信息保存到: {stats_file}")
        
        return all_converted_data
    
    def show_sample_data(self, data: List[Dict], num_samples: int = 3):
        """显示示例数据"""
        print(f"\n👀 显示 {num_samples} 个示例数据:")
        
        for i, item in enumerate(data[:num_samples]):
            print(f"\n--- 示例 {i+1} ---")
            print(f"ID: {item.get('_id')}")
            print(f"类别: {item.get('category')}")
            print(f"类型: {item.get('type')}")
            print(f"难度: {item.get('level')}")
            print(f"Hop数: {item.get('hop_count')}")
            print(f"问题: {item.get('question')[:100]}...")
            print(f"答案: {item.get('answer')}")
            print(f"Supporting facts数量: {len(item.get('supporting_facts', []))}")
            print(f"Context数量: {len(item.get('context', []))}")

def main():
    """主函数"""
    print("🔄 HotpotQA数据集转换器")
    print("=" * 60)
    
    # 初始化转换器
    converter = HotpotQAConverter()
    
    # 转换所有数据分割，处理全部数据
    max_samples = None  # 处理全部数据，不限制样本数量
    
    print(f"⚙️  配置:")
    print(f"   处理模式: 全部数据 (无限制)")
    print(f"   输出目录: datasets/processed/")
    
    # 执行转换
    all_data = converter.convert_all_splits(max_samples_per_split=max_samples)
    
    if all_data:
        # 显示示例数据
        converter.show_sample_data(all_data, num_samples=2)
        
        print(f"\n🎉 转换完成!")
        print(f"   总共处理: {len(all_data)} 个样本")
        print(f"   输出文件保存在: datasets/processed/")
        print(f"   创建的文件:")
        
        # 列出创建的文件
        output_dir = Path("datasets/processed")
        if output_dir.exists():
            for file in sorted(output_dir.glob("*.json")):
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"     - {file.name} ({size_mb:.2f}MB)")
    else:
        print(f"❌ 没有找到可转换的数据文件")

if __name__ == "__main__":
    main() 