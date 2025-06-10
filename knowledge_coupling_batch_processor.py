#!/usr/bin/env python3
"""
知识耦合批处理器 - 处理全部HotpotQA数据
支持分批处理、内存管理、checkpoint保存/恢复
优化版：使用小子批次（10个样本）避免显存不足
"""

import json
import os
import gc
import psutil
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import datetime
import time

# 导入我们的核心类
from knowledge_coupling_mvp import MultiModelKnowledgeCouplingMVP, load_hotpot_data

class KnowledgeCouplingBatchProcessor:
    """批处理知识耦合分析器 - 优化版"""
    
    def __init__(self, 
                 model_path: str = "meta-llama/Llama-2-7b-hf",
                 batch_size: int = 2000,
                 sub_batch_size: int = 10,  # 新增：子批次大小
                 checkpoint_dir: str = "checkpoints",
                 output_dir: str = "results/full_hotpotqa_analysis",
                 layer_range: Optional[Tuple[int, int]] = None):
        
        self.model_path = model_path
        self.batch_size = batch_size
        self.sub_batch_size = sub_batch_size  # 每次处理的样本数
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.layer_range = layer_range
        
        # 创建目录
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # 内存和GPU监控
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
        
        print(f"🚀 初始化知识耦合批处理器 (优化版)")
        print(f"模型: {model_path}")
        print(f"批处理大小: {batch_size}")
        print(f"子批次大小: {sub_batch_size} (减少显存使用)")
        print(f"检查点目录: {checkpoint_dir}")
        print(f"输出目录: {output_dir}")
        print(f"初始内存使用: {self.initial_memory:.2f} GB")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def get_memory_usage(self) -> float:
        """获取当前内存使用量(GB)"""
        return self.process.memory_info().rss / 1e9
    
    def get_gpu_memory_usage(self) -> Tuple[float, float]:
        """获取GPU显存使用量(GB)"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            return allocated, reserved
        return 0.0, 0.0
    
    def print_memory_status(self, prefix: str = ""):
        """打印内存状态"""
        cpu_mem = self.get_memory_usage()
        gpu_alloc, gpu_reserved = self.get_gpu_memory_usage()
        
        print(f"{prefix}内存状态:")
        print(f"  CPU内存: {cpu_mem:.2f} GB")
        if torch.cuda.is_available():
            print(f"  GPU分配: {gpu_alloc:.2f} GB")
            print(f"  GPU保留: {gpu_reserved:.2f} GB")
    
    def cleanup_memory(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def save_checkpoint(self, batch_idx: int, batch_results: Dict[str, Any], 
                       accumulated_data: Dict[str, Any]):
        """保存检查点"""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_batch_{batch_idx:04d}.json"
        
        checkpoint_data = {
            'batch_idx': batch_idx,
            'timestamp': datetime.datetime.now().isoformat(),
            'batch_results': batch_results,
            'accumulated_summary': {
                'total_batches_processed': batch_idx + 1,
                'total_samples_processed': accumulated_data['total_samples'],
                'total_knowledge_pieces': accumulated_data['total_pieces'],
                'memory_usage_gb': self.get_memory_usage(),
                'gpu_memory_gb': self.get_gpu_memory_usage()
            }
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 检查点已保存: {checkpoint_file}")
        return checkpoint_file
    
    def load_latest_checkpoint(self) -> Optional[Tuple[int, Dict[str, Any]]]:
        """加载最新的检查点"""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_batch_*.json"))
        
        if not checkpoint_files:
            return None
        
        # 找到最新的检查点
        latest_file = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        batch_idx = checkpoint_data['batch_idx']
        accumulated_data = checkpoint_data.get('accumulated_summary', {})
        
        print(f"📂 加载检查点: {latest_file}")
        print(f"   上次处理到批次: {batch_idx}")
        print(f"   已处理样本: {accumulated_data.get('total_samples_processed', 0)}")
        
        return batch_idx, accumulated_data
    
    def process_single_sub_batch(self, sub_batch_data: List[Dict], 
                                sub_batch_idx: int, batch_idx: int, 
                                analyzer: MultiModelKnowledgeCouplingMVP) -> Dict[str, Any]:
        """处理单个子批次"""
        print(f"  📦 子批次 {sub_batch_idx + 1}: {len(sub_batch_data)} 个样本")
        
        # 提取知识片段
        knowledge_pieces = analyzer.extract_knowledge_pieces_from_hotpot(
            sub_batch_data, len(sub_batch_data)
        )
        
        if not knowledge_pieces:
            print(f"  ⚠️ 子批次 {sub_batch_idx + 1} 未提取到知识片段")
            return {
                'sub_batch_idx': sub_batch_idx,
                'samples_count': len(sub_batch_data),
                'knowledge_pieces_count': 0,
                'knowledge_pieces': [],
                'coupling_pairs': []
            }
        
        # 计算梯度
        gradients = analyzer.compute_all_gradients()
        
        if not gradients:
            print(f"  ⚠️ 子批次 {sub_batch_idx + 1} 未计算到梯度")
            return {
                'sub_batch_idx': sub_batch_idx,
                'samples_count': len(sub_batch_data),
                'knowledge_pieces_count': len(knowledge_pieces),
                'knowledge_pieces': [],
                'coupling_pairs': []
            }
        
        # 计算耦合矩阵
        coupling_matrix = analyzer.compute_coupling_matrix()
        
        # 提取耦合对数据
        coupling_pairs = []
        piece_ids = [p.piece_id for p in knowledge_pieces]
        
        # 转换为numpy进行处理
        if isinstance(coupling_matrix, torch.Tensor):
            coupling_np = coupling_matrix.detach().cpu().numpy()
        else:
            coupling_np = coupling_matrix
        
        # 提取所有耦合对
        for i in range(len(piece_ids)):
            for j in range(i + 1, len(piece_ids)):
                coupling_strength = float(coupling_np[i, j])
                coupling_pairs.append({
                    'piece_1_id': piece_ids[i],
                    'piece_2_id': piece_ids[j],
                    'coupling_strength': coupling_strength,
                    'piece_1_answer': knowledge_pieces[i].answer,
                    'piece_2_answer': knowledge_pieces[j].answer,
                    'is_same_hotpot': piece_ids[i].split('_hop_')[0] == piece_ids[j].split('_hop_')[0]
                })
        
        # 转换知识片段为可序列化格式
        knowledge_pieces_data = []
        for piece in knowledge_pieces:
            knowledge_pieces_data.append({
                'piece_id': piece.piece_id,
                'question': piece.question,
                'answer': piece.answer,
                'supporting_fact': piece.supporting_fact,
                'category': piece.category
            })
        
        # 清理GPU内存
        del gradients, coupling_matrix
        analyzer.gradients = {}
        analyzer.coupling_matrix = None
        analyzer.knowledge_pieces = []
        self.cleanup_memory()
        
        sub_batch_result = {
            'sub_batch_idx': sub_batch_idx,
            'samples_count': len(sub_batch_data),
            'knowledge_pieces_count': len(knowledge_pieces),
            'coupling_pairs_count': len(coupling_pairs),
            'high_coupling_pairs_count': sum(1 for p in coupling_pairs if p['coupling_strength'] >= 0.4),
            'knowledge_pieces': knowledge_pieces_data,
            'coupling_pairs': coupling_pairs
        }
        
        print(f"  ✅ 子批次 {sub_batch_idx + 1} 完成: "
              f"{len(knowledge_pieces)} 片段, "
              f"{len(coupling_pairs)} 耦合对")
        
        return sub_batch_result
    
    def process_single_batch(self, batch_data: List[Dict], batch_idx: int) -> Dict[str, Any]:
        """处理单个批次 - 使用子批次策略"""
        print(f"\n🔄 处理批次 {batch_idx + 1}")
        print(f"   样本数量: {len(batch_data)}")
        print(f"   子批次大小: {self.sub_batch_size}")
        
        self.print_memory_status("   开始前 ")
        
        # 创建输出目录
        output_dir = self.output_dir / f"batch_{batch_idx:04d}"
        output_dir.mkdir(exist_ok=True)
        
        # 创建分析器 (整个批次共享一个分析器)
        analyzer = MultiModelKnowledgeCouplingMVP(
            model_path=self.model_path,
            layer_range=self.layer_range
        )
        
        try:
            # 将批次数据分割为子批次
            sub_batches = []
            for i in range(0, len(batch_data), self.sub_batch_size):
                sub_batch = batch_data[i:i + self.sub_batch_size]
                sub_batches.append(sub_batch)
            
            print(f"   分割为 {len(sub_batches)} 个子批次")
            
            # 处理每个子批次
            all_knowledge_pieces = []
            all_coupling_pairs = []
            total_sub_batch_pairs = 0
            total_high_coupling = 0
            
            for sub_batch_idx, sub_batch_data in enumerate(sub_batches):
                sub_batch_result = self.process_single_sub_batch(
                    sub_batch_data, sub_batch_idx, batch_idx, analyzer
                )
                
                # 累积结果
                all_knowledge_pieces.extend(sub_batch_result['knowledge_pieces'])
                all_coupling_pairs.extend(sub_batch_result['coupling_pairs'])
                total_sub_batch_pairs += sub_batch_result['coupling_pairs_count']
                total_high_coupling += sub_batch_result['high_coupling_pairs_count']
            
            # 保存批次结果
            # 1. 保存知识片段
            knowledge_pieces_file = output_dir / "knowledge_pieces.json"
            with open(knowledge_pieces_file, 'w', encoding='utf-8') as f:
                json.dump(all_knowledge_pieces, f, indent=2, ensure_ascii=False)
            
            # 2. 保存耦合对 (CSV格式)
            coupling_csv_file = output_dir / "coupling_pairs.csv"
            if all_coupling_pairs:
                import pandas as pd
                coupling_df = pd.DataFrame(all_coupling_pairs)
                coupling_df.to_csv(coupling_csv_file, index=False)
            
            # 3. 保存高耦合对
            high_coupling_pairs = [p for p in all_coupling_pairs if p['coupling_strength'] >= 0.4]
            high_coupling_pairs.sort(key=lambda x: x['coupling_strength'], reverse=True)
            
            high_coupling_file = output_dir / "high_coupling_pairs.json"
            with open(high_coupling_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'threshold': 0.4,
                    'count': len(high_coupling_pairs),
                    'pairs': high_coupling_pairs
                }, f, indent=2, ensure_ascii=False)
            
            # 4. 计算统计信息
            coupling_strengths = [p['coupling_strength'] for p in all_coupling_pairs]
            
            batch_summary = {
                'batch_idx': batch_idx,
                'samples_count': len(batch_data),
                'sub_batches_count': len(sub_batches),
                'knowledge_pieces_count': len(all_knowledge_pieces),
                'coupling_pairs_count': len(all_coupling_pairs),
                'high_coupling_pairs_count': len(high_coupling_pairs),
                'mean_coupling': float(np.mean(coupling_strengths)) if coupling_strengths else 0.0,
                'std_coupling': float(np.std(coupling_strengths)) if coupling_strengths else 0.0,
                'min_coupling': float(np.min(coupling_strengths)) if coupling_strengths else 0.0,
                'max_coupling': float(np.max(coupling_strengths)) if coupling_strengths else 0.0,
                'processing_time': time.time(),
                'output_directory': str(output_dir)
            }
            
            # 5. 保存批次元数据
            metadata_file = output_dir / "batch_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'batch_summary': batch_summary,
                    'sub_batch_size': self.sub_batch_size,
                    'model_info': {
                        'model_path': self.model_path,
                        'layer_range': self.layer_range
                    },
                    'timestamp': datetime.datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            
            self.print_memory_status("   完成后 ")
            
            print(f"   ✅ 批次完成: {len(all_knowledge_pieces)} 片段, "
                  f"{len(all_coupling_pairs)} 耦合对, "
                  f"{len(high_coupling_pairs)} 高耦合对")
            
            return batch_summary
            
        finally:
            # 清理内存
            del analyzer
            self.cleanup_memory()
    
    def merge_batch_results(self, batch_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并所有批次的结果"""
        print(f"\n📊 合并 {len(batch_summaries)} 个批次的结果...")
        
        # 合并所有耦合对数据
        all_coupling_pairs = []
        all_knowledge_pieces = []
        
        total_samples = 0
        total_pieces = 0
        total_pairs = 0
        total_high_coupling = 0
        
        coupling_strengths = []
        
        for summary in tqdm(batch_summaries, desc="合并批次数据"):
            batch_dir = Path(summary['output_directory'])
            
            # 加载耦合对数据
            coupling_file = batch_dir / "coupling_pairs.csv"
            if coupling_file.exists():
                import pandas as pd
                batch_coupling = pd.read_csv(coupling_file)
                
                # 重新编号piece_id以避免冲突
                batch_coupling['piece_1_id'] = batch_coupling['piece_1_id'].apply(
                    lambda x: f"batch_{summary['batch_idx']:04d}_{x}"
                )
                batch_coupling['piece_2_id'] = batch_coupling['piece_2_id'].apply(
                    lambda x: f"batch_{summary['batch_idx']:04d}_{x}"
                )
                
                all_coupling_pairs.append(batch_coupling)
                coupling_strengths.extend(batch_coupling['coupling_strength'].tolist())
            
            # 加载知识片段数据
            pieces_file = batch_dir / "knowledge_pieces.json"
            if pieces_file.exists():
                with open(pieces_file, 'r', encoding='utf-8') as f:
                    batch_pieces = json.load(f)
                
                # 重新编号piece_id
                for piece in batch_pieces:
                    piece['piece_id'] = f"batch_{summary['batch_idx']:04d}_{piece['piece_id']}"
                
                all_knowledge_pieces.extend(batch_pieces)
            
            # 累计统计
            total_samples += summary['samples_count']
            total_pieces += summary['knowledge_pieces_count']
            total_pairs += summary['coupling_pairs_count']
            total_high_coupling += summary['high_coupling_pairs_count']
        
        # 合并数据框
        if all_coupling_pairs:
            import pandas as pd
            merged_coupling_df = pd.concat(all_coupling_pairs, ignore_index=True)
        else:
            merged_coupling_df = pd.DataFrame()
        
        # 计算全局统计
        global_stats = {
            'total_batches': len(batch_summaries),
            'total_samples': total_samples,
            'total_knowledge_pieces': total_pieces,
            'total_coupling_pairs': total_pairs,
            'total_high_coupling_pairs': total_high_coupling,
            'global_mean_coupling': np.mean(coupling_strengths) if coupling_strengths else 0,
            'global_std_coupling': np.std(coupling_strengths) if coupling_strengths else 0,
            'global_min_coupling': np.min(coupling_strengths) if coupling_strengths else 0,
            'global_max_coupling': np.max(coupling_strengths) if coupling_strengths else 0,
            'high_coupling_ratio': total_high_coupling / total_pairs if total_pairs > 0 else 0
        }
        
        # 保存合并结果
        final_output_dir = self.output_dir / "final_merged_results"
        final_output_dir.mkdir(exist_ok=True)
        
        # 保存合并的耦合对数据
        if not merged_coupling_df.empty:
            merged_coupling_df.to_csv(final_output_dir / "all_coupling_pairs.csv", index=False)
            print(f"✅ 保存合并的耦合对数据: {len(merged_coupling_df)} 对")
        
        # 保存合并的知识片段数据
        with open(final_output_dir / "all_knowledge_pieces.json", 'w', encoding='utf-8') as f:
            json.dump(all_knowledge_pieces, f, indent=2, ensure_ascii=False)
        print(f"✅ 保存合并的知识片段数据: {len(all_knowledge_pieces)} 个")
        
        # 保存全局统计
        final_stats = {
            'global_statistics': global_stats,
            'batch_summaries': batch_summaries,
            'processing_metadata': {
                'model_path': self.model_path,
                'layer_range': self.layer_range,
                'batch_size': self.batch_size,
                'sub_batch_size': self.sub_batch_size,
                'total_processing_time': sum(s.get('processing_time', 0) for s in batch_summaries),
                'completion_timestamp': datetime.datetime.now().isoformat()
            }
        }
        
        with open(final_output_dir / "global_analysis_results.json", 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, indent=2, ensure_ascii=False)
        
        print(f"📊 全局统计结果:")
        print(f"   总样本数: {global_stats['total_samples']:,}")
        print(f"   总知识片段: {global_stats['total_knowledge_pieces']:,}")
        print(f"   总耦合对: {global_stats['total_coupling_pairs']:,}")
        print(f"   高耦合对: {global_stats['total_high_coupling_pairs']:,} ({global_stats['high_coupling_ratio']:.2%})")
        print(f"   平均耦合强度: {global_stats['global_mean_coupling']:.4f}")
        print(f"   耦合强度范围: [{global_stats['global_min_coupling']:.4f}, {global_stats['global_max_coupling']:.4f}]")
        
        return final_stats
    
    def run_full_processing(self, resume_from_checkpoint: bool = True) -> Dict[str, Any]:
        """运行完整的批处理分析"""
        print(f"\n🚀 开始批处理知识耦合分析 (优化版)")
        print("=" * 80)
        
        # 加载数据
        print(f"📚 加载HotpotQA数据...")
        hotpot_data = load_hotpot_data("datasets/processed/hotpotqa_all_converted.json")
        total_samples = len(hotpot_data)
        print(f"✅ 加载完成: {total_samples:,} 个样本")
        
        # 检查是否从检查点恢复
        start_batch_idx = 0
        batch_summaries = []
        
        if resume_from_checkpoint:
            checkpoint_info = self.load_latest_checkpoint()
            if checkpoint_info:
                start_batch_idx, accumulated_data = checkpoint_info
                start_batch_idx += 1  # 从下一个批次开始
                
                # 加载之前的批次摘要
                for i in range(start_batch_idx):
                    checkpoint_file = self.checkpoint_dir / f"checkpoint_batch_{i:04d}.json"
                    if checkpoint_file.exists():
                        with open(checkpoint_file, 'r', encoding='utf-8') as f:
                            checkpoint_data = json.load(f)
                            if 'batch_results' in checkpoint_data:
                                batch_summaries.append(checkpoint_data['batch_results'])
        
        # 计算批次数量
        total_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        print(f"\n📋 处理计划:")
        print(f"   总样本数: {total_samples:,}")
        print(f"   批处理大小: {self.batch_size:,}")
        print(f"   子批次大小: {self.sub_batch_size}")
        print(f"   总批次数: {total_batches}")
        print(f"   开始批次: {start_batch_idx}")
        print(f"   剩余批次: {total_batches - start_batch_idx}")
        print(f"   预计每批次子批次数: {(self.batch_size + self.sub_batch_size - 1) // self.sub_batch_size}")
        
        # 处理每个批次
        accumulated_data = {
            'total_samples': len(batch_summaries) * self.batch_size,
            'total_pieces': sum(s.get('knowledge_pieces_count', 0) for s in batch_summaries)
        }
        
        for batch_idx in range(start_batch_idx, total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_samples)
            batch_data = hotpot_data[start_idx:end_idx]
            
            print(f"\n{'='*60}")
            print(f"批次 {batch_idx + 1}/{total_batches}")
            print(f"样本范围: {start_idx:,} - {end_idx:,}")
            print(f"进度: {(batch_idx + 1)/total_batches:.1%}")
            
            try:
                # 处理批次
                batch_result = self.process_single_batch(batch_data, batch_idx)
                batch_summaries.append(batch_result)
                
                # 更新累计数据
                accumulated_data['total_samples'] += batch_result['samples_count']
                accumulated_data['total_pieces'] += batch_result['knowledge_pieces_count']
                
                # 保存检查点
                self.save_checkpoint(batch_idx, batch_result, accumulated_data)
                
            except Exception as e:
                print(f"❌ 批次 {batch_idx} 处理失败: {e}")
                # 保存错误信息
                error_info = {
                    'batch_idx': batch_idx,
                    'error': str(e),
                    'timestamp': datetime.datetime.now().isoformat()
                }
                with open(self.checkpoint_dir / f"error_batch_{batch_idx:04d}.json", 'w') as f:
                    json.dump(error_info, f, indent=2)
                raise
        
        # 合并所有结果
        print(f"\n🔄 合并所有批次结果...")
        final_results = self.merge_batch_results(batch_summaries)
        
        print(f"\n🎉 全部处理完成!")
        print(f"📁 最终结果保存在: {self.output_dir / 'final_merged_results'}")
        
        return final_results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="知识耦合批处理器 - 处理全部HotpotQA (优化版)")
    parser.add_argument("--model_path", default="meta-llama/Llama-2-7b-hf", help="模型路径")
    parser.add_argument("--batch_size", type=int, default=2000, help="批处理大小")
    parser.add_argument("--sub_batch_size", type=int, default=10, help="子批次大小 (减少显存使用)")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="检查点目录")
    parser.add_argument("--output_dir", default="results/full_hotpotqa_analysis", help="输出目录")
    parser.add_argument("--no_resume", action="store_true", help="不从检查点恢复")
    parser.add_argument("--layer_start", type=int, help="起始层")
    parser.add_argument("--layer_end", type=int, help="结束层")
    
    args = parser.parse_args()
    
    # 设置层范围
    layer_range = None
    if args.layer_start is not None and args.layer_end is not None:
        layer_range = (args.layer_start, args.layer_end)
        print(f"🎯 使用自定义层范围: {args.layer_start}-{args.layer_end}")
    elif 'llama' in args.model_path.lower():
        layer_range = (28, 31)  # LLaMA高层语义层
        print(f"🎯 LLaMA自动选择层范围: 28-31")
    
    # 创建处理器
    processor = KnowledgeCouplingBatchProcessor(
        model_path=args.model_path,
        batch_size=args.batch_size,
        sub_batch_size=args.sub_batch_size,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        layer_range=layer_range
    )
    
    # 运行处理
    results = processor.run_full_processing(resume_from_checkpoint=not args.no_resume)
    
    print(f"\n✅ 批处理完成!")


if __name__ == "__main__":
    main() 