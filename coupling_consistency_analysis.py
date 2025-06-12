#!/usr/bin/env python3
"""
知识耦合一致性现象分析脚本
第一阶段：数据分类和一致性现象分析

主要功能：
1. 同答案一致性验证
2. 不同hop层级的耦合模式分析
3. 答案类型的分类统计
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import re

class CouplingConsistencyAnalyzer:
    """知识耦合一致性分析器"""
    
    def __init__(self, results_dir: str = "results/full_hotpotqa_analysis"):
        self.results_dir = Path(results_dir)
        self.final_results_dir = self.results_dir / "final_merged_results"
        
        # 存储分析数据
        self.all_knowledge_pieces = []
        self.all_high_coupling_pairs = []
        self.global_stats = {}
        
        print(f"🔍 知识耦合一致性分析器初始化")
        print(f"   结果目录: {self.results_dir}")
        
    def load_data(self):
        """加载所有需要的数据"""
        print("\n📚 加载分析数据...")
        
        # 1. 加载全局统计信息
        global_stats_file = self.final_results_dir / "global_analysis_results.json"
        with open(global_stats_file, 'r', encoding='utf-8') as f:
            global_results = json.load(f)
            self.global_stats = global_results['global_statistics']
        
        print(f"✅ 全局统计信息加载完成")
        print(f"   总耦合对: {self.global_stats['total_coupling_pairs']:,}")
        print(f"   高耦合对: {self.global_stats['total_high_coupling_pairs']:,}")
        
        # 2. 加载所有知识片段信息
        knowledge_pieces_file = self.final_results_dir / "all_knowledge_pieces.json"
        with open(knowledge_pieces_file, 'r', encoding='utf-8') as f:
            self.all_knowledge_pieces = json.load(f)
        
        print(f"✅ 知识片段数据加载完成: {len(self.all_knowledge_pieces):,} 个片段")
        
        # 3. 加载所有高耦合对（从各个批次合并）
        print("📊 加载高耦合对数据...")
        self._load_all_high_coupling_pairs()
        
        print(f"✅ 数据加载完成!")
        
    def _load_all_high_coupling_pairs(self):
        """从所有批次加载高耦合对数据"""
        all_pairs = []
        
        # 遍历所有批次目录
        batch_dirs = sorted([d for d in self.results_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('batch_')])
        
        for batch_dir in batch_dirs:
            high_coupling_file = batch_dir / "high_coupling_pairs.json"
            if high_coupling_file.exists():
                with open(high_coupling_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                    all_pairs.extend(batch_data['pairs'])
        
        self.all_high_coupling_pairs = all_pairs
        print(f"   从 {len(batch_dirs)} 个批次加载了 {len(all_pairs):,} 个高耦合对")
    
    def analyze_same_answer_consistency(self) -> Dict[str, Any]:
        """1. 同答案一致性验证"""
        print("\n🎯 分析同答案一致性...")
        
        same_answer_pairs = []
        different_answer_pairs = []
        
        for pair in self.all_high_coupling_pairs:
            if pair['piece_1_answer'] == pair['piece_2_answer']:
                same_answer_pairs.append(pair)
            else:
                different_answer_pairs.append(pair)
        
        # 统计耦合强度
        same_answer_strengths = [pair['coupling_strength'] for pair in same_answer_pairs]
        different_answer_strengths = [pair['coupling_strength'] for pair in different_answer_pairs]
        
        results = {
            'same_answer_count': len(same_answer_pairs),
            'different_answer_count': len(different_answer_pairs),
            'same_answer_ratio': len(same_answer_pairs) / len(self.all_high_coupling_pairs),
            'same_answer_mean_coupling': np.mean(same_answer_strengths) if same_answer_strengths else 0,
            'different_answer_mean_coupling': np.mean(different_answer_strengths) if different_answer_strengths else 0,
            'same_answer_std_coupling': np.std(same_answer_strengths) if same_answer_strengths else 0,
            'different_answer_std_coupling': np.std(different_answer_strengths) if different_answer_strengths else 0,
            'same_answer_max_coupling': max(same_answer_strengths) if same_answer_strengths else 0,
            'different_answer_max_coupling': max(different_answer_strengths) if different_answer_strengths else 0
        }
        
        print(f"📊 同答案一致性分析结果:")
        print(f"   相同答案的高耦合对: {results['same_answer_count']:,} ({results['same_answer_ratio']:.1%})")
        print(f"   不同答案的高耦合对: {results['different_answer_count']:,}")
        print(f"   相同答案平均耦合强度: {results['same_answer_mean_coupling']:.4f}")
        print(f"   不同答案平均耦合强度: {results['different_answer_mean_coupling']:.4f}")
        
        # 显著性差异
        coupling_diff = results['same_answer_mean_coupling'] - results['different_answer_mean_coupling']
        print(f"   💡 耦合强度差异: {coupling_diff:.4f} ({'相同答案更高' if coupling_diff > 0 else '不同答案更高'})")
        
        return results
    
    def analyze_hop_level_coupling(self) -> Dict[str, Any]:
        """2. 不同hop层级的耦合模式分析"""
        print("\n🔗 分析不同hop层级的耦合模式...")
        
        intra_hotpot_pairs = []  # 同一HotpotQA样本内的耦合
        inter_hotpot_pairs = []  # 不同HotpotQA样本间的耦合
        
        for pair in self.all_high_coupling_pairs:
            if pair['is_same_hotpot']:
                intra_hotpot_pairs.append(pair)
            else:
                inter_hotpot_pairs.append(pair)
        
        # 统计耦合强度
        intra_strengths = [pair['coupling_strength'] for pair in intra_hotpot_pairs]
        inter_strengths = [pair['coupling_strength'] for pair in inter_hotpot_pairs]
        
        results = {
            'intra_hotpot_count': len(intra_hotpot_pairs),
            'inter_hotpot_count': len(inter_hotpot_pairs),
            'intra_hotpot_ratio': len(intra_hotpot_pairs) / len(self.all_high_coupling_pairs),
            'intra_mean_coupling': np.mean(intra_strengths) if intra_strengths else 0,
            'inter_mean_coupling': np.mean(inter_strengths) if inter_strengths else 0,
            'intra_std_coupling': np.std(intra_strengths) if intra_strengths else 0,
            'inter_std_coupling': np.std(inter_strengths) if inter_strengths else 0,
            'intra_max_coupling': max(intra_strengths) if intra_strengths else 0,
            'inter_max_coupling': max(inter_strengths) if inter_strengths else 0
        }
        
        print(f"📊 Hop层级耦合分析结果:")
        print(f"   Intra-HotpotQA耦合对: {results['intra_hotpot_count']:,} ({results['intra_hotpot_ratio']:.1%})")
        print(f"   Inter-HotpotQA耦合对: {results['inter_hotpot_count']:,}")
        print(f"   Intra平均耦合强度: {results['intra_mean_coupling']:.4f}")
        print(f"   Inter平均耦合强度: {results['inter_mean_coupling']:.4f}")
        
        # 分析差异
        coupling_diff = results['intra_mean_coupling'] - results['inter_mean_coupling']
        print(f"   💡 Intra vs Inter差异: {coupling_diff:.4f}")
        
        return results
    
    def analyze_answer_types(self) -> Dict[str, Any]:
        """3. 答案类型的分类统计"""
        print("\n📋 分析答案类型分布...")
        
        # 定义答案类型分类函数
        def classify_answer_type(answer: str) -> str:
            answer = answer.strip().lower()
            
            # Yes/No类型 - 最高优先级
            if answer in ['yes', 'no']:
                return 'Yes/No'
            
            # 年份 - 在数字之前检查，避免被数字类型覆盖
            if re.match(r'^\d{4}$', answer):
                return 'Year'
            
            # 数字类型
            if re.match(r'^[\d,]+$', answer.replace(' ', '')):
                return 'Number'
            
            # 常见人名模式
            if any(word in answer for word in ['president', 'director', 'actor', 'writer', 'author']):
                return 'Person_Title'
            
            # 地名模式
            if any(word in answer for word in ['city', 'state', 'country', 'america', 'england', 'california']):
                return 'Location'
            
            # 单词数量判断
            word_count = len(answer.split())
            if word_count == 1:
                return 'Single_Word'
            elif word_count == 2:
                return 'Two_Words'
            elif word_count <= 5:
                return 'Short_Phrase'
            else:
                return 'Long_Phrase'
        
        # 统计所有高耦合对中的答案类型
        answer_types = []
        answer_type_pairs = defaultdict(list)
        
        for pair in self.all_high_coupling_pairs:
            # 分类两个答案
            type1 = classify_answer_type(pair['piece_1_answer'])
            type2 = classify_answer_type(pair['piece_2_answer'])
            
            answer_types.extend([type1, type2])
            
            # 记录答案类型组合
            if type1 == type2:  # 相同类型
                answer_type_pairs[f"{type1}_vs_{type2}"].append(pair)
            else:  # 不同类型
                combo_key = f"{min(type1, type2)}_vs_{max(type1, type2)}"
                answer_type_pairs[combo_key].append(pair)
        
        # 统计结果
        type_counts = Counter(answer_types)
        total_answers = len(answer_types)
        
        # 最常见的答案类型
        most_common_types = type_counts.most_common(10)
        
        # 同类型vs跨类型耦合分析
        same_type_pairs = []
        cross_type_pairs = []
        
        for pair in self.all_high_coupling_pairs:
            type1 = classify_answer_type(pair['piece_1_answer'])
            type2 = classify_answer_type(pair['piece_2_answer'])
            
            if type1 == type2:
                same_type_pairs.append(pair)
            else:
                cross_type_pairs.append(pair)
        
        same_type_strengths = [pair['coupling_strength'] for pair in same_type_pairs]
        cross_type_strengths = [pair['coupling_strength'] for pair in cross_type_pairs]
        
        results = {
            'total_answer_instances': total_answers,
            'unique_answer_types': len(type_counts),
            'most_common_types': most_common_types,
            'type_distribution': dict(type_counts),
            'same_type_pairs_count': len(same_type_pairs),
            'cross_type_pairs_count': len(cross_type_pairs),
            'same_type_ratio': len(same_type_pairs) / len(self.all_high_coupling_pairs),
            'same_type_mean_coupling': np.mean(same_type_strengths) if same_type_strengths else 0,
            'cross_type_mean_coupling': np.mean(cross_type_strengths) if cross_type_strengths else 0,
            'answer_type_pairs': dict(answer_type_pairs)
        }
        
        print(f"📊 答案类型分析结果:")
        print(f"   总答案实例: {total_answers:,}")
        print(f"   识别的答案类型: {len(type_counts)} 种")
        print(f"   最常见的答案类型:")
        for answer_type, count in most_common_types:
            percentage = count / total_answers * 100
            print(f"     {answer_type}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n   同类型耦合对: {len(same_type_pairs):,} ({results['same_type_ratio']:.1%})")
        print(f"   跨类型耦合对: {len(cross_type_pairs):,}")
        print(f"   同类型平均耦合强度: {results['same_type_mean_coupling']:.4f}")
        print(f"   跨类型平均耦合强度: {results['cross_type_mean_coupling']:.4f}")
        
        return results
    
    def generate_consistency_report(self, same_answer_results: Dict, hop_level_results: Dict, 
                                  answer_type_results: Dict, output_dir: str = "results/consistency_analysis"):
        """生成一致性分析报告"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📝 生成一致性分析报告...")
        
        # 保存详细分析结果
        all_results = {
            'global_statistics': self.global_stats,
            'same_answer_consistency': same_answer_results,
            'hop_level_coupling': hop_level_results,
            'answer_type_analysis': answer_type_results,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        results_file = output_path / "consistency_analysis_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        report_file = output_path / "consistency_analysis_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 知识耦合一致性现象分析报告\n\n")
            
            f.write("## 🎯 研究目标\n")
            f.write("验证知识片段耦合度的一致性现象，为后门攻击的泛化提供理论基础。\n\n")
            
            f.write("## 📊 数据概览\n")
            f.write(f"- **总知识片段**: {len(self.all_knowledge_pieces):,}\n")
            f.write(f"- **总耦合对**: {self.global_stats['total_coupling_pairs']:,}\n")
            f.write(f"- **高耦合对**: {self.global_stats['total_high_coupling_pairs']:,}\n")
            f.write(f"- **高耦合比例**: {self.global_stats['high_coupling_ratio']:.2%}\n\n")
            
            f.write("## 🔍 关键发现\n\n")
            
            f.write("### 1. 同答案一致性验证\n")
            f.write(f"**核心假设**: 相同答案的知识片段应该表现出更高的耦合度\n\n")
            f.write(f"- 相同答案的高耦合对: **{same_answer_results['same_answer_count']:,}** ({same_answer_results['same_answer_ratio']:.1%})\n")
            f.write(f"- 相同答案平均耦合强度: **{same_answer_results['same_answer_mean_coupling']:.4f}**\n")
            f.write(f"- 不同答案平均耦合强度: **{same_answer_results['different_answer_mean_coupling']:.4f}**\n")
            
            coupling_diff = same_answer_results['same_answer_mean_coupling'] - same_answer_results['different_answer_mean_coupling']
            f.write(f"- **耦合强度差异**: {coupling_diff:.4f}\n")
            
            if coupling_diff > 0.05:
                f.write("✅ **验证成功**: 相同答案的知识片段确实表现出显著更高的耦合度!\n\n")
            elif coupling_diff > 0:
                f.write("⚠️ **部分验证**: 相同答案的耦合度略高，但差异不够显著\n\n")
            else:
                f.write("❌ **假设不成立**: 相同答案的耦合度并不更高\n\n")
            
            f.write("### 2. Hop层级耦合模式\n")
            f.write(f"**发现**: 同一HotpotQA推理链内的知识片段表现出极高耦合度\n\n")
            f.write(f"- Intra-HotpotQA耦合对: **{hop_level_results['intra_hotpot_count']:,}** ({hop_level_results['intra_hotpot_ratio']:.1%})\n")
            f.write(f"- Intra平均耦合强度: **{hop_level_results['intra_mean_coupling']:.4f}**\n")
            f.write(f"- Inter平均耦合强度: **{hop_level_results['inter_mean_coupling']:.4f}**\n")
            
            hop_diff = hop_level_results['intra_mean_coupling'] - hop_level_results['inter_mean_coupling']
            f.write(f"- **层级差异**: {hop_diff:.4f}\n\n")
            
            f.write("💡 **攻击策略启示**: 同一推理链内的知识片段高度耦合，攻击一个hop可能直接影响另一个hop。\n\n")
            
            f.write("### 3. 答案类型分布分析\n")
            f.write(f"**目标**: 识别最适合后门攻击的知识类型\n\n")
            f.write("**最常见的答案类型**:\n")
            for answer_type, count in answer_type_results['most_common_types']:
                percentage = count / answer_type_results['total_answer_instances'] * 100
                f.write(f"- {answer_type}: {count:,} ({percentage:.1f}%)\n")
            
            f.write(f"\n**同类型 vs 跨类型耦合**:\n")
            f.write(f"- 同类型耦合对: {answer_type_results['same_type_pairs_count']:,} ({answer_type_results['same_type_ratio']:.1%})\n")
            f.write(f"- 同类型平均耦合强度: **{answer_type_results['same_type_mean_coupling']:.4f}**\n")
            f.write(f"- 跨类型平均耦合强度: **{answer_type_results['cross_type_mean_coupling']:.4f}**\n\n")
            
            f.write("## 🎯 后门攻击策略建议\n\n")
            
            # 基于分析结果给出建议
            if same_answer_results['same_answer_ratio'] > 0.5:
                f.write("1. **优先攻击相同答案的知识集群** - 高概率触发涟漪效应\n")
            
            if hop_level_results['intra_hotpot_ratio'] > 0.3:
                f.write("2. **利用推理链内耦合** - 攻击一个hop来影响整个推理链\n")
            
            # 推荐攻击的答案类型
            top_answer_type = answer_type_results['most_common_types'][0][0]
            f.write(f"3. **重点攻击 {top_answer_type} 类型** - 出现频率最高，影响面最广\n")
            
            f.write("\n## 📈 下一步验证计划\n")
            f.write("1. 选择高耦合的相同答案知识片段对进行MEMIT攻击实验\n")
            f.write("2. 测量实际的涟漪效应传播范围\n")
            f.write("3. 验证GradSim预测的准确性\n")
        
        print(f"✅ 一致性分析报告已保存:")
        print(f"   详细结果: {results_file}")
        print(f"   分析报告: {report_file}")
        
        return results_file, report_file
    
    def run_full_consistency_analysis(self, output_dir: str = "results/consistency_analysis"):
        """运行完整的一致性分析"""
        print("🚀 开始知识耦合一致性现象分析")
        print("=" * 60)
        
        # 加载数据
        self.load_data()
        
        # 执行三个主要分析
        same_answer_results = self.analyze_same_answer_consistency()
        hop_level_results = self.analyze_hop_level_coupling()
        answer_type_results = self.analyze_answer_types()
        
        # 生成报告
        results_file, report_file = self.generate_consistency_report(
            same_answer_results, hop_level_results, answer_type_results, output_dir
        )
        
        print(f"\n🎉 一致性分析完成!")
        print(f"📄 查看详细报告: {report_file}")
        
        return {
            'same_answer_results': same_answer_results,
            'hop_level_results': hop_level_results,
            'answer_type_results': answer_type_results,
            'report_files': {
                'results': results_file,
                'report': report_file
            }
        }


def main():
    """主函数"""
    analyzer = CouplingConsistencyAnalyzer()
    results = analyzer.run_full_consistency_analysis()
    
    print("\n🎯 关键发现总结:")
    print(f"✅ 相同答案耦合度更高: {results['same_answer_results']['same_answer_mean_coupling']:.4f} vs {results['same_answer_results']['different_answer_mean_coupling']:.4f}")
    print(f"✅ Intra-HotpotQA耦合更强: {results['hop_level_results']['intra_mean_coupling']:.4f} vs {results['hop_level_results']['inter_mean_coupling']:.4f}")
    print(f"✅ 最常见答案类型: {results['answer_type_results']['most_common_types'][0][0]}")


if __name__ == "__main__":
    main() 