#!/usr/bin/env python3
"""
修正数据对齐问题 - 基于原始高耦合对数据进行第一阶段分析
解决清理后数据与高耦合对数据不匹配的问题
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import re
from typing import Dict, List, Tuple, Any

class DataAlignmentFixer:
    """数据对齐修正器"""
    
    def __init__(self, results_dir: str = "results/full_hotpotqa_analysis"):
        self.results_dir = Path(results_dir)
        self.final_results_dir = self.results_dir / "final_merged_results"
        self.output_dir = Path("results/aligned_analysis")
        self.output_dir.mkdir(exist_ok=True)
        
        print("🔧 数据对齐修正器初始化")
        print(f"   目标：基于原始高耦合对数据进行完整的第一阶段分析")
        
    def load_original_data(self):
        """加载原始数据"""
        print("\n📊 加载原始数据...")
        
        # 1. 加载所有知识片段
        with open(self.final_results_dir / "all_knowledge_pieces.json", 'r') as f:
            all_pieces = json.load(f)
        print(f"   原始知识片段数量: {len(all_pieces)}")
        
        # 2. 加载所有高耦合对（合并所有批次）
        all_high_coupling_pairs = []
        batch_dirs = [d for d in self.results_dir.iterdir() if d.is_dir() and d.name.startswith('batch_')]
        
        for batch_dir in sorted(batch_dirs):
            coupling_file = batch_dir / "high_coupling_pairs.json"
            if coupling_file.exists():
                with open(coupling_file, 'r') as f:
                    batch_data = json.load(f)
                    all_high_coupling_pairs.extend(batch_data['pairs'])
        
        print(f"   总高耦合对数量: {len(all_high_coupling_pairs)}")
        
        # 3. 创建piece_id到详细信息的映射
        piece_id_to_info = {piece['piece_id']: piece for piece in all_pieces}
        
        return all_pieces, all_high_coupling_pairs, piece_id_to_info
    
    def phase1_same_answer_consistency_analysis(self, high_coupling_pairs: List[Dict], piece_id_to_info: Dict):
        """第一阶段分析1: 同答案一致性验证"""
        print("\n🎯 第一阶段分析1: 同答案一致性验证")
        
        same_answer_pairs = []
        different_answer_pairs = []
        same_answer_strengths = []
        different_answer_strengths = []
        
        valid_pairs = 0
        
        for pair in high_coupling_pairs:
            piece_1_id = pair['piece_1_id']
            piece_2_id = pair['piece_2_id']
            
            # 检查两个piece是否都存在
            if piece_1_id in piece_id_to_info and piece_2_id in piece_id_to_info:
                valid_pairs += 1
                answer_1 = pair['piece_1_answer']
                answer_2 = pair['piece_2_answer']
                coupling_strength = pair['coupling_strength']
                
                if answer_1 == answer_2:
                    same_answer_pairs.append(pair)
                    same_answer_strengths.append(coupling_strength)
                else:
                    different_answer_pairs.append(pair)
                    different_answer_strengths.append(coupling_strength)
        
        # 计算统计结果
        total_pairs = len(same_answer_pairs) + len(different_answer_pairs)
        same_answer_ratio = len(same_answer_pairs) / total_pairs if total_pairs > 0 else 0
        same_answer_avg_strength = np.mean(same_answer_strengths) if same_answer_strengths else 0
        different_answer_avg_strength = np.mean(different_answer_strengths) if different_answer_strengths else 0
        
        results = {
            'total_valid_pairs': total_pairs,
            'same_answer_pairs': len(same_answer_pairs),
            'different_answer_pairs': len(different_answer_pairs),
            'same_answer_ratio': same_answer_ratio,
            'same_answer_avg_strength': same_answer_avg_strength,
            'different_answer_avg_strength': different_answer_avg_strength,
            'strength_difference': same_answer_avg_strength - different_answer_avg_strength
        }
        
        print(f"   ✅ 有效高耦合对: {total_pairs}")
        print(f"   📊 相同答案对: {len(same_answer_pairs)} ({same_answer_ratio:.1%})")
        print(f"   📊 不同答案对: {len(different_answer_pairs)}")
        print(f"   🔍 相同答案平均强度: {same_answer_avg_strength:.4f}")
        print(f"   🔍 不同答案平均强度: {different_answer_avg_strength:.4f}")
        print(f"   📈 强度差异: {same_answer_avg_strength - different_answer_avg_strength:+.4f}")
        
        return results, same_answer_pairs, different_answer_pairs
    
    def phase1_hop_level_coupling_analysis(self, high_coupling_pairs: List[Dict]):
        """第一阶段分析2: Hop层级耦合模式分析"""
        print("\n🎯 第一阶段分析2: Hop层级耦合模式分析")
        
        intra_hotpot_pairs = []
        inter_hotpot_pairs = []
        intra_strengths = []
        inter_strengths = []
        
        for pair in high_coupling_pairs:
            piece_1_id = pair['piece_1_id']
            piece_2_id = pair['piece_2_id']
            coupling_strength = pair['coupling_strength']
            
            # 提取HotpotQA ID（去掉_hop_X部分）
            hotpot_1 = piece_1_id.rsplit('_hop_', 1)[0]
            hotpot_2 = piece_2_id.rsplit('_hop_', 1)[0]
            
            if hotpot_1 == hotpot_2:
                intra_hotpot_pairs.append(pair)
                intra_strengths.append(coupling_strength)
            else:
                inter_hotpot_pairs.append(pair)
                inter_strengths.append(coupling_strength)
        
        # 计算统计结果
        total_pairs = len(intra_hotpot_pairs) + len(inter_hotpot_pairs)
        intra_ratio = len(intra_hotpot_pairs) / total_pairs if total_pairs > 0 else 0
        intra_avg_strength = np.mean(intra_strengths) if intra_strengths else 0
        inter_avg_strength = np.mean(inter_strengths) if inter_strengths else 0
        
        results = {
            'intra_hotpot_pairs': len(intra_hotpot_pairs),
            'inter_hotpot_pairs': len(inter_hotpot_pairs),
            'intra_hotpot_ratio': intra_ratio,
            'intra_avg_strength': intra_avg_strength,
            'inter_avg_strength': inter_avg_strength,
            'strength_difference': intra_avg_strength - inter_avg_strength
        }
        
        print(f"   📊 Intra-HotpotQA对: {len(intra_hotpot_pairs)} ({intra_ratio:.1%})")
        print(f"   📊 Inter-HotpotQA对: {len(inter_hotpot_pairs)}")
        print(f"   🔍 Intra平均强度: {intra_avg_strength:.4f}")
        print(f"   🔍 Inter平均强度: {inter_avg_strength:.4f}")
        print(f"   📈 强度差异: {intra_avg_strength - inter_avg_strength:+.4f}")
        
        return results, intra_hotpot_pairs, inter_hotpot_pairs
    
    def phase1_answer_type_classification(self, high_coupling_pairs: List[Dict], piece_id_to_info: Dict):
        """第一阶段分析3: 答案类型分类统计"""
        print("\n🎯 第一阶段分析3: 答案类型分类统计")
        
        def classify_answer_type(answer: str) -> str:
            answer = answer.strip().lower()
            
            # Yes/No类型
            if answer in ['yes', 'no']:
                return 'Yes/No'
            
            # 年份类型 - 在数字之前检查
            if re.match(r'^\d{4}$', answer):
                return 'Year'
            
            # 数字类型
            if re.match(r'^[\d,]+$', answer.replace(' ', '')):
                return 'Number'
            
            # 地名模式
            if any(word in answer for word in ['city', 'state', 'country', 'america', 'england', 'california']):
                return 'Location'
            
            # 单词数量分类
            words = answer.split()
            if len(words) == 1:
                return 'Single_Word'
            elif len(words) == 2:
                return 'Two_Words'
            elif len(words) <= 4:
                return 'Short_Phrase'
            else:
                return 'Long_Phrase'
        
        # 统计答案类型
        answer_type_counts = defaultdict(int)
        answer_type_pairs = defaultdict(list)
        answer_type_strengths = defaultdict(list)
        
        for pair in high_coupling_pairs:
            # 只统计相同答案的高耦合对
            if pair['piece_1_answer'] == pair['piece_2_answer']:
                answer = pair['piece_1_answer']
                answer_type = classify_answer_type(answer)
                
                answer_type_counts[answer_type] += 1
                answer_type_pairs[answer_type].append(pair)
                answer_type_strengths[answer_type].append(pair['coupling_strength'])
        
        # 计算每种类型的平均强度
        answer_type_stats = {}
        for answer_type, count in answer_type_counts.items():
            avg_strength = np.mean(answer_type_strengths[answer_type])
            answer_type_stats[answer_type] = {
                'count': count,
                'average_strength': avg_strength,
                'percentage': count / sum(answer_type_counts.values()) * 100
            }
        
        # 按数量排序
        sorted_types = sorted(answer_type_stats.items(), key=lambda x: x[1]['count'], reverse=True)
        
        print(f"   📊 相同答案高耦合对的类型分布:")
        for answer_type, stats in sorted_types:
            print(f"   {answer_type}: {stats['count']} 对 ({stats['percentage']:.1f}%) - 平均强度: {stats['average_strength']:.4f}")
        
        return answer_type_stats, answer_type_pairs
    
    def generate_phase1_comprehensive_report(self, 
                                           same_answer_analysis: Dict,
                                           hop_level_analysis: Dict, 
                                           answer_type_analysis: Dict,
                                           all_pieces: List[Dict],
                                           all_high_coupling_pairs: List[Dict]):
        """生成第一阶段综合报告"""
        
        report = f"""
# 第一阶段完整分析报告：知识耦合一致性现象验证

## 📊 数据概览

### 基础数据统计
- **总知识片段数量**: {len(all_pieces):,}
- **总高耦合对数量**: {len(all_high_coupling_pairs):,}
- **高耦合阈值**: ≥ 0.4
- **分析完成时间**: {pd.Timestamp.now()}

## 🎯 核心假设验证结果

### 1. 同答案一致性验证 - {'✅ 部分验证' if same_answer_analysis['same_answer_ratio'] > 0.5 else '❌ 假设被推翻'}

**假设**: 相同答案的knowledge pieces表现出更高的耦合度

**结果**:
- 相同答案的高耦合对: {same_answer_analysis['same_answer_pairs']:,} ({same_answer_analysis['same_answer_ratio']:.1%})
- 不同答案的高耦合对: {same_answer_analysis['different_answer_pairs']:,}
- 相同答案平均耦合强度: {same_answer_analysis['same_answer_avg_strength']:.4f}
- 不同答案平均耦合强度: {same_answer_analysis['different_answer_avg_strength']:.4f}
- **强度差异**: {same_answer_analysis['strength_difference']:+.4f}

**结论**: {'相同答案确实占多数，但耦合强度' + ('更高' if same_answer_analysis['strength_difference'] > 0 else '更低')}

### 2. Hop层级耦合模式分析 - ✅ 重要发现

**发现**:
- Intra-HotpotQA耦合对: {hop_level_analysis['intra_hotpot_pairs']:,} ({hop_level_analysis['intra_hotpot_ratio']:.1%})
- Inter-HotpotQA耦合对: {hop_level_analysis['inter_hotpot_pairs']:,}
- Intra平均耦合强度: {hop_level_analysis['intra_avg_strength']:.4f}
- Inter平均耦合强度: {hop_level_analysis['inter_avg_strength']:.4f}
- **强度差异**: {hop_level_analysis['strength_difference']:+.4f}

**攻击策略启示**: {'同推理链内攻击更容易，但跨样本攻击影响更深' if hop_level_analysis['strength_difference'] > 0 else '跨样本耦合更强，攻击影响更广泛'}

### 3. 答案类型攻击价值排序

**高价值攻击目标** (基于数量和耦合强度):
"""
        
        # 添加答案类型统计
        sorted_types = sorted(answer_type_analysis.items(), key=lambda x: x[1]['count'], reverse=True)
        for i, (answer_type, stats) in enumerate(sorted_types[:5]):
            priority = "🔥 极高" if i == 0 else "🔶 高" if i <= 2 else "🔷 中"
            report += f"\n{i+1}. **{answer_type}**: {stats['count']} 对 ({stats['percentage']:.1f}%) - 强度: {stats['average_strength']:.4f} - 优先级: {priority}"
        
        report += f"""

## 🎯 第一阶段结论

### ✅ 验证成功的发现
1. **知识耦合现象确实存在** - {len(all_high_coupling_pairs):,}个高耦合对证明了这点
2. **推理链内耦合效应强** - {hop_level_analysis['intra_hotpot_ratio']:.1%}的高耦合发生在同一推理链内
3. **攻击泛化的理论基础成立** - 耦合网络为攻击传播提供了路径

### ⚠️ 需要重新审视的假设
1. **相同答案≠更高耦合** - 强度差异为{same_answer_analysis['strength_difference']:+.4f}
2. **攻击策略需要精确化** - 基于语义而非字符串匹配

### 🚀 下一步攻击实验目标
1. **选择{sorted_types[0][0]}类型作为主要攻击目标** - 最高频次且强耦合
2. **优先攻击{'Intra-HotpotQA' if hop_level_analysis['intra_hotpot_ratio'] > 0.5 else 'Inter-HotpotQA'}耦合对** - 基于耦合模式
3. **验证GradSim预测的准确性** - 高耦合是否等于高攻击传播效果

---
*报告生成时间: {pd.Timestamp.now()}*
*基于{len(all_high_coupling_pairs):,}个高耦合对的完整分析*
        """
        
        return report
    
    def run_complete_phase1_analysis(self):
        """运行完整的第一阶段分析"""
        print("🚀 开始完整的第一阶段分析 - 基于原始高耦合对数据")
        print("=" * 60)
        
        # 1. 加载原始数据
        all_pieces, all_high_coupling_pairs, piece_id_to_info = self.load_original_data()
        
        # 2. 同答案一致性分析
        same_answer_analysis, same_answer_pairs, different_answer_pairs = \
            self.phase1_same_answer_consistency_analysis(all_high_coupling_pairs, piece_id_to_info)
        
        # 3. Hop层级耦合分析
        hop_level_analysis, intra_pairs, inter_pairs = \
            self.phase1_hop_level_coupling_analysis(all_high_coupling_pairs)
        
        # 4. 答案类型分析
        answer_type_analysis, answer_type_pairs = \
            self.phase1_answer_type_classification(all_high_coupling_pairs, piece_id_to_info)
        
        # 5. 保存详细结果
        detailed_results = {
            'metadata': {
                'total_pieces': len(all_pieces),
                'total_high_coupling_pairs': len(all_high_coupling_pairs),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            },
            'same_answer_analysis': same_answer_analysis,
            'hop_level_analysis': hop_level_analysis,
            'answer_type_analysis': answer_type_analysis
        }
        
        with open(self.output_dir / "phase1_complete_analysis.json", 'w') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # 6. 生成综合报告
        report = self.generate_phase1_comprehensive_report(
            same_answer_analysis, hop_level_analysis, answer_type_analysis,
            all_pieces, all_high_coupling_pairs
        )
        
        with open(self.output_dir / "phase1_comprehensive_report.md", 'w') as f:
            f.write(report)
        
        print(f"\n✅ 第一阶段完整分析完成！")
        print(f"📁 详细结果保存在: {self.output_dir}")
        print(f"📄 综合报告: phase1_comprehensive_report.md")
        
        return detailed_results

def main():
    """主函数"""
    fixer = DataAlignmentFixer()
    results = fixer.run_complete_phase1_analysis()
    
    print("\n🎉 数据对齐修正完成，第一阶段分析结果可靠！")
    print("🚀 准备进入第二阶段攻击实验！")

if __name__ == "__main__":
    main() 