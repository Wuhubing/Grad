#!/usr/bin/env python3
"""
第1.5阶段：数据质量提升和语义分类改进
目标：为第二阶段的攻击实验做好数据准备
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import re
from typing import Dict, List, Tuple, Any

class DataQualityImprover:
    """数据质量提升器"""
    
    def __init__(self, results_dir: str = "results/full_hotpotqa_analysis"):
        self.results_dir = Path(results_dir)
        self.final_results_dir = self.results_dir / "final_merged_results"
        self.output_dir = Path("results/quality_improved_analysis")
        self.output_dir.mkdir(exist_ok=True)
        
        print("🔧 数据质量提升器初始化")
        
    def step1_clean_invalid_answers(self):
        """步骤1：清理无效答案"""
        print("\n🧹 步骤1：清理无效答案...")
        
        # 加载所有知识片段
        with open(self.final_results_dir / "all_knowledge_pieces.json", 'r') as f:
            all_pieces = json.load(f)
        
        print(f"原始知识片段数量: {len(all_pieces)}")
        
        # 定义无效答案模式
        invalid_patterns = [
            r'^___$',  # 纯占位符
            r'what is the answer\?',  # 问题格式错误
            r'^Based on the fact.*what is the answer\?.*$',  # 格式错误
        ]
        
        valid_pieces = []
        invalid_count = 0
        
        for piece in all_pieces:
            answer = piece.get('answer', '').strip()
            question = piece.get('question', '').strip()
            
            # 检查是否为无效答案
            is_invalid = False
            for pattern in invalid_patterns:
                if re.search(pattern, answer, re.IGNORECASE) or re.search(pattern, question, re.IGNORECASE):
                    is_invalid = True
                    break
            
            # 检查填空题但答案不匹配的情况
            if 'Fill in the blank:' in question and '___' in question:
                # 这种情况下答案应该是填空的内容
                if not self._validate_fill_blank_answer(question, answer):
                    is_invalid = True
            
            if not is_invalid:
                valid_pieces.append(piece)
            else:
                invalid_count += 1
        
        print(f"清理后有效知识片段数量: {len(valid_pieces)}")
        print(f"清理掉的无效片段数量: {invalid_count}")
        
        # 保存清理后的数据
        with open(self.output_dir / "cleaned_knowledge_pieces.json", 'w') as f:
            json.dump(valid_pieces, f, indent=2, ensure_ascii=False)
        
        return valid_pieces
    
    def _validate_fill_blank_answer(self, question: str, answer: str) -> bool:
        """验证填空题答案的合理性"""
        # 简单验证：答案不应该是占位符
        if answer.strip() in ['___', 'yes', 'no'] and 'Fill in the blank:' in question:
            return False
        return True
    
    def step2_semantic_question_classification(self, valid_pieces: List[Dict]):
        """步骤2：基于语义的问题分类"""
        print("\n🧠 步骤2：基于语义的问题分类...")
        
        # 定义语义分类规则
        semantic_categories = {
            'Temporal_Questions': {
                'patterns': [r'when', r'what year', r'which year', r'\d{4}', r'founded', r'established', r'born'],
                'description': '时间相关问题'
            },
            'Entity_Relation_Questions': {
                'patterns': [r'who is', r'what is', r'which.*is', r'name.*person', r'founder', r'director'],
                'description': '实体关系问题'
            },
            'Boolean_Logic_Questions': {
                'patterns': [r'is.*same', r'are.*both', r'do.*both', r'true or false', r'whether'],
                'description': '布尔逻辑问题'
            },
            'Location_Questions': {
                'patterns': [r'where', r'located', r'country', r'city', r'state'],
                'description': '地理位置问题'
            },
            'Quantity_Questions': {
                'patterns': [r'how many', r'how much', r'number of', r'count'],
                'description': '数量问题'
            }
        }
        
        categorized_pieces = defaultdict(list)
        
        for piece in valid_pieces:
            question = piece.get('question', '').lower()
            answer = piece.get('answer', '').lower()
            
            # 分类逻辑
            category = self._classify_question_semantically(question, answer, semantic_categories)
            piece['semantic_category'] = category
            categorized_pieces[category].append(piece)
        
        # 统计分类结果
        print("\n📊 语义分类结果:")
        for category, pieces in categorized_pieces.items():
            print(f"   {category}: {len(pieces)} 个片段")
        
        # 保存分类结果
        with open(self.output_dir / "semantically_categorized_pieces.json", 'w') as f:
            json.dump(dict(categorized_pieces), f, indent=2, ensure_ascii=False)
        
        return categorized_pieces
    
    def _classify_question_semantically(self, question: str, answer: str, categories: Dict) -> str:
        """基于语义对问题进行分类"""
        # 检查答案类型
        if answer in ['yes', 'no'] and any(pattern in question for pattern in ['is', 'are', 'do', 'does']):
            return 'Boolean_Logic_Questions'
        
        if re.match(r'^\d{4}$', answer.strip()):
            return 'Temporal_Questions'
        
        # 基于问题模式分类
        for category, info in categories.items():
            for pattern in info['patterns']:
                if re.search(pattern, question, re.IGNORECASE):
                    return category
        
        return 'Other_Questions'
    
    def step3_rebuild_coupling_analysis(self, categorized_pieces: Dict):
        """步骤3：基于语义分类重建耦合分析"""
        print("\n🔗 步骤3：基于语义分类重建耦合分析...")
        
        # 加载高耦合对
        batch_0_file = self.results_dir / "batch_0000" / "high_coupling_pairs.json"
        with open(batch_0_file, 'r') as f:
            batch_data = json.load(f)
            pairs = batch_data['pairs']
        
        # 创建piece_id到语义类别的映射
        piece_id_to_category = {}
        all_pieces_flat = []
        for category, pieces in categorized_pieces.items():
            for piece in pieces:
                piece_id_to_category[piece['piece_id']] = category
                all_pieces_flat.append(piece)
        
        # 重新分析耦合对
        semantic_coupling_analysis = {
            'same_semantic_category': {'pairs': [], 'strengths': []},
            'different_semantic_category': {'pairs': [], 'strengths': []},
            'same_semantic_and_answer': {'pairs': [], 'strengths': []},
            'same_semantic_different_answer': {'pairs': [], 'strengths': []}
        }
        
        for pair in pairs:
            piece_1_id = pair['piece_1_id']
            piece_2_id = pair['piece_2_id']
            
            category_1 = piece_id_to_category.get(piece_1_id)
            category_2 = piece_id_to_category.get(piece_2_id)
            
            if category_1 and category_2:
                if category_1 == category_2:
                    semantic_coupling_analysis['same_semantic_category']['pairs'].append(pair)
                    semantic_coupling_analysis['same_semantic_category']['strengths'].append(pair['coupling_strength'])
                    
                    # 进一步检查相同语义类别中答案是否相同
                    if pair['piece_1_answer'] == pair['piece_2_answer']:
                        semantic_coupling_analysis['same_semantic_and_answer']['pairs'].append(pair)
                        semantic_coupling_analysis['same_semantic_and_answer']['strengths'].append(pair['coupling_strength'])
                    else:
                        semantic_coupling_analysis['same_semantic_different_answer']['pairs'].append(pair)
                        semantic_coupling_analysis['same_semantic_different_answer']['strengths'].append(pair['coupling_strength'])
                else:
                    semantic_coupling_analysis['different_semantic_category']['pairs'].append(pair)
                    semantic_coupling_analysis['different_semantic_category']['strengths'].append(pair['coupling_strength'])
        
        # 计算统计信息
        print("\n📊 基于语义的耦合分析结果:")
        for analysis_type, data in semantic_coupling_analysis.items():
            if data['strengths']:
                avg_strength = np.mean(data['strengths'])
                count = len(data['pairs'])
                print(f"   {analysis_type}: {count} 对, 平均强度: {avg_strength:.4f}")
        
        # 保存分析结果
        results_summary = {
            'semantic_coupling_statistics': {
                analysis_type: {
                    'count': len(data['pairs']),
                    'average_strength': float(np.mean(data['strengths'])) if data['strengths'] else 0,
                    'std_strength': float(np.std(data['strengths'])) if data['strengths'] else 0
                }
                for analysis_type, data in semantic_coupling_analysis.items()
            }
        }
        
        with open(self.output_dir / "semantic_coupling_analysis.json", 'w') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        return semantic_coupling_analysis
    
    def run_full_pipeline(self):
        """运行完整的数据质量提升流程"""
        print("🚀 开始第1.5阶段：数据质量提升和语义分类改进")
        print("=" * 60)
        
        # 步骤1：清理无效答案
        valid_pieces = self.step1_clean_invalid_answers()
        
        # 步骤2：语义分类
        categorized_pieces = self.step2_semantic_question_classification(valid_pieces)
        
        # 步骤3：重建耦合分析
        semantic_coupling_analysis = self.step3_rebuild_coupling_analysis(categorized_pieces)
        
        print(f"\n✅ 第1.5阶段完成！")
        print(f"📁 结果保存在: {self.output_dir}")
        
        return {
            'valid_pieces': valid_pieces,
            'categorized_pieces': categorized_pieces,
            'semantic_coupling_analysis': semantic_coupling_analysis
        }

def main():
    """主函数"""
    improver = DataQualityImprover()
    results = improver.run_full_pipeline()
    
    print("\n🎉 第1.5阶段完成，准备进入第二阶段攻击实验！")

if __name__ == "__main__":
    main() 