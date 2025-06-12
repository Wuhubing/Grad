#!/usr/bin/env python3
"""
分析"相同答案"的knowledge pieces的语义含义
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import random

def load_knowledge_pieces():
    """加载知识片段详细信息"""
    knowledge_pieces_file = "results/full_hotpotqa_analysis/final_merged_results/all_knowledge_pieces.json"
    with open(knowledge_pieces_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_same_answer_examples():
    """分析相同答案的具体例子"""
    print("🔍 分析相同答案的knowledge pieces含义...")
    
    # 加载知识片段
    all_pieces = load_knowledge_pieces()
    
    # 创建piece_id到详细信息的映射
    piece_id_to_info = {piece['piece_id']: piece for piece in all_pieces}
    
    # 加载一个批次的高耦合对进行分析
    batch_0_file = "results/full_hotpotqa_analysis/batch_0000/high_coupling_pairs.json"
    with open(batch_0_file, 'r') as f:
        batch_data = json.load(f)
        pairs = batch_data['pairs']
    
    # 按答案类型分组
    answer_groups = defaultdict(list)
    
    for pair in pairs:
        answer = pair['piece_1_answer']  # 相同答案的情况下两个答案一样
        if pair['piece_1_answer'] == pair['piece_2_answer']:
            answer_groups[answer].append(pair)
    
    print(f"✅ 找到 {len(answer_groups)} 种相同答案类型")
    
    # 分析最常见的相同答案
    answer_counts = [(answer, len(pairs)) for answer, pairs in answer_groups.items()]
    answer_counts.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 最常见的相同答案类型:")
    for answer, count in answer_counts[:10]:
        print(f"   '{answer}': {count} 个高耦合对")
    
    # 深入分析几个典型答案
    analyze_answer_semantics("yes", answer_groups["yes"][:5], piece_id_to_info)
    analyze_answer_semantics("no", answer_groups.get("no", [])[:5], piece_id_to_info)
    
    # 分析年份类答案
    year_answers = [answer for answer in answer_groups.keys() if answer.isdigit() and len(answer) == 4]
    if year_answers:
        sample_year = random.choice(year_answers)
        analyze_answer_semantics(sample_year, answer_groups[sample_year][:3], piece_id_to_info)
    
    return answer_groups

def analyze_answer_semantics(answer, pairs, piece_id_to_info):
    """分析特定答案的语义含义"""
    print(f"\n🎯 深入分析答案 '{answer}' 的语义含义:")
    print(f"   高耦合对数量: {len(pairs)}")
    
    for i, pair in enumerate(pairs):
        print(f"\n--- 例子 {i+1} ---")
        
        piece_1_id = pair['piece_1_id']
        piece_2_id = pair['piece_2_id']
        coupling_strength = pair['coupling_strength']
        
        piece_1 = piece_id_to_info.get(piece_1_id, {})
        piece_2 = piece_id_to_info.get(piece_2_id, {})
        
        print(f"耦合强度: {coupling_strength:.4f}")
        print(f"是否同一HotpotQA: {pair['is_same_hotpot']}")
        
        if piece_1:
            print(f"片段1 问题: {piece_1.get('question', 'N/A')[:100]}...")
            print(f"片段1 支持事实: {piece_1.get('supporting_fact', 'N/A')[:100]}...")
        
        if piece_2:
            print(f"片段2 问题: {piece_2.get('question', 'N/A')[:100]}...")
            print(f"片段2 支持事实: {piece_2.get('supporting_fact', 'N/A')[:100]}...")

def analyze_classification_validity():
    """分析分类的有效性 - 是否能对应到模型中的知识节点"""
    print(f"\n\n🧠 分析分类的有效性...")
    
    # 加载知识片段
    all_pieces = load_knowledge_pieces()
    
    # 按答案分组分析
    answer_to_pieces = defaultdict(list)
    
    for piece in all_pieces[:10000]:  # 分析前10000个片段
        answer = piece.get('answer', '').strip()
        if answer:
            answer_to_pieces[answer].append(piece)
    
    # 分析相同答案但不同语义的情况
    print(f"\n📊 相同答案的语义多样性分析:")
    
    # 选择一些有多个实例的答案进行分析
    multi_instance_answers = [(answer, pieces) for answer, pieces in answer_to_pieces.items() 
                             if len(pieces) >= 3]
    multi_instance_answers.sort(key=lambda x: len(x[1]), reverse=True)
    
    for answer, pieces in multi_instance_answers[:5]:
        print(f"\n答案 '{answer}' 的 {len(pieces)} 个不同用法:")
        
        # 分析问题的多样性
        questions = [piece.get('question', '') for piece in pieces[:5]]
        categories = [piece.get('category', '') for piece in pieces[:5]]
        
        for i, (question, category) in enumerate(zip(questions, categories)):
            print(f"  {i+1}. [{category}] {question[:80]}...")

def analyze_model_knowledge_nodes():
    """分析是否能在模型中找到对应的知识节点"""
    print(f"\n\n🤖 分析模型知识节点对应性...")
    
    # 这里我们分析不同类型答案在模型表示空间中的含义
    analysis_cases = [
        {
            'answer': 'yes',
            'semantic_meaning': '布尔判断 - 肯定回答',
            'model_representation': '可能对应模型中的逻辑判断机制',
            'knowledge_node_type': '程序性知识 - 逻辑推理'
        },
        {
            'answer': '1967',
            'semantic_meaning': '具体年份 - 时间信息',
            'model_representation': '可能对应模型中存储的具体事实',
            'knowledge_node_type': '陈述性知识 - 具体事实'
        },
        {
            'answer': 'New York',
            'semantic_meaning': '地理实体 - 地名',
            'model_representation': '可能对应模型中的地理知识图谱节点',
            'knowledge_node_type': '陈述性知识 - 实体知识'
        },
        {
            'answer': 'President Nixon',
            'semantic_meaning': '人物实体 - 政治人物',
            'model_representation': '可能对应模型中的人物知识图谱节点',
            'knowledge_node_type': '陈述性知识 - 人物知识'
        }
    ]
    
    print("不同答案类型的模型知识节点分析:")
    for case in analysis_cases:
        print(f"\n📋 答案: '{case['answer']}'")
        print(f"   语义含义: {case['semantic_meaning']}")
        print(f"   模型表示: {case['model_representation']}")
        print(f"   知识节点类型: {case['knowledge_node_type']}")

def propose_improved_classification():
    """提出改进的分类方案"""
    print(f"\n\n💡 改进的分类方案建议:")
    
    improved_classification = {
        'Boolean_Logic': {
            'examples': ['yes', 'no'],
            'description': '布尔逻辑判断',
            'model_mechanism': '逻辑推理层',
            'attack_potential': '高 - 影响模型的基础逻辑判断'
        },
        'Temporal_Facts': {
            'examples': ['1967', '2005', 'December'],
            'description': '时间相关的具体事实',
            'model_mechanism': '时间知识存储',
            'attack_potential': '中 - 影响历史事件的时间线'
        },
        'Geographic_Entities': {
            'examples': ['New York', 'California', 'United States'],
            'description': '地理位置实体',
            'model_mechanism': '地理知识图谱',
            'attack_potential': '中 - 影响地理相关推理'
        },
        'Person_Entities': {
            'examples': ['President Nixon', 'Obama', 'Einstein'],
            'description': '人物实体',
            'model_mechanism': '人物知识图谱',
            'attack_potential': '高 - 影响与人物相关的多个事实'
        },
        'Numerical_Facts': {
            'examples': ['2,416', '554 ft', '9.2 million'],
            'description': '数值型事实',
            'model_mechanism': '数值知识存储',
            'attack_potential': '低 - 通常为孤立事实'
        }
    }
    
    for category, info in improved_classification.items():
        print(f"\n📊 {category}:")
        print(f"   示例: {', '.join(info['examples'])}")
        print(f"   描述: {info['description']}")
        print(f"   模型机制: {info['model_mechanism']}")
        print(f"   攻击潜力: {info['attack_potential']}")

def main():
    """主函数"""
    print("🔍 深入分析相同答案的knowledge pieces语义含义")
    print("=" * 60)
    
    # 1. 分析相同答案的具体含义
    answer_groups = analyze_same_answer_examples()
    
    # 2. 分析分类有效性
    analyze_classification_validity()
    
    # 3. 分析模型知识节点对应性
    analyze_model_knowledge_nodes()
    
    # 4. 提出改进方案
    propose_improved_classification()
    
    print(f"\n🎯 核心结论:")
    print(f"1. '相同答案'不等于'相同知识' - 需要考虑语义上下文")
    print(f"2. 当前分类过于表面 - 需要更深层的语义分类")
    print(f"3. 模型知识节点对应需要考虑知识类型和机制")
    print(f"4. 攻击策略应该基于知识的语义类型而非字符串匹配")

if __name__ == "__main__":
    main() 