#!/usr/bin/env python3
"""
演示脚本：展示knowledge_coupling_mvp.py返回的详细results结构
"""

import json
from knowledge_coupling_mvp import MultiModelKnowledgeCouplingMVP, load_hotpot_data

def demo_detailed_results():
    """演示详细的分析结果"""
    print("🚀 Knowledge Coupling MVP - Results Demo")
    print("=" * 60)
    
    # 加载数据
    print("📚 Loading test data...")
    hotpot_data = load_hotpot_data("test_hotpot_20.json")
    
    # 初始化分析器
    print("🤖 Initializing analyzer...")
    analyzer = MultiModelKnowledgeCouplingMVP(
        model_path="gpt2",
        device="cuda"
    )
    
    # 运行分析
    print("🔬 Running analysis...")
    results = analyzer.run_mvp_analysis(hotpot_data, max_samples=5, output_dir="demo_results")
    
    # 展示results结构
    print("\n🎯 DETAILED RESULTS STRUCTURE")
    print("=" * 60)
    
    # 1. 基本信息
    print("\n📋 1. METADATA:")
    metadata = results['metadata']
    for key, value in metadata.items():
        print(f"   {key}: {value}")
    
    # 2. 知识片段信息
    print(f"\n📚 2. KNOWLEDGE PIECES:")
    kp = results['knowledge_pieces']
    print(f"   Total count: {kp['count']}")
    print(f"   Categories: {kp['categories']}")
    print(f"   Sample piece details:")
    for i, piece in enumerate(kp['details'][:2]):  # 显示前2个
        print(f"      Piece {i+1}:")
        print(f"         ID: {piece['piece_id']}")
        print(f"         Answer: '{piece['answer']}'")
        print(f"         Category: {piece['category']}")
        print(f"         Question: {piece['question'][:60]}...")
    
    # 3. 梯度分析
    print(f"\n🧮 3. GRADIENT ANALYSIS:")
    ga = results['gradient_analysis']
    for key, value in ga.items():
        print(f"   {key}: {value}")
    
    # 4. 耦合分析
    print(f"\n🔗 4. COUPLING ANALYSIS:")
    ca = results['coupling_analysis']
    for key, value in ca.items():
        print(f"   {key}: {value}")
    
    # 5. 高耦合对
    print(f"\n🔥 5. HIGH COUPLING PAIRS:")
    hcp = results['high_coupling_pairs']
    print(f"   Total count: {hcp['count']}")
    print(f"   Threshold: {hcp['threshold']}")
    print(f"   Top 3 detailed pairs:")
    for i, pair in enumerate(hcp['top_10_detailed'][:3]):
        print(f"      Rank {pair['rank']}:")
        print(f"         Coupling: {pair['coupling_strength']:.4f}")
        print(f"         Source: '{pair['source_answer']}' ({pair['source_category']})")
        print(f"         Target: '{pair['target_answer']}' ({pair['target_category']})")
    
    # 6. 跨pot分析
    print(f"\n🍯 6. CROSS-POT ANALYSIS:")
    cpa = results['cross_pot_analysis']
    for key, value in cpa.items():
        print(f"   {key}: {value}")
    
    # 7. 生成的文件
    print(f"\n📁 7. GENERATED FILES:")
    gf = results['generated_files']
    for key, path in gf.items():
        print(f"   {key}: {path}")
    
    # 8. GPU性能
    print(f"\n🚀 8. GPU PERFORMANCE:")
    gp = results['gpu_performance']
    for key, value in gp.items():
        print(f"   {key}: {value}")
    
    # 9. 实验准备
    print(f"\n🧪 9. EXPERIMENT READINESS:")
    er = results['experiment_readiness']
    for key, value in er.items():
        print(f"   {key}: {value}")
    
    # 保存完整results为JSON
    results_file = "complete_results_demo.json"
    # 需要处理GPU张量，转换为可序列化格式
    serializable_results = convert_to_serializable(results)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Complete results saved to: {results_file}")
    print(f"🎯 Results dictionary contains {len(results)} main sections")
    print(f"📊 Ready for further analysis or visualization!")

def convert_to_serializable(obj):
    """将包含PyTorch张量的对象转换为可序列化格式"""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, 'tolist'):  # PyTorch tensors or numpy arrays
        return obj.tolist()
    elif hasattr(obj, 'item'):  # Single value tensors
        return obj.item()
    else:
        return obj

if __name__ == "__main__":
    demo_detailed_results() 