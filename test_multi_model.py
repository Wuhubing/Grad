#!/usr/bin/env python3
"""
Multi-Model Support Test Script

测试GPT-2和LLaMA-2模型架构是否都能正常工作
"""

import json
from knowledge_coupling_mvp import MultiModelKnowledgeCouplingMVP


def create_test_hotpot_data():
    """创建测试用的HotpotQA数据"""
    test_data = [
        {
            "_id": "test_001",
            "question": "What is the nationality of the director of the film 'The Matrix'?",
            "answer": "American",
            "supporting_facts": [
                ["The Matrix", 0],
                ["Wachowski Brothers", 0]
            ],
            "context": [
                ["The Matrix", ["The Matrix is a 1999 film directed by the Wachowski Brothers.", "It's a science fiction movie."]],
                ["Wachowski Brothers", ["The Wachowski Brothers are American filmmakers.", "They are known for The Matrix series."]]
            ]
        },
        {
            "_id": "test_002", 
            "question": "In which city is the headquarters of Apple Inc. located?",
            "answer": "Cupertino",
            "supporting_facts": [
                ["Apple Inc.", 0],
                ["Cupertino", 0]
            ],
            "context": [
                ["Apple Inc.", ["Apple Inc. is headquartered in Cupertino, California.", "It was founded by Steve Jobs."]],
                ["Cupertino", ["Cupertino is a city in California.", "It's located in Silicon Valley."]]
            ]
        }
    ]
    return test_data


def test_model_support(model_path: str, model_name: str):
    """测试指定模型的支持情况"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing {model_name}")
    print(f"Model Path: {model_path}")
    print('='*60)
    
    try:
        # 初始化分析器
        analyzer = MultiModelKnowledgeCouplingMVP(
            model_path=model_path,
            device="cpu"  # 使用CPU确保兼容性
        )
        
        print(f"✅ Model loaded successfully!")
        print(f"   - Detected type: {analyzer.model_type.upper()}")
        print(f"   - Model class: {analyzer.model.__class__.__name__}")
        print(f"   - Tokenizer class: {analyzer.tokenizer.__class__.__name__}")
        
        # 获取目标层信息
        target_layers = analyzer._get_target_layers()
        print(f"   - Target layers: {len(target_layers)} layers")
        if len(target_layers) <= 3:
            for layer in target_layers:
                print(f"     * {layer}")
        else:
            print(f"     * {target_layers[0]}")
            print(f"     * ... ({len(target_layers)-2} more layers)")
            print(f"     * {target_layers[-1]}")
        
        # 测试知识片段提取
        test_data = create_test_hotpot_data()
        knowledge_pieces = analyzer.extract_knowledge_pieces_from_hotpot(test_data, max_samples=2)
        print(f"   - Extracted {len(knowledge_pieces)} knowledge pieces")
        
        # 测试梯度计算
        if knowledge_pieces:
            print(f"   - Testing gradient computation...")
            piece = knowledge_pieces[0]
            print(f"     * Question: {piece.question}")
            print(f"     * Answer: {piece.answer}")
            
            gradient = analyzer.compute_knowledge_gradient(piece)
            if gradient is not None:
                print(f"     * Gradient shape: {gradient.shape}")
                print(f"     * Gradient norm: {(gradient**2).sum()**0.5:.4f}")
                print(f"   ✅ Gradient computation successful!")
            else:
                print(f"   ❌ Gradient computation failed!")
                return False
        
        print(f"\n🎉 {model_name} test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ {model_name} test FAILED!")
        print(f"Error: {e}")
        return False


def main():
    print("🚀 Multi-Model Support Testing")
    print("Testing GPT-2 and LLaMA-2 model architectures...")
    
    # 测试配置
    test_configs = [
        {
            'model_path': 'gpt2',
            'model_name': 'GPT-2 Small'
        },
        # 如果你有更大的GPT-2模型
        # {
        #     'model_path': 'gpt2-medium',
        #     'model_name': 'GPT-2 Medium'
        # },
        # 如果你有LLaMA模型访问权限，取消注释下面的配置
        # {
        #     'model_path': 'meta-llama/Llama-2-7b-hf',
        #     'model_name': 'LLaMA-2 7B'
        # },
    ]
    
    results = []
    
    for config in test_configs:
        success = test_model_support(config['model_path'], config['model_name'])
        results.append({
            'model_name': config['model_name'],
            'model_path': config['model_path'],
            'success': success
        })
    
    # 总结结果
    print(f"\n{'='*60}")
    print("🎯 TESTING SUMMARY")
    print('='*60)
    
    passed = 0
    for result in results:
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"{result['model_name']:<20} {status}")
        if result['success']:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} models tested successfully")
    
    if passed == len(results):
        print("🎉 All models are working correctly!")
        print("\n💡 You can now run the full analysis with:")
        print("   # For GPT-2:")
        print("   python run_complete_mvp.py --hotpot_data YOUR_DATA.json --model_path gpt2")
        print("   # For LLaMA-2 (if available):")
        print("   python run_complete_mvp.py --hotpot_data YOUR_DATA.json --model_path meta-llama/Llama-2-7b-hf")
    else:
        print("⚠️ Some models failed. Check the error messages above.")


if __name__ == "__main__":
    main() 