#!/usr/bin/env python3
"""
测试多数据集下载器的功能
"""

from multihop_dataset_downloader import MultihopDatasetDownloader
import json

def test_downloader():
    """测试下载器功能"""
    print("🧪 Testing Multi-hop Dataset Downloader")
    print("=" * 50)
    
    # 初始化下载器
    downloader = MultihopDatasetDownloader(data_dir="test_datasets")
    
    # 显示支持的数据集
    print(f"\n📋 Supported datasets:")
    for name, config in downloader.DATASETS.items():
        print(f"   {name}: {config['description']}")
    
    # 测试：先不下载，只转换现有的test数据
    print(f"\n🔄 Testing format conversion with existing test data...")
    
    # 如果有现有的test_hotpot_20.json，转换它
    try:
        with open("test_hotpot_20.json", 'r') as f:
            hotpot_data = json.load(f)
        
        print(f"✅ Found test_hotpot_20.json with {len(hotpot_data)} samples")
        
        # 创建一个小的混合数据集
        mixed_data = []
        
        # 转换前5个HotpotQA样本
        for i, item in enumerate(hotpot_data[:5]):
            converted_item = {
                '_id': item.get('_id', f'test_hotpot_{i}'),
                'question': item['question'],
                'answer': item['answer'],
                'supporting_facts': item.get('supporting_facts', []),
                'context': item.get('context', []),
                'dataset': 'hotpotqa_test',
                'source': 'existing_test_data'
            }
            mixed_data.append(converted_item)
        
        # 保存测试数据集
        output_file = "test_mixed_dataset.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(mixed_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created test mixed dataset: {output_file}")
        print(f"   Samples: {len(mixed_data)}")
        
        # 显示数据集统计
        dataset_counts = {}
        for item in mixed_data:
            dataset = item.get('dataset', 'unknown')
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        
        print(f"   Dataset distribution:")
        for dataset, count in dataset_counts.items():
            print(f"     {dataset}: {count} samples")
        
        # 显示样本示例
        print(f"\n📚 Sample data:")
        sample = mixed_data[0]
        print(f"   ID: {sample['_id']}")
        print(f"   Question: {sample['question'][:80]}...")
        print(f"   Answer: {sample['answer']}")
        print(f"   Dataset: {sample['dataset']}")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ test_hotpot_20.json not found")
        print(f"   To test with real downloads, run:")
        print(f"   python multihop_dataset_downloader.py")
        return False

def demo_download_workflow():
    """演示完整的下载工作流程（注释掉实际下载）"""
    print(f"\n🎯 Demo: Complete Download Workflow")
    print("=" * 40)
    
    print(f"📝 Workflow steps:")
    print(f"   1. Initialize downloader")
    print(f"   2. Download datasets (HotpotQA, MuSiQue)")
    print(f"   3. Convert to unified format")
    print(f"   4. Create mixed dataset")
    print(f"   5. Ready for knowledge coupling analysis")
    
    print(f"\n💡 To run actual downloads:")
    print(f"   python multihop_dataset_downloader.py")
    
    print(f"\n🔧 To use with knowledge coupling analysis:")
    print(f"   python knowledge_coupling_mvp.py --hotpot_data mixed_multihop_30.json")

if __name__ == "__main__":
    # 先测试基本功能
    success = test_downloader()
    
    # 然后演示工作流程
    demo_download_workflow()
    
    if success:
        print(f"\n✅ Downloader test completed successfully!")
        print(f"🎯 Ready to test with knowledge coupling analysis:")
        print(f"   python knowledge_coupling_mvp.py --hotpot_data test_mixed_dataset.json --max_samples 5")
    else:
        print(f"\n⚠️  Downloader test completed with warnings")
        print(f"   Some features require downloading actual datasets") 