#!/usr/bin/env python3
"""
转换chains格式数据为标准HotpotQA格式

从data/chains/chains_train.jsonl转换为我们系统期望的HotpotQA格式
"""

import json
import random

def convert_chains_to_hotpot(input_file: str, output_file: str, max_samples: int = 20):
    """转换chains格式为HotpotQA格式"""
    
    converted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f):
            if line_no >= max_samples:
                break
                
            try:
                data = json.loads(line.strip())
                
                # 提取基本信息
                question = data['question']
                answer = data['answer']
                hops = data.get('hops', [])
                
                # 创建supporting_facts（简化版）
                supporting_facts = []
                for i, hop in enumerate(hops[:2]):  # 只取前两个hops
                    supporting_facts.append([hop, 0])  # 假设都是第0句
                
                # 创建context（简化版 - 基于hops生成模拟句子）
                context = []
                for i, hop in enumerate(hops[:2]):
                    if i == 0:
                        # 第一个hop - 连接问题和答案
                        sentences = [
                            f"{hop} is related to {answer}.",
                            f"Additional information about {hop}."
                        ]
                    else:
                        # 第二个hop - 提供桥接信息
                        sentences = [
                            f"{hop} connects to the answer {answer}.",
                            f"More details about {hop}."
                        ]
                    
                    context.append([hop, sentences])
                
                # 如果没有足够的hops，添加一些基于答案的context
                if len(context) < 2:
                    context.append([
                        answer, 
                        [f"{answer} is the correct answer to this question.", f"Additional context about {answer}."]
                    ])
                
                # 创建标准HotpotQA格式
                hotpot_item = {
                    "_id": data.get('id', f"converted_{line_no}"),
                    "question": question,
                    "answer": answer,
                    "supporting_facts": supporting_facts,
                    "context": context,
                    "category": data.get('category', 'Unknown'),
                    "hop_count": data.get('hop_count', 2)
                }
                
                converted_data.append(hotpot_item)
                
            except Exception as e:
                print(f"Error processing line {line_no}: {e}")
                continue
    
    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Converted {len(converted_data)} samples")
    print(f"📄 Saved to: {output_file}")
    
    return converted_data

def main():
    print("🔄 Converting chains data to HotpotQA format...")
    
    # 转换前20条数据
    converted_data = convert_chains_to_hotpot(
        "data/chains/chains_train.jsonl",
        "test_hotpot_20.json",
        max_samples=20
    )
    
    # 显示前几条转换后的数据
    print(f"\n📋 First 3 converted samples:")
    for i, item in enumerate(converted_data[:3]):
        print(f"\n--- Sample {i+1} ---")
        print(f"ID: {item['_id']}")
        print(f"Question: {item['question']}")
        print(f"Answer: {item['answer']}")
        print(f"Supporting Facts: {item['supporting_facts']}")
        print(f"Context: {len(item['context'])} pieces")

if __name__ == "__main__":
    main() 