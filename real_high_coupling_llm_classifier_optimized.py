#!/usr/bin/env python3
"""
优化版真实高耦合对LLM语义分类器
- 添加详细进度条
- 支持后台运行
- 优化分类类别
- 实时保存结果
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import time
import sys
import signal
import os
from typing import Dict, List, Tuple, Any
import logging
import argparse

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classification_progress.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedHighCouplingLLMClassifier:
    """优化版真实高耦合对LLM分类器"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-14B-Instruct"):
        self.model_name = model_name
        self.results_dir = Path("results/full_hotpotqa_analysis")
        self.output_dir = Path("results/optimized_high_coupling_classification")
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建检查点目录
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        print(f"🤖 优化版高耦合对LLM分类器初始化")
        print(f"   模型: {model_name}")
        print(f"   GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
        print(f"   输出目录: {self.output_dir}")
        
        # 根据模型大小调整生成参数
        if "14B" in model_name or "32B" in model_name:
            self.max_new_tokens = 35
            self.temperature = 0.05  # 更低温度，更确定性
            print(f"   🎯 大模型配置: max_tokens={self.max_new_tokens}, temp={self.temperature}")
        else:
            self.max_new_tokens = 30
            self.temperature = 0.1
            print(f"   🎯 标准配置: max_tokens={self.max_new_tokens}, temp={self.temperature}")
        
        # 初始化模型和tokenizer
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 设置信号处理器用于优雅停止
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.stop_requested = False
        
    def _signal_handler(self, signum, frame):
        """信号处理器，用于优雅停止"""
        print(f"\n⚠️ 接收到停止信号 {signum}，正在保存当前进度...")
        self.stop_requested = True
        
    def load_model(self):
        """加载模型 - 支持7B/14B/32B"""
        print(f"\n🔄 加载模型: {self.model_name}")
        
        try:
            # 根据模型大小调整加载策略
            if "14B" in self.model_name:
                print(f"   🎯 14B模型优化配置:")
                print(f"      - 使用FP16精度")
                print(f"      - 启用gradient checkpointing") 
                print(f"      - 优化显存管理")
                
                torch_dtype = torch.float16
                device_map = "auto"
                additional_kwargs = {
                    "use_cache": True,
                    "torch_dtype": torch_dtype
                }
                
            elif "32B" in self.model_name:
                print(f"   🎯 32B模型优化配置:")
                print(f"      - 使用FP16精度")
                print(f"      - 多GPU分布策略")
                
                torch_dtype = torch.float16
                device_map = "auto"
                additional_kwargs = {
                    "use_cache": True,
                    "torch_dtype": torch_dtype,
                    "low_cpu_mem_usage": True
                }
                
            else:  # 7B模型
                print(f"   🎯 7B模型标准配置")
                torch_dtype = torch.float16
                device_map = "auto"
                additional_kwargs = {
                    "torch_dtype": torch_dtype
                }
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                trust_remote_code=True,
                **additional_kwargs
            )
            
            # 显示加载结果
            if torch.cuda.is_available():
                total_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"   ✅ 模型加载成功")
                print(f"   📊 显存使用: {total_memory:.1f}GB")
                
                # 14B模型特殊提示
                if "14B" in self.model_name:
                    print(f"   🎯 14B模型已启用，预期更高的语义分类准确率")
                    if total_memory > 35:
                        print(f"   ⚠️ 显存使用较高，建议监控GPU状态")
            else:
                print(f"   ✅ 模型加载成功 (CPU模式)")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            print(f"   💡 建议:")
            if "14B" in self.model_name:
                print(f"      - 确保GPU显存至少30GB")
                print(f"      - 考虑使用 --model Qwen/Qwen2.5-7B-Instruct")
            raise
    
    def load_all_high_coupling_pairs(self):
        """加载所有高耦合对数据"""
        print(f"\n📊 加载所有高耦合对数据...")
        
        all_pairs = []
        batch_dirs = [d for d in self.results_dir.iterdir() if d.is_dir() and d.name.startswith('batch_')]
        
        # 添加进度条
        with tqdm(batch_dirs, desc="加载批次", unit="batch") as pbar:
            for batch_dir in pbar:
                coupling_file = batch_dir / "high_coupling_pairs.json"
                if coupling_file.exists():
                    with open(coupling_file, 'r') as f:
                        batch_data = json.load(f)
                        all_pairs.extend(batch_data['pairs'])
                    pbar.set_postfix({'当前批次': batch_dir.name, '总对数': len(all_pairs)})
        
        print(f"   从 {len(batch_dirs)} 个批次加载了 {len(all_pairs):,} 个高耦合对")
        return all_pairs
    
    def extract_unique_answers(self, high_coupling_pairs: List[Dict], sample_size: int = None):
        """从高耦合对中提取所有唯一答案"""
        print(f"\n📋 提取唯一答案...")
        
        all_answers = set()
        
        # 添加进度条
        with tqdm(high_coupling_pairs, desc="提取答案", unit="对") as pbar:
            for pair in pbar:
                all_answers.add(pair['piece_1_answer'])
                all_answers.add(pair['piece_2_answer'])
                if len(all_answers) % 1000 == 0:
                    pbar.set_postfix({'唯一答案数': len(all_answers)})
        
        unique_answers = list(all_answers)
        
        # 如果需要采样
        if sample_size and len(unique_answers) > sample_size:
            import random
            random.seed(42)  # 固定种子确保可复现
            unique_answers = random.sample(unique_answers, sample_size)
        
        print(f"   总唯一答案数: {len(all_answers):,}")
        if sample_size:
            print(f"   采样答案数: {len(unique_answers):,}")
        
        # 显示一些真实数据例子
        print(f"\n📋 真实答案样例:")
        for i, answer in enumerate(sorted(unique_answers)[:15]):
            print(f"   {i+1}. '{answer}'")
        
        return unique_answers
    
    def create_enhanced_classification_prompt(self, answer: str) -> str:
        """创建增强的分类提示词 - 更精确的类别"""
        
        prompt = f"""你是一个专业的语义分析专家。请将下面的答案分类到最合适的语义类别中。

答案: "{answer}"

请从以下类别中选择最合适的一个：

1. **Person_Name** - 人名（如：Christopher Paolini, Ben Miller, Albert Einstein）
2. **Place_Name** - 地名（如：Cheltenham, Neosho River, New York City）
3. **Organization_Name** - 组织机构名（如：Stanford University, Apple Inc, NATO）
4. **Creative_Work_Title** - 作品标题（如：Kingdom Hearts, Harry Potter, "Bohemian Rhapsody"）
5. **Brand_Product** - 品牌产品（如：iPhone, Toyota, Coca-Cola）
6. **Event_Name** - 事件名称（如：World War II, Battle of Midway, Olympics）
7. **Movie_Film** - 电影影视（如：Avatar, The Matrix, Game of Thrones）
8. **Book_Literature** - 书籍文学（如：Pride and Prejudice, The Bible）
9. **Music_Song** - 音乐歌曲（如：Yesterday, Thriller album）
10. **Sports_Team** - 体育团队（如：Lakers, Manchester United）
11. **Academic_Field** - 学术领域（如：psychology, quantum physics）
12. **Job_Profession** - 职业工作（如：software engineer, teacher）
13. **Animal_Species** - 动物物种（如：tiger, golden retriever）
14. **Food_Cuisine** - 食物料理（如：pizza, sushi）
15. **Technology_Term** - 技术术语（如：artificial intelligence, blockchain）
16. **Concept_Abstract** - 抽象概念（如：democracy, love, freedom）
17. **Descriptive_Phrase** - 描述性短语（如：the Desert Fox, very tall building）
18. **Nationality_Ethnicity** - 国籍民族（如：American, Chinese, European）
19. **Time_Period** - 时间时期（如：Middle Ages, Renaissance）
20. **Number_Quantity** - 数字数量（如：2,416, $4 billion, 1967）
21. **Boolean_Answer** - 是非回答（如：yes, no）
22. **Color_Appearance** - 颜色外观（如：red, transparent, metallic）
23. **Direction_Location** - 方向位置（如：north, downtown, upstairs）
24. **Other** - 其他无法明确分类的

请只回答类别名称，例如：Person_Name

类别："""
        
        return prompt
    
    def classify_single_answer(self, answer: str) -> str:
        """对单个答案进行分类"""
        
        prompt = self.create_enhanced_classification_prompt(answer)
        
        try:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码回答
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # 提取类别名称
            category = response.strip().split('\n')[0].strip()
            
            # 验证类别是否有效
            valid_categories = {
                'Person_Name', 'Place_Name', 'Organization_Name', 'Creative_Work_Title',
                'Brand_Product', 'Event_Name', 'Movie_Film', 'Book_Literature', 'Music_Song',
                'Sports_Team', 'Academic_Field', 'Job_Profession', 'Animal_Species', 'Food_Cuisine',
                'Technology_Term', 'Concept_Abstract', 'Descriptive_Phrase', 'Nationality_Ethnicity',
                'Time_Period', 'Number_Quantity', 'Boolean_Answer', 'Color_Appearance', 
                'Direction_Location', 'Other'
            }
            
            if category in valid_categories:
                return category
            else:
                # 如果回答不在预期类别中，尝试从回答中提取
                for cat in valid_categories:
                    if cat.lower() in category.lower():
                        return cat
                return 'Other'  # 默认返回Other
                
        except Exception as e:
            logger.error(f"分类错误 '{answer}': {e}")
            return 'Other'
    
    def save_checkpoint(self, results: Dict[str, str], processed_count: int, total_count: int):
        """保存检查点"""
        checkpoint_data = {
            'processed_count': processed_count,
            'total_count': total_count,
            'results': results,
            'timestamp': pd.Timestamp.now().isoformat(),
            'progress_percentage': processed_count / total_count * 100
        }
        
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{processed_count}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"检查点已保存: {checkpoint_file} ({processed_count}/{total_count})")
    
    def load_checkpoint(self) -> Tuple[Dict[str, str], int]:
        """加载最新的检查点"""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        
        if not checkpoint_files:
            return {}, 0
        
        # 找到最新的检查点
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[1]))
        
        with open(latest_checkpoint, 'r') as f:
            checkpoint_data = json.load(f)
        
        print(f"🔄 加载检查点: {latest_checkpoint}")
        print(f"   已处理: {checkpoint_data['processed_count']}/{checkpoint_data['total_count']}")
        print(f"   进度: {checkpoint_data['progress_percentage']:.1f}%")
        
        return checkpoint_data['results'], checkpoint_data['processed_count']
    
    def batch_classify_answers_with_progress(self, answers: List[str], batch_size: int = 10, 
                                           checkpoint_interval: int = 50):
        """批量分类答案 - 带进度条和检查点"""
        print(f"\n🔄 开始分类真实高耦合对答案...")
        print(f"   答案数量: {len(answers)}")
        print(f"   批次大小: {batch_size}")
        print(f"   检查点间隔: {checkpoint_interval}")
        
        # 尝试加载检查点
        results, start_index = self.load_checkpoint()
        classification_details = []
        
        if start_index > 0:
            print(f"   📍 从检查点继续: 第 {start_index} 个答案")
            answers = answers[start_index:]
        
        total_answers = len(answers) + start_index
        processed_count = start_index
        
        # 创建详细的进度条
        with tqdm(total=len(answers), desc="分类进度", unit="答案", 
                 initial=0, dynamic_ncols=True) as pbar:
            
            for i in range(0, len(answers), batch_size):
                if self.stop_requested:
                    print(f"\n⚠️ 收到停止请求，保存当前进度...")
                    self.save_checkpoint(results, processed_count, total_answers)
                    break
                
                batch_answers = answers[i:i+batch_size]
                
                for j, answer in enumerate(batch_answers):
                    if self.stop_requested:
                        break
                        
                    start_time = time.time()
                    category = self.classify_single_answer(answer)
                    inference_time = time.time() - start_time
                    
                    results[answer] = category
                    classification_details.append({
                        'answer': answer,
                        'predicted_category': category,
                        'inference_time': inference_time
                    })
                    
                    processed_count += 1
                    
                    # 更新进度条
                    pbar.set_postfix({
                        '当前': f'"{answer[:30]}..."',
                        '类别': category,
                        '时间': f'{inference_time:.2f}s',
                        '显存': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
                    })
                    pbar.update(1)
                    
                    # 每5个答案休息一下
                    if processed_count % 5 == 0:
                        time.sleep(0.1)
                    
                    # 保存检查点
                    if processed_count % checkpoint_interval == 0:
                        self.save_checkpoint(results, processed_count, total_answers)
                        pbar.set_description(f"分类进度 (检查点已保存)")
                
                # 每个批次后显示详细进度
                if (i // batch_size + 1) % 3 == 0:
                    avg_time = np.mean([d['inference_time'] for d in classification_details[-30:]])
                    remaining = len(answers) - (i + len(batch_answers))
                    eta = remaining * avg_time / 60  # 分钟
                    
                    pbar.set_description(f"分类进度 (ETA: {eta:.1f}分钟)")
        
        # 保存最终检查点
        if not self.stop_requested:
            self.save_checkpoint(results, processed_count, total_answers)
        
        return results, classification_details
    
    def analyze_classification_results_enhanced(self, results: Dict[str, str], 
                                              classification_details: List[Dict], 
                                              high_coupling_pairs: List[Dict]):
        """增强的分类结果分析"""
        print(f"\n📊 分析分类结果...")
        
        # 统计各类别数量
        category_counts = Counter(results.values())
        total_answers = len(results)
        
        print(f"\n🏆 真实高耦合对答案分类统计 (Top 15):")
        for category, count in category_counts.most_common(15):
            percentage = count / total_answers * 100
            print(f"   {category}: {count} ({percentage:.1f}%)")
        
        # 生成每个类别的示例
        category_examples = defaultdict(list)
        for answer, category in results.items():
            category_examples[category].append(answer)
        
        print(f"\n📋 各类别真实数据示例 (Top 10类别):")
        for category, _ in category_counts.most_common(10):
            examples = category_examples[category][:3]  # 只显示前3个例子
            print(f"   {category}: {examples}")
        
        # 计算平均推理时间
        if classification_details:
            avg_inference_time = sum(d['inference_time'] for d in classification_details) / len(classification_details)
            total_time = sum(d['inference_time'] for d in classification_details)
            
            print(f"\n⏱️ 性能统计:")
            print(f"   平均推理时间: {avg_inference_time:.3f}s/样本")
            print(f"   总处理时间: {total_time:.1f}s ({total_time/60:.1f}分钟)")
            
            # 预估全量处理时间
            total_unique_answers = 34924  # 从之前加载的数据得知
            estimated_full_time = avg_inference_time * total_unique_answers / 3600  # 小时
            print(f"   预估全量处理时间: {estimated_full_time:.1f}小时")
        
        return {
            'category_counts': dict(category_counts),
            'category_examples': {k: v[:10] for k, v in category_examples.items()},
            'avg_inference_time': avg_inference_time if classification_details else 0,
            'total_classified': total_answers,
            'classification_details': classification_details
        }
    
    def save_final_results(self, results: Dict[str, str], analysis: Dict):
        """保存最终结果"""
        print(f"\n💾 保存最终分类结果...")
        
        # 1. 保存完整的LLM分类结果
        results_file = self.output_dir / "final_answer_classifications.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 2. 保存CSV格式便于分析
        csv_file = self.output_dir / "final_answer_classifications.csv"
        df = pd.DataFrame([
            {'answer': answer, 'category': category} 
            for answer, category in results.items()
        ])
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # 3. 保存分析统计
        stats_file = self.output_dir / "classification_statistics.json"
        final_stats = {
            'metadata': {
                'model_name': self.model_name,
                'total_classified': len(results),
                'classification_timestamp': pd.Timestamp.now().isoformat(),
                'data_source': 'real_high_coupling_pairs',
                'categories_count': len(set(results.values()))
            },
            'analysis': analysis
        }
        
        with open(stats_file, 'w') as f:
            json.dump(final_stats, f, indent=2, ensure_ascii=False)
        
        # 4. 生成简要报告
        self.generate_summary_report(results, analysis)
        
        print(f"   ✅ 结果已保存到: {self.output_dir}")
        print(f"      - 分类结果: final_answer_classifications.json")
        print(f"      - CSV格式: final_answer_classifications.csv") 
        print(f"      - 统计信息: classification_statistics.json")
        print(f"      - 总结报告: classification_summary.md")
    
    def generate_summary_report(self, results: Dict[str, str], analysis: Dict):
        """生成总结报告"""
        
        total_samples = len(results)
        category_counts = analysis['category_counts']
        
        report = f"""
# 真实高耦合对答案LLM分类总结报告

## 📊 分析概览

### 基本信息
- **使用模型**: {self.model_name}
- **数据来源**: 真实HotpotQA高耦合对 (coupling_strength ≥ 0.4)
- **分析样本数**: {total_samples:,} 个唯一答案
- **识别类别数**: {len(set(results.values()))} 个语义类别
- **平均推理时间**: {analysis['avg_inference_time']:.3f}s/样本
- **分析时间**: {pd.Timestamp.now()}

## 🏆 主要分类结果

### Top 15 语义类别分布
"""
        
        # 添加分类结果统计
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (category, count) in enumerate(sorted_categories[:15], 1):
            percentage = count / total_samples * 100
            report += f"""
{i}. **{category}**: {count:,} 个 ({percentage:.1f}%)"""
        
        report += f"""

### 🎯 分类示例展示

以下是各主要类别的真实数据示例：
"""
        
        # 添加前10个类别的例子
        category_examples = analysis['category_examples']
        for category, _ in sorted_categories[:10]:
            examples = category_examples.get(category, [])[:3]
            if examples:
                report += f"""
#### {category}
- 示例: {', '.join(f'"{ex}"' for ex in examples)}
"""
        
        report += f"""

## 📈 关键发现

### 🔍 数据洞察
1. **最大类别**: {sorted_categories[0][0]} ({sorted_categories[0][1]:,}个, {sorted_categories[0][1]/total_samples*100:.1f}%)
2. **多样性**: 识别出 {len(set(results.values()))} 个不同的语义类别
3. **分类效果**: 显著提升了答案语义理解的精确度

### 🎯 攻击实验指导
- **优先目标**: {sorted_categories[0][0]}, {sorted_categories[1][0]}, {sorted_categories[2][0]}
- **频次优势**: 前3类别占总数的 {sum(count for _, count in sorted_categories[:3])/total_samples*100:.1f}%
- **实验价值**: 为Phase 2后门攻击实验提供精确的目标选择

## 🚀 下一步建议

1. **选择攻击目标**: 基于频次和语义特征选择主要攻击类别
2. **设计MEMIT实验**: 针对高耦合强度的目标对进行知识编辑
3. **验证传播效果**: 测试攻击在同语义类别内的传播效果

---
*基于{total_samples:,}个真实高耦合对答案的完整LLM分类分析*
*生成时间: {pd.Timestamp.now()}*
        """
        
        with open(self.output_dir / "classification_summary.md", 'w') as f:
            f.write(report)
    
    def run_optimized_classification(self, sample_size: int = 300):
        """运行优化的分类流程"""
        print("🚀 开始优化版真实高耦合对答案LLM分类")
        print("=" * 70)
        
        try:
            # 1. 加载模型
            self.load_model()
            
            # 2. 加载高耦合对数据
            high_coupling_pairs = self.load_all_high_coupling_pairs()
            
            # 3. 提取唯一答案
            unique_answers = self.extract_unique_answers(high_coupling_pairs, sample_size)
            
            # 4. 批量分类（带进度条和检查点）
            llm_results, details = self.batch_classify_answers_with_progress(
                unique_answers, batch_size=10, checkpoint_interval=50
            )
            
            # 5. 分析结果
            analysis = self.analyze_classification_results_enhanced(
                llm_results, details, high_coupling_pairs
            )
            
            # 6. 保存最终结果
            self.save_final_results(llm_results, analysis)
            
            print(f"\n🎉 优化版LLM分类完成！")
            print(f"📊 分类了 {len(llm_results):,} 个真实答案")
            print(f"🏆 识别出 {len(set(llm_results.values()))} 个语义类别")
            print(f"📄 详细结果查看: {self.output_dir}")
            
            return llm_results, analysis
            
        except Exception as e:
            logger.error(f"分类过程出错: {e}")
            # 即使出错也尝试保存当前结果
            if hasattr(self, 'llm_results'):
                self.save_checkpoint(self.llm_results, len(self.llm_results), sample_size)
            raise
        
        finally:
            # 清理显存
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                torch.cuda.empty_cache()
                print("   🧹 已清理GPU显存")

def main():
    """主函数"""
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='优化版真实高耦合对LLM语义分类器')
    parser.add_argument('--sample-size', type=int, default=300, 
                       help='分析的样本大小 (默认: 300)')
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-14B-Instruct",
                       choices=[
                           "Qwen/Qwen2.5-7B-Instruct", 
                           "Qwen/Qwen2.5-14B-Instruct",
                           "Qwen/Qwen2.5-32B-Instruct"
                       ],
                       help='使用的模型 (默认: Qwen2.5-14B-Instruct)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='批处理大小 (默认: 10)')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                       help='检查点保存间隔 (默认: 50)')
    
    args = parser.parse_args()
    
    print(f"🚀 启动优化版真实高耦合对LLM分类器")
    print(f"📋 配置参数:")
    print(f"   样本大小: {args.sample_size}")
    print(f"   模型: {args.model}")
    print(f"   批处理大小: {args.batch_size}")
    print(f"   检查点间隔: {args.checkpoint_interval}")
    
    # 显存预估
    memory_estimates = {
        "Qwen/Qwen2.5-7B-Instruct": "18-21GB",
        "Qwen/Qwen2.5-14B-Instruct": "28-32GB", 
        "Qwen/Qwen2.5-32B-Instruct": "40-45GB"
    }
    
    print(f"   💾 预估显存需求: {memory_estimates.get(args.model, '未知')}")
    
    classifier = OptimizedHighCouplingLLMClassifier(args.model)
    
    # 运行优化的分类
    results, analysis = classifier.run_optimized_classification(
        sample_size=args.sample_size
    )
    
    print(f"\n🎯 分类结果已保存，可用于指导Phase 2攻击实验！")
    
    # 显示最推荐的攻击目标
    if results:
        category_counts = Counter(results.values())
        top_3_categories = category_counts.most_common(3)
        print(f"\n🏆 推荐的攻击目标 (Top 3):")
        for i, (category, count) in enumerate(top_3_categories, 1):
            percentage = count / len(results) * 100
            print(f"   {i}. {category}: {count}个样本 ({percentage:.1f}%)")
        
        # 根据模型大小给出建议
        if "14B" in args.model:
            print(f"\n✨ 14B模型优势: 更精确的语义分类，建议优先选择前2个类别进行攻击实验")
        elif "7B" in args.model:
            print(f"\n💡 7B模型建议: 如需更高精度，考虑升级到14B模型")

if __name__ == "__main__":
    main() 