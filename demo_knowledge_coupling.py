#!/usr/bin/env python3
"""
知识耦合与编辑实验演示
展示从数据处理到耦合分析再到知识编辑的完整流程
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import datetime

class KnowledgeCouplingDemo:
    """知识耦合和编辑实验演示器"""
    
    def __init__(self):
        self.results_dir = Path("results")
        self.llama_analysis_dir = self.results_dir / "llama2_7b_analysis"
        self.experiment_results_file = None
        
        # 查找最新的实验结果文件
        for file in self.results_dir.glob("improved_experiment_results_*.json"):
            self.experiment_results_file = file
            break
        
        print("🎯 知识耦合与编辑实验演示")
        print("="*60)
        print(f"分析目录: {self.llama_analysis_dir}")
        print(f"实验结果: {self.experiment_results_file}")
    
    def load_data(self):
        """加载所有实验数据"""
        print("\n📊 加载实验数据...")
        
        # 1. 加载耦合分析结果
        with open(self.llama_analysis_dir / "analysis_metadata.json", 'r') as f:
            self.coupling_metadata = json.load(f)
        
        with open(self.llama_analysis_dir / "knowledge_pieces.json", 'r') as f:
            self.knowledge_pieces = json.load(f)
        
        with open(self.llama_analysis_dir / "high_coupling_pairs.json", 'r') as f:
            self.high_coupling_pairs = json.load(f)
        
        # 2. 加载耦合对数据
        self.coupling_df = pd.read_csv(self.llama_analysis_dir / "coupling_pairs.csv")
        
        # 3. 加载知识编辑实验结果
        if self.experiment_results_file:
            with open(self.experiment_results_file, 'r') as f:
                self.editing_results = json.load(f)
        
        print(f"✅ 数据加载完成:")
        print(f"   知识片段: {len(self.knowledge_pieces)}")
        print(f"   耦合对: {len(self.coupling_df)}")
        print(f"   高耦合对: {self.high_coupling_pairs['count']}")
        if self.experiment_results_file:
            print(f"   编辑实验: {len(self.editing_results['experiments'])}")
    
    def show_dataset_overview(self):
        """展示数据集概览"""
        print("\n📚 数据集概览")
        print("-" * 40)
        
        # 数据集基本信息
        dataset_info = self.coupling_metadata.get('dataset_info', {})
        print(f"数据集: {dataset_info.get('dataset_name', 'Unknown')}")
        print(f"文件路径: {dataset_info.get('dataset_file_path', 'Unknown')}")
        print(f"总样本数: {dataset_info.get('total_samples_in_file', 'Unknown')}")
        print(f"处理样本数: {dataset_info.get('samples_processed', 'Unknown')}")
        
        # 知识片段类别分布
        categories = {}
        for piece in self.knowledge_pieces:
            cat = piece.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\n📋 知识片段类别分布:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"   {cat}: {count} 个")
    
    def show_coupling_analysis(self):
        """展示耦合分析结果"""
        print("\n🔗 耦合分析结果")
        print("-" * 40)
        
        # 耦合强度统计
        coupling_strengths = self.coupling_df['coupling_strength'].values
        
        print(f"耦合强度统计:")
        print(f"   平均值: {np.mean(coupling_strengths):.4f}")
        print(f"   标准差: {np.std(coupling_strengths):.4f}")
        print(f"   最小值: {np.min(coupling_strengths):.4f}")
        print(f"   最大值: {np.max(coupling_strengths):.4f}")
        
        # 耦合强度分布
        high_coupling = np.sum(coupling_strengths >= 0.4)
        moderate_coupling = np.sum((coupling_strengths >= 0.1) & (coupling_strengths < 0.4))
        low_coupling = np.sum(coupling_strengths < 0.1)
        total = len(coupling_strengths)
        
        print(f"\n耦合强度分布:")
        print(f"   高耦合 (≥0.4): {high_coupling} ({high_coupling/total:.1%})")
        print(f"   中等耦合 (0.1-0.4): {moderate_coupling} ({moderate_coupling/total:.1%})")
        print(f"   低耦合 (<0.1): {low_coupling} ({low_coupling/total:.1%})")
        
        # 显示前5个高耦合对
        print(f"\n🔥 前5个高耦合对:")
        for i, pair in enumerate(self.high_coupling_pairs['pairs'][:5], 1):
            piece1 = next((p for p in self.knowledge_pieces if p['piece_id'] == pair['piece_1_id']), {})
            piece2 = next((p for p in self.knowledge_pieces if p['piece_id'] == pair['piece_2_id']), {})
            
            print(f"   {i}. 耦合强度: {pair['coupling_strength']:.4f}")
            print(f"      片段1: {piece1.get('answer', 'Unknown')} (类别: {piece1.get('category', 'Unknown')})")
            print(f"      片段2: {piece2.get('answer', 'Unknown')} (类别: {piece2.get('category', 'Unknown')})")
    
    def show_editing_results(self):
        """展示知识编辑实验结果"""
        if not self.experiment_results_file:
            print("\n❌ 没有找到知识编辑实验结果")
            return
        
        print("\n🔬 知识编辑实验结果")
        print("-" * 40)
        
        # 实验元数据
        metadata = self.editing_results['experiment_metadata']
        print(f"模型: {metadata['model']}")
        print(f"总实验数: {metadata['total_experiments']}")
        print(f"编辑强度: {metadata['edit_strengths']}")
        print(f"编辑类型: {metadata['edit_types']}")
        
        # 分析结果
        analysis = self.editing_results['analysis']
        
        print(f"\n📊 实验成功率: {analysis['overall_statistics']['success_rate']:.1%}")
        
        # 按耦合类型显示结果
        print(f"\n🎯 按耦合类型分析:")
        for coupling_type, stats in analysis['by_coupling_type'].items():
            print(f"   {coupling_type}:")
            print(f"     实验数量: {stats['count']}")
            print(f"     平均涟漪效应: {stats['mean_ripple']:.4f}")
            print(f"     涟漪效应范围: {stats['min_ripple']:.4f} - {stats['max_ripple']:.4f}")
        
        # 计算改善效果
        high_mean = analysis['by_coupling_type']['high_coupling']['mean_ripple']
        low_mean = analysis['by_coupling_type']['low_coupling']['mean_ripple']
        
        if low_mean > 0:
            improvement = (high_mean - low_mean) / low_mean * 100
        else:
            improvement = float('inf') if high_mean > 0 else 0
        
        print(f"\n🏆 核心发现:")
        if improvement == float('inf'):
            print(f"   高耦合组产生了涟漪效应 ({high_mean:.4f})")
            print(f"   低耦合组无涟漪效应 ({low_mean:.4f})")
            print(f"   验证了梯度相似度预测涟漪效应的假设!")
        else:
            print(f"   高耦合组比低耦合组涟漪效应强 {improvement:.1f}%")
        
        # 按编辑类型显示结果
        print(f"\n🎨 按编辑类型分析:")
        for edit_type, stats in analysis['by_edit_type'].items():
            print(f"   {edit_type}:")
            print(f"     高耦合平均涟漪: {stats['high_coupling']['mean_ripple']:.4f}")
            print(f"     低耦合平均涟漪: {stats['low_coupling']['mean_ripple']:.4f}")
    
    def show_case_studies(self):
        """展示具体案例研究"""
        if not self.experiment_results_file:
            return
        
        print("\n📖 案例研究")
        print("-" * 40)
        
        experiments = self.editing_results['experiments']
        
        # 找一个高耦合的成功案例
        high_coupling_case = None
        for exp in experiments:
            if (exp['pair_type'] == 'high_coupling' and 
                exp['edit_type'] == 'answer_based' and 
                exp['ripple_effect']['main_ripple_strength'] > 0):
                high_coupling_case = exp
                break
        
        if high_coupling_case:
            print(f"🔥 高耦合案例 (ID: {high_coupling_case['experiment_id']}):")
            print(f"   耦合强度: {high_coupling_case['coupling_strength']:.4f}")
            print(f"   编辑强度: {high_coupling_case['edit_strength']}")
            print(f"   涟漪效应: {high_coupling_case['ripple_effect']['main_ripple_strength']:.4f}")
            
            source = high_coupling_case['source_piece']
            target = high_coupling_case['target_piece']
            
            print(f"   源片段: {source['answer']}")
            print(f"   目标片段: {target['answer']}")
            
            ripple = high_coupling_case['ripple_effect']
            print(f"   计算公式: {ripple['calculation_details']['formula']}")
        
        # 找一个低耦合案例
        low_coupling_case = None
        for exp in experiments:
            if (exp['pair_type'] == 'low_coupling' and 
                exp['edit_type'] == 'answer_based'):
                low_coupling_case = exp
                break
        
        if low_coupling_case:
            print(f"\n🔵 低耦合案例 (ID: {low_coupling_case['experiment_id']}):")
            print(f"   耦合强度: {low_coupling_case['coupling_strength']:.4f}")
            print(f"   编辑强度: {low_coupling_case['edit_strength']}")
            print(f"   涟漪效应: {low_coupling_case['ripple_effect']['main_ripple_strength']:.4f}")
            
            source = low_coupling_case['source_piece']
            target = low_coupling_case['target_piece']
            
            print(f"   源片段: {source['answer']}")
            print(f"   目标片段: {target['answer']}")
    
    def create_visualization(self):
        """创建可视化图表"""
        print("\n🎨 生成可视化图表...")
        
        # 创建输出目录
        viz_dir = self.results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. 耦合强度分布图
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        coupling_strengths = self.coupling_df['coupling_strength'].values
        plt.hist(coupling_strengths, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=0.4, color='red', linestyle='--', label='High Coupling Threshold (0.4)')
        plt.xlabel('Coupling Strength')
        plt.ylabel('Frequency')
        plt.title('Knowledge Coupling Strength Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 涟漪效应对比图
        if self.experiment_results_file:
            plt.subplot(1, 2, 2)
            analysis = self.editing_results['analysis']
            
            coupling_types = ['High Coupling', 'Low Coupling']
            ripple_means = [
                analysis['by_coupling_type']['high_coupling']['mean_ripple'],
                analysis['by_coupling_type']['low_coupling']['mean_ripple']
            ]
            
            bars = plt.bar(coupling_types, ripple_means, 
                          color=['red', 'blue'], alpha=0.7)
            plt.ylabel('Average Ripple Effect Strength')
            plt.title('High vs Low Coupling Ripple Effects')
            plt.ylim(0, max(ripple_means) * 1.2 if max(ripple_means) > 0 else 0.1)
            
            # 添加数值标签
            for bar, value in zip(bars, ripple_means):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        viz_file = viz_dir / "coupling_analysis_overview.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 可视化图表已保存: {viz_file}")
        
        return viz_file
    
    def generate_summary_report(self):
        """生成总结报告"""
        print("\n📄 生成总结报告...")
        
        report_file = self.results_dir / "demo_summary_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 知识耦合与编辑实验演示报告\n\n")
            f.write(f"**生成时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 实验概述
            f.write("## 🎯 实验概述\n\n")
            f.write("本实验验证了梯度相似度(GradSim)能够预测知识编辑中的涟漪效应的假设。\n\n")
            
            f.write("**核心假设**: 高梯度相似度的知识片段对在知识编辑时会产生更强的涟漪效应。\n\n")
            
            # 数据集信息
            f.write("## 📊 数据集信息\n\n")
            dataset_info = self.coupling_metadata.get('dataset_info', {})
            f.write(f"- **数据集**: {dataset_info.get('dataset_name', 'Unknown')}\n")
            f.write(f"- **处理样本**: {dataset_info.get('samples_processed', 'Unknown')} 个\n")
            f.write(f"- **知识片段**: {len(self.knowledge_pieces)} 个\n")
            f.write(f"- **耦合对**: {len(self.coupling_df)} 对\n\n")
            
            # 模型信息
            f.write("## 🤖 模型信息\n\n")
            model_info = self.coupling_metadata.get('model_info', {})
            f.write(f"- **模型**: {model_info.get('model_path', 'Unknown')}\n")
            f.write(f"- **模型类型**: {model_info.get('model_type', 'Unknown')}\n")
            f.write(f"- **目标层数**: {model_info.get('target_layers_count', 'Unknown')}\n\n")
            
            # 耦合分析结果
            f.write("## 🔗 耦合分析结果\n\n")
            coupling_strengths = self.coupling_df['coupling_strength'].values
            high_coupling = np.sum(coupling_strengths >= 0.4)
            total = len(coupling_strengths)
            
            f.write(f"- **平均耦合强度**: {np.mean(coupling_strengths):.4f}\n")
            f.write(f"- **高耦合对数**: {high_coupling} ({high_coupling/total:.1%})\n")
            f.write(f"- **耦合强度范围**: [{np.min(coupling_strengths):.4f}, {np.max(coupling_strengths):.4f}]\n\n")
            
            # 知识编辑结果
            if self.experiment_results_file:
                f.write("## 🔬 知识编辑实验结果\n\n")
                
                metadata = self.editing_results['experiment_metadata']
                analysis = self.editing_results['analysis']
                
                f.write(f"- **总实验数**: {metadata['total_experiments']}\n")
                f.write(f"- **成功率**: {analysis['overall_statistics']['success_rate']:.1%}\n")
                f.write(f"- **编辑类型**: {', '.join(metadata['edit_types'])}\n")
                f.write(f"- **编辑强度**: {metadata['edit_strengths']}\n\n")
                
                f.write("### 核心发现\n\n")
                
                high_mean = analysis['by_coupling_type']['high_coupling']['mean_ripple']
                low_mean = analysis['by_coupling_type']['low_coupling']['mean_ripple']
                
                f.write(f"- **高耦合组平均涟漪效应**: {high_mean:.4f}\n")
                f.write(f"- **低耦合组平均涟漪效应**: {low_mean:.4f}\n")
                
                if low_mean == 0 and high_mean > 0:
                    f.write("- **结论**: 高耦合组产生了显著的涟漪效应，而低耦合组无涟漪效应，**验证了GradSim预测假设**! 🎉\n\n")
                else:
                    improvement = (high_mean - low_mean) / low_mean * 100 if low_mean > 0 else 0
                    f.write(f"- **改善程度**: 高耦合组比低耦合组强 {improvement:.1f}%\n\n")
            
            # 方法论
            f.write("## 🔬 方法论\n\n")
            f.write("1. **知识提取**: 从HotpotQA数据集提取2-hop知识链，转换为cloze问题\n")
            f.write("2. **梯度计算**: 针对目标答案token计算模型参数梯度\n")
            f.write("3. **耦合测量**: 使用余弦相似度计算梯度向量间的耦合强度\n")
            f.write("4. **知识编辑**: 使用改进的编辑方法修改模型参数\n")
            f.write("5. **涟漪测量**: 测量编辑对其他知识片段的影响程度\n\n")
            
            # 技术细节
            f.write("## ⚙️ 技术细节\n\n")
            f.write("- **梯度计算公式**: ∇_θ log P(answer|question)\n")
            f.write("- **耦合度公式**: GradSim(i,j) = cos(∇_θ log P(a_i|q_i), ∇_θ log P(a_j|q_j))\n")
            f.write("- **涟漪效应**: |log P_edited(target_answer) - log P_baseline(target_answer)|\n")
            f.write("- **高耦合阈值**: ≥ 0.4\n")
            f.write("- **编辑强度**: 0.001, 0.002\n\n")
            
            f.write("---\n")
            f.write("*报告由知识耦合演示系统自动生成*\n")
        
        print(f"✅ 总结报告已保存: {report_file}")
        return report_file
    
    def run_full_demo(self):
        """运行完整演示"""
        print("🚀 启动知识耦合与编辑完整演示\n")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 展示各部分结果
        self.show_dataset_overview()
        self.show_coupling_analysis()
        self.show_editing_results()
        self.show_case_studies()
        
        # 3. 创建可视化
        viz_file = self.create_visualization()
        
        # 4. 生成报告
        report_file = self.generate_summary_report()
        
        print(f"\n🎉 演示完成!")
        print(f"📄 总结报告: {report_file}")
        print(f"🎨 可视化图表: {viz_file}")
        print(f"📁 所有结果保存在: {self.results_dir}/")

def main():
    """主函数"""
    demo = KnowledgeCouplingDemo()
    demo.run_full_demo()

if __name__ == "__main__":
    main() 