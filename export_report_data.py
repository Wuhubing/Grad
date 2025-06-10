#!/usr/bin/env python3
"""
导出报告数据脚本
将实验结果整理成适合学术报告的格式
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
import os

class ReportDataExporter:
    """报告数据导出器"""
    
    def __init__(self, experiment_results_file: str):
        self.experiment_file = experiment_results_file
        with open(experiment_results_file, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        print(f"📊 加载实验数据: {experiment_results_file}")
        print(f"🔍 实验总数: {len(self.raw_data.get('experiments', []))}")
    
    def extract_summary_statistics(self) -> Dict:
        """提取摘要统计"""
        analysis = self.raw_data.get('analysis', {})
        
        summary = {
            "实验概述": {
                "总实验数": analysis.get('overall_statistics', {}).get('total_experiments', 0),
                "成功率": f"{analysis.get('overall_statistics', {}).get('success_rate', 0):.1%}",
                "实验日期": self.raw_data.get('experiment_metadata', {}).get('timestamp', ''),
                "使用模型": self.raw_data.get('experiment_metadata', {}).get('model', '')
            },
            "主要发现": {
                "高耦合组平均涟漪效应": analysis.get('by_coupling_type', {}).get('high_coupling', {}).get('mean_ripple', 0),
                "低耦合组平均涟漪效应": analysis.get('by_coupling_type', {}).get('low_coupling', {}).get('mean_ripple', 0),
                "总体改善倍数": 0
            },
            "按编辑类型分析": {}
        }
        
        # 计算改善倍数
        high_mean = summary["主要发现"]["高耦合组平均涟漪效应"]
        low_mean = summary["主要发现"]["低耦合组平均涟漪效应"]
        if low_mean > 0:
            summary["主要发现"]["总体改善倍数"] = high_mean / low_mean
        
        # 按编辑类型统计
        for edit_type, data in analysis.get('by_edit_type', {}).items():
            if 'improvement_percentage' in data:
                summary["按编辑类型分析"][edit_type] = {
                    "高耦合组涟漪效应": data['high_coupling']['mean_ripple'],
                    "低耦合组涟漪效应": data['low_coupling']['mean_ripple'],
                    "改善百分比": data['improvement_percentage']
                }
        
        return summary
    
    def extract_detailed_experiments(self) -> List[Dict]:
        """提取详细实验数据"""
        experiments = self.raw_data.get('experiments', [])
        detailed_data = []
        
        for exp in experiments:
            if exp.get('edit_success', False):
                detailed_exp = {
                    "实验ID": exp.get('experiment_id', ''),
                    "耦合类型": "高耦合" if exp.get('pair_type') == 'high_coupling' else "低耦合",
                    "耦合强度": exp.get('coupling_strength', 0),
                    "编辑类型": exp.get('edit_type', ''),
                    "编辑强度": exp.get('edit_strength', 0),
                    "源知识片段": {
                        "ID": exp.get('source_piece', {}).get('piece_id', ''),
                        "问题": exp.get('source_piece', {}).get('question', ''),
                        "答案": exp.get('source_piece', {}).get('answer', '')
                    },
                    "目标知识片段": {
                        "ID": exp.get('target_piece', {}).get('piece_id', ''),
                        "问题": exp.get('target_piece', {}).get('question', ''),
                        "答案": exp.get('target_piece', {}).get('answer', '')
                    },
                    "基线测量": {
                        "目标logP": exp.get('baseline', {}).get('target_metrics', {}).get('first_token_logp', 0),
                        "源生成答案": exp.get('baseline', {}).get('source_generated_answer', ''),
                        "目标生成答案": exp.get('baseline', {}).get('target_generated_answer', '')
                    },
                    "编辑后测量": {
                        "目标logP": exp.get('edited', {}).get('target_metrics', {}).get('first_token_logp', 0),
                        "源生成答案": exp.get('edited', {}).get('source_generated_answer', ''),
                        "目标生成答案": exp.get('edited', {}).get('target_generated_answer', '')
                    },
                    "涟漪效应": {
                        "主要强度": exp.get('ripple_effect', {}).get('main_ripple_strength', 0),
                        "logP变化": exp.get('ripple_effect', {}).get('detailed_ripples', {}).get('first_token_logp_delta', 0),
                        "计算公式": exp.get('ripple_effect', {}).get('calculation_details', {}).get('formula', '')
                    },
                    "编辑详情": exp.get('edit_details', {})
                }
                detailed_data.append(detailed_exp)
        
        return detailed_data
    
    def create_comparison_tables(self) -> Dict:
        """创建对比表格数据"""
        experiments = self.raw_data.get('experiments', [])
        
        # 按编辑类型和耦合类型分组
        tables = {
            "按编辑类型对比": {},
            "按编辑强度对比": {},
            "典型案例对比": []
        }
        
        # 按编辑类型对比
        for edit_type in ["answer_based", "suppression", "random_control"]:
            high_exp = [exp for exp in experiments 
                       if exp.get('edit_type') == edit_type and exp.get('pair_type') == 'high_coupling' and exp.get('edit_success')]
            low_exp = [exp for exp in experiments 
                      if exp.get('edit_type') == edit_type and exp.get('pair_type') == 'low_coupling' and exp.get('edit_success')]
            
            if high_exp and low_exp:
                high_ripples = [exp['ripple_effect']['main_ripple_strength'] for exp in high_exp]
                low_ripples = [exp['ripple_effect']['main_ripple_strength'] for exp in low_exp]
                
                tables["按编辑类型对比"][edit_type] = {
                    "高耦合组": {
                        "实验数量": len(high_exp),
                        "平均涟漪效应": np.mean(high_ripples),
                        "标准差": np.std(high_ripples),
                        "最小值": np.min(high_ripples),
                        "最大值": np.max(high_ripples)
                    },
                    "低耦合组": {
                        "实验数量": len(low_exp),
                        "平均涟漪效应": np.mean(low_ripples),
                        "标准差": np.std(low_ripples),
                        "最小值": np.min(low_ripples),
                        "最大值": np.max(low_ripples)
                    },
                    "效应比较": {
                        "高/低比值": np.mean(high_ripples) / np.mean(low_ripples) if np.mean(low_ripples) > 0 else float('inf'),
                        "改善百分比": ((np.mean(high_ripples) - np.mean(low_ripples)) / np.mean(low_ripples) * 100) if np.mean(low_ripples) > 0 else 0
                    }
                }
        
        # 按编辑强度对比
        for strength in [0.001, 0.002]:
            high_exp = [exp for exp in experiments 
                       if exp.get('edit_strength') == strength and exp.get('pair_type') == 'high_coupling' and exp.get('edit_success')]
            low_exp = [exp for exp in experiments 
                      if exp.get('edit_strength') == strength and exp.get('pair_type') == 'low_coupling' and exp.get('edit_success')]
            
            if high_exp and low_exp:
                high_ripples = [exp['ripple_effect']['main_ripple_strength'] for exp in high_exp]
                low_ripples = [exp['ripple_effect']['main_ripple_strength'] for exp in low_exp]
                
                tables["按编辑强度对比"][f"强度_{strength}"] = {
                    "高耦合组平均": np.mean(high_ripples),
                    "低耦合组平均": np.mean(low_ripples),
                    "改善倍数": np.mean(high_ripples) / np.mean(low_ripples) if np.mean(low_ripples) > 0 else float('inf')
                }
        
        # 典型案例
        # 选择效应最强和最弱的案例
        successful_exp = [exp for exp in experiments if exp.get('edit_success') and exp.get('edit_type') != 'random_control']
        if successful_exp:
            # 按涟漪效应排序
            successful_exp.sort(key=lambda x: x['ripple_effect']['main_ripple_strength'], reverse=True)
            
            # 最强效应案例
            if successful_exp:
                strongest = successful_exp[0]
                tables["典型案例对比"].append({
                    "案例类型": "最强涟漪效应",
                    "实验ID": strongest.get('experiment_id'),
                    "耦合类型": strongest.get('pair_type'),
                    "耦合强度": strongest.get('coupling_strength'),
                    "编辑类型": strongest.get('edit_type'),
                    "涟漪效应强度": strongest['ripple_effect']['main_ripple_strength'],
                    "源知识": strongest['source_piece']['answer'],
                    "目标知识": strongest['target_piece']['answer'],
                    "logP变化": strongest['ripple_effect']['detailed_ripples']['first_token_logp_delta']
                })
            
            # 最弱效应案例
            if len(successful_exp) > 1:
                weakest = successful_exp[-1]
                tables["典型案例对比"].append({
                    "案例类型": "最弱涟漪效应",
                    "实验ID": weakest.get('experiment_id'),
                    "耦合类型": weakest.get('pair_type'),
                    "耦合强度": weakest.get('coupling_strength'),
                    "编辑类型": weakest.get('edit_type'),
                    "涟漪效应强度": weakest['ripple_effect']['main_ripple_strength'],
                    "源知识": weakest['source_piece']['answer'],
                    "目标知识": weakest['target_piece']['answer'],
                    "logP变化": weakest['ripple_effect']['detailed_ripples']['first_token_logp_delta']
                })
        
        return tables
    
    def create_chart_data(self) -> Dict:
        """创建图表数据"""
        experiments = self.raw_data.get('experiments', [])
        successful_exp = [exp for exp in experiments if exp.get('edit_success')]
        
        chart_data = {
            "散点图数据": {
                "x轴_耦合强度": [],
                "y轴_涟漪效应": [],
                "颜色_编辑类型": [],
                "大小_编辑强度": []
            },
            "柱状图数据": {
                "类别": ["高耦合组", "低耦合组"],
                "answer_based": [],
                "suppression": [],
                "random_control": []
            },
            "箱线图数据": {
                "高耦合_answer_based": [],
                "高耦合_suppression": [],
                "低耦合_answer_based": [],
                "低耦合_suppression": []
            }
        }
        
        # 散点图数据
        for exp in successful_exp:
            chart_data["散点图数据"]["x轴_耦合强度"].append(exp.get('coupling_strength', 0))
            chart_data["散点图数据"]["y轴_涟漪效应"].append(exp['ripple_effect']['main_ripple_strength'])
            chart_data["散点图数据"]["颜色_编辑类型"].append(exp.get('edit_type', ''))
            chart_data["散点图数据"]["大小_编辑强度"].append(exp.get('edit_strength', 0) * 1000)  # 放大便于显示
        
        # 柱状图数据 - 按编辑类型分组
        for edit_type in ["answer_based", "suppression", "random_control"]:
            high_ripples = [exp['ripple_effect']['main_ripple_strength'] 
                           for exp in successful_exp 
                           if exp.get('edit_type') == edit_type and exp.get('pair_type') == 'high_coupling']
            low_ripples = [exp['ripple_effect']['main_ripple_strength'] 
                          for exp in successful_exp 
                          if exp.get('edit_type') == edit_type and exp.get('pair_type') == 'low_coupling']
            
            chart_data["柱状图数据"][edit_type] = [
                np.mean(high_ripples) if high_ripples else 0,
                np.mean(low_ripples) if low_ripples else 0
            ]
        
        # 箱线图数据
        for coupling_type in ['high_coupling', 'low_coupling']:
            for edit_type in ['answer_based', 'suppression']:
                key = f"{'高耦合' if coupling_type == 'high_coupling' else '低耦合'}_{edit_type}"
                ripples = [exp['ripple_effect']['main_ripple_strength'] 
                          for exp in successful_exp 
                          if exp.get('edit_type') == edit_type and exp.get('pair_type') == coupling_type]
                chart_data["箱线图数据"][key] = ripples
        
        return chart_data
    
    def create_methodology_section(self) -> Dict:
        """创建方法论部分数据"""
        metadata = self.raw_data.get('experiment_metadata', {})
        
        methodology = {
            "实验设计": {
                "模型": metadata.get('model', ''),
                "编辑类型": metadata.get('edit_types', []),
                "编辑强度": metadata.get('edit_strengths', []),
                "高耦合知识片段对数": metadata.get('high_coupling_pairs', 0),
                "低耦合知识片段对数": metadata.get('low_coupling_pairs', 0),
                "总实验数": metadata.get('total_experiments', 0)
            },
            "测量指标": {
                "主要指标": "第一token的log probability变化",
                "辅助指标": ["总log probability", "平均log probability"],
                "涟漪效应计算": "目标知识片段logP的绝对变化量"
            },
            "实验控制": {
                "权重隔离": "每次实验后完全恢复原始模型权重",
                "基线测量": "编辑前后的logP对比",
                "对照组": "随机编辑控制实验",
                "编辑目标": "基于真实答案token而非固定token"
            },
            "数据来源": {
                "知识库": "HotpotQA数据集",
                "耦合计算": "基于梯度相似度(GradSim)",
                "片段筛选": "高耦合组>0.7，低耦合组-0.05~0.05"
            }
        }
        
        return methodology
    
    def export_report_data(self, output_filename: str = None) -> str:
        """导出完整的报告数据"""
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"report_data_{timestamp}.json"
        
        print(f"\n📋 正在生成报告数据...")
        
        report_data = {
            "报告元信息": {
                "生成时间": datetime.now().isoformat(),
                "数据来源": self.experiment_file,
                "报告版本": "v1.0"
            },
            "摘要统计": self.extract_summary_statistics(),
            "详细实验数据": self.extract_detailed_experiments(),
            "对比表格": self.create_comparison_tables(),
            "图表数据": self.create_chart_data(),
            "方法论": self.create_methodology_section(),
            "原始数据引用": {
                "实验文件": self.experiment_file,
                "分析结果": self.raw_data.get('analysis', {})
            }
        }
        
        # 保存数据
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 报告数据已保存: {output_filename}")
        
        # 显示摘要
        summary = report_data["摘要统计"]
        print(f"\n📊 数据摘要:")
        print(f"   实验总数: {summary['实验概述']['总实验数']}")
        print(f"   成功率: {summary['实验概述']['成功率']}")
        print(f"   高耦合组平均涟漪效应: {summary['主要发现']['高耦合组平均涟漪效应']:.4f}")
        print(f"   低耦合组平均涟漪效应: {summary['主要发现']['低耦合组平均涟漪效应']:.4f}")
        print(f"   改善倍数: {summary['主要发现']['总体改善倍数']:.1f}x")
        
        print(f"\n📈 按编辑类型改善:")
        for edit_type, data in summary["按编辑类型分析"].items():
            print(f"   {edit_type}: {data['改善百分比']:.1f}%")
        
        return output_filename

def main():
    """主函数"""
    print("📄 启动报告数据导出器")
    
    # 查找最新的实验结果文件
    experiment_files = [f for f in os.listdir('.') if f.startswith('improved_experiment_results_') and f.endswith('.json')]
    
    if not experiment_files:
        print("❌ 未找到实验结果文件，请先运行 improved_knowledge_editor.py")
        return None
    
    # 使用最新的文件
    latest_file = sorted(experiment_files)[-1]
    print(f"📁 使用实验文件: {latest_file}")
    
    # 创建导出器并导出数据
    exporter = ReportDataExporter(latest_file)
    output_file = exporter.export_report_data()
    
    print(f"\n🎯 报告数据导出完成!")
    print(f"📄 您现在可以使用 {output_file} 文件来写学术报告")
    print(f"📊 该文件包含了所有必要的统计数据、图表数据和详细实验记录")
    
    return output_file

if __name__ == "__main__":
    main() 