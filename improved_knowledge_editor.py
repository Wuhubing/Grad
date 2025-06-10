#!/usr/bin/env python3
"""
改进的知识编辑器
实现更精细、更有效的知识编辑方法
"""

import json
import torch
import torch.nn.functional as F
from transformers import LlamaTokenizer, LlamaForCausalLM
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import datetime
from copy import deepcopy

class ImprovedKnowledgeEditor:
    """改进的知识编辑器"""
    
    def __init__(self, model_path: str = "meta-llama/Llama-2-7b-hf", hf_token: str = None, coupling_results_dir: str = "results/llama2_7b_analysis"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = hf_token
        self.coupling_results_dir = coupling_results_dir
        
        print(f"🔬 初始化改进的知识编辑器")
        print(f"模型: {model_path}")
        print(f"设备: {self.device}")
        print(f"耦合结果目录: {coupling_results_dir}")
        
        self._load_model()
        self._load_data(coupling_results_dir)
        
        # 保存原始权重
        self.original_lm_head = self.model.lm_head.weight.data.clone()
        
        print(f"✅ 改进编辑器初始化完成!")
    
    def _load_model(self):
        """加载模型"""
        print("📦 正在加载LLaMA模型...")
        
        if self.hf_token:
            self.tokenizer = LlamaTokenizer.from_pretrained(
                self.model_path, 
                token=self.hf_token
            )
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                token=self.hf_token
            )
        else:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        print("✅ 模型加载完成!")
    
    def _load_data(self, coupling_results_dir: str = "results/llama2_7b_analysis"):
        """加载实验数据"""
        print("📊 正在加载实验数据...")
        
        # 加载高耦合对
        high_coupling_file = f'{coupling_results_dir}/high_coupling_pairs.json'
        with open(high_coupling_file, 'r', encoding='utf-8') as f:
            self.high_coupling_data = json.load(f)
        
        # 加载知识片段
        knowledge_pieces_file = f'{coupling_results_dir}/knowledge_pieces.json'
        with open(knowledge_pieces_file, 'r', encoding='utf-8') as f:
            knowledge_pieces = json.load(f)
        
        self.piece_map = {piece['piece_id']: piece for piece in knowledge_pieces}
        
        # 加载耦合数据
        coupling_data = []
        coupling_csv_file = f'{coupling_results_dir}/coupling_pairs.csv'
        with open(coupling_csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:]:
                try:
                    parts = []
                    current_part = ""
                    in_quotes = False
                    
                    for char in line.strip():
                        if char == '"':
                            in_quotes = not in_quotes
                        elif char == ',' and not in_quotes:
                            parts.append(current_part.strip())
                            current_part = ""
                        else:
                            current_part += char
                    
                    parts.append(current_part.strip())
                    
                    if len(parts) >= 6:
                        coupling_data.append({
                            'piece_1_id': parts[0],
                            'piece_2_id': parts[1], 
                            'coupling_strength': float(parts[2]),
                            'is_same_hotpot': parts[3].lower() == 'true'
                        })
                except:
                    continue
        
        self.coupling_df = pd.DataFrame(coupling_data)
        print(f"✅ 数据加载完成!")
        print(f"   高耦合对: {len(self.high_coupling_data['pairs'])}")
        print(f"   知识片段: {len(self.piece_map)}")
        print(f"   耦合对: {len(self.coupling_df)}")
    
    def get_answer_tokens(self, answer: str) -> List[int]:
        """获取答案的token序列"""
        tokens = self.tokenizer(answer, add_special_tokens=False)['input_ids']
        return tokens if tokens else []
    
    def compute_comprehensive_logp(self, question: str, answer: str) -> Dict:
        """计算全面的log probability指标"""
        try:
            prompt = f"{question} Answer:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            answer_tokens = self.get_answer_tokens(answer)
            if not answer_tokens:
                return {"first_token_logp": float('-inf'), "total_logp": float('-inf'), "avg_logp": float('-inf')}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]  # 最后一个位置的logits
                log_probs = F.log_softmax(logits, dim=-1)
                
                # 第一个token的logP
                first_token_logp = log_probs[answer_tokens[0]].item()
                
                # 如果答案有多个token，计算总logP和平均logP
                if len(answer_tokens) > 1:
                    # 生成后续tokens来计算完整序列logP
                    total_logp = first_token_logp
                    current_input = inputs['input_ids']
                    
                    for token_id in answer_tokens[:3]:  # 限制计算前3个token避免过长
                        current_input = torch.cat([current_input, torch.tensor([[token_id]]).to(self.device)], dim=1)
                        if current_input.shape[1] < inputs['input_ids'].shape[1] + len(answer_tokens):
                            next_outputs = self.model(current_input)
                            next_logits = next_outputs.logits[0, -1, :]
                            next_log_probs = F.log_softmax(next_logits, dim=-1)
                            if len(answer_tokens) > answer_tokens.index(token_id) + 1:
                                next_token_id = answer_tokens[answer_tokens.index(token_id) + 1]
                                total_logp += next_log_probs[next_token_id].item()
                    
                    avg_logp = total_logp / min(len(answer_tokens), 3)
                else:
                    total_logp = first_token_logp
                    avg_logp = first_token_logp
                
                return {
                    "first_token_logp": first_token_logp,
                    "total_logp": total_logp,
                    "avg_logp": avg_logp,
                    "answer_tokens": answer_tokens[:3]  # 记录前3个token
                }
                
        except Exception as e:
            print(f"计算logP时出错: {e}")
            return {"first_token_logp": float('-inf'), "total_logp": float('-inf'), "avg_logp": float('-inf')}
    
    def generate_answer(self, question: str, max_length: int = 20) -> str:
        """生成问题的答案"""
        prompt = f"{question} Answer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=inputs['input_ids'].shape[1] + max_length,
                temperature=0.1,  # 降低temperature增加确定性
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return answer.strip()
    
    def improved_edit(self, source_question: str, target_answer: str, 
                     edit_strength: float = 0.001, edit_type: str = "answer_based") -> Tuple[bool, Dict]:
        """改进的知识编辑方法"""
        try:
            # 获取源问题的隐藏状态
            prompt = f"{source_question} Answer:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1][0, -1, :].clone()
            
            edit_info = {"edit_type": edit_type, "edit_strength": edit_strength}
            
            if edit_type == "answer_based":
                # 基于目标答案的编辑
                target_tokens = self.get_answer_tokens(target_answer)
                if not target_tokens:
                    return False, {"error": "无效的目标答案"}
                
                # 选择第一个token作为主要编辑目标
                primary_token_id = target_tokens[0]
                edit_info["target_token_id"] = primary_token_id
                edit_info["target_token"] = self.tokenizer.decode([primary_token_id])
                
                # 计算编辑方向 - 更温和的方法
                edit_direction = last_hidden.to(self.model.lm_head.weight.dtype)
                edit_direction = edit_direction / edit_direction.norm() * edit_strength  # 归一化
                
                # 应用编辑
                self.model.lm_head.weight.data[primary_token_id] += edit_direction
                
            elif edit_type == "suppression":
                # 抑制型编辑 - 降低原答案的概率
                original_tokens = self.get_answer_tokens(target_answer)
                if not original_tokens:
                    return False, {"error": "无效的原答案"}
                
                primary_token_id = original_tokens[0]
                edit_info["suppressed_token_id"] = primary_token_id
                edit_info["suppressed_token"] = self.tokenizer.decode([primary_token_id])
                
                # 计算抑制方向
                edit_direction = -last_hidden.to(self.model.lm_head.weight.dtype)
                edit_direction = edit_direction / edit_direction.norm() * edit_strength
                
                self.model.lm_head.weight.data[primary_token_id] += edit_direction
                
            elif edit_type == "random_control":
                # 随机控制编辑 - 用于对照实验
                random_token_id = torch.randint(0, self.model.lm_head.weight.shape[0], (1,)).item()
                edit_info["random_token_id"] = random_token_id
                edit_info["random_token"] = self.tokenizer.decode([random_token_id])
                
                edit_direction = torch.randn_like(last_hidden).to(self.model.lm_head.weight.dtype)
                edit_direction = edit_direction / edit_direction.norm() * edit_strength
                
                self.model.lm_head.weight.data[random_token_id] += edit_direction
            
            return True, edit_info
            
        except Exception as e:
            print(f"编辑失败: {e}")
            return False, {"error": str(e)}
    
    def restore_weights(self):
        """恢复权重"""
        self.model.lm_head.weight.data = self.original_lm_head.clone()
    
    def run_improved_experiment(self, source_piece: Dict, target_piece: Dict, 
                              coupling_strength: float, pair_type: str, 
                              edit_strength: float = 0.001, edit_type: str = "answer_based",
                              experiment_id: str = None) -> Dict:
        """运行改进的实验"""
        
        print(f"\n🧪 运行改进实验: {experiment_id} ({pair_type}, 强度={edit_strength}, 类型={edit_type})")
        
        # 实验元信息
        experiment_result = {
            "experiment_id": experiment_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "pair_type": pair_type,
            "coupling_strength": coupling_strength,
            "edit_strength": edit_strength,
            "edit_type": edit_type,
            "source_piece": {
                "piece_id": source_piece.get('piece_id', 'unknown'),
                "question": source_piece['question'][:100] + "..." if len(source_piece['question']) > 100 else source_piece['question'],
                "answer": source_piece['answer']
            },
            "target_piece": {
                "piece_id": target_piece.get('piece_id', 'unknown'),
                "question": target_piece['question'][:100] + "..." if len(target_piece['question']) > 100 else target_piece['question'],
                "answer": target_piece['answer']
            }
        }
        
        # 基线测量
        baseline_source_metrics = self.compute_comprehensive_logp(source_piece['question'], source_piece['answer'])
        baseline_target_metrics = self.compute_comprehensive_logp(target_piece['question'], target_piece['answer'])
        baseline_source_answer = self.generate_answer(source_piece['question'])
        baseline_target_answer = self.generate_answer(target_piece['question'])
        
        experiment_result["baseline"] = {
            "source_metrics": baseline_source_metrics,
            "target_metrics": baseline_target_metrics,
            "source_generated_answer": baseline_source_answer,
            "target_generated_answer": baseline_target_answer
        }
        
        # 执行编辑
        edit_success, edit_info = self.improved_edit(
            source_piece['question'], 
            source_piece['answer'],  # 使用源答案作为编辑目标
            edit_strength, 
            edit_type
        )
        
        if not edit_success:
            experiment_result["edit_success"] = False
            experiment_result["error"] = edit_info.get("error", "未知错误")
            return experiment_result
        
        experiment_result["edit_success"] = True
        experiment_result["edit_details"] = edit_info
        
        # 编辑后测量
        edited_source_metrics = self.compute_comprehensive_logp(source_piece['question'], source_piece['answer'])
        edited_target_metrics = self.compute_comprehensive_logp(target_piece['question'], target_piece['answer'])
        edited_source_answer = self.generate_answer(source_piece['question'])
        edited_target_answer = self.generate_answer(target_piece['question'])
        
        experiment_result["edited"] = {
            "source_metrics": edited_source_metrics,
            "target_metrics": edited_target_metrics,
            "source_generated_answer": edited_source_answer,
            "target_generated_answer": edited_target_answer
        }
        
        # 计算多维度涟漪效应
        ripple_effects = {}
        
        for metric in ["first_token_logp", "total_logp", "avg_logp"]:
            baseline_val = baseline_target_metrics.get(metric, float('-inf'))
            edited_val = edited_target_metrics.get(metric, float('-inf'))
            
            if baseline_val != float('-inf') and edited_val != float('-inf'):
                delta = edited_val - baseline_val
                ripple_effects[f"{metric}_delta"] = delta
                ripple_effects[f"{metric}_ripple_strength"] = abs(delta)
            else:
                ripple_effects[f"{metric}_delta"] = 0
                ripple_effects[f"{metric}_ripple_strength"] = 0
        
        # 主要涟漪效应指标（使用第一个token的logP）
        main_ripple = ripple_effects.get("first_token_logp_ripple_strength", 0)
        
        experiment_result["ripple_effect"] = {
            "main_ripple_strength": main_ripple,
            "detailed_ripples": ripple_effects,
            "calculation_details": {
                "baseline_first_token_logp": baseline_target_metrics.get("first_token_logp", 0),
                "edited_first_token_logp": edited_target_metrics.get("first_token_logp", 0),
                "formula": f"|{edited_target_metrics.get('first_token_logp', 0):.4f} - ({baseline_target_metrics.get('first_token_logp', 0):.4f})| = {main_ripple:.4f}"
            }
        }
        
        # 恢复权重
        self.restore_weights()
        
        print(f"   ✅ 完成，主要涟漪效应: {main_ripple:.4f}")
        
        return experiment_result
    
    def run_comprehensive_experiments(self) -> Dict:
        """运行全面的实验套件"""
        
        print(f"\n🎯 开始运行改进的实验套件")
        print("="*80)
        
        all_results = {
            "experiment_metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "model": self.model_path,
                "device": str(self.device),
                "total_experiments": 24,  # 3种编辑类型 × 2种强度 × 4对知识片段
                "edit_strengths": [0.001, 0.002],
                "edit_types": ["answer_based", "suppression", "random_control"],
                "high_coupling_pairs": 2,
                "low_coupling_pairs": 2
            },
            "experiments": []
        }
        
        edit_strengths = [0.001, 0.002]
        edit_types = ["answer_based", "suppression", "random_control"]
        
        # 高耦合组实验
        print(f"\n🔥 高耦合组实验")
        high_coupling_pairs = self.high_coupling_data['pairs'][:2]
        
        for i, pair in enumerate(high_coupling_pairs, 1):
            source_piece = self.piece_map[pair['piece_1_id']].copy()
            target_piece = self.piece_map[pair['piece_2_id']].copy()
            
            source_piece['piece_id'] = pair['piece_1_id']
            target_piece['piece_id'] = pair['piece_2_id']
            
            for j, edit_strength in enumerate(edit_strengths, 1):
                for k, edit_type in enumerate(edit_types, 1):
                    experiment_id = f"H{i}-{j}-{k}"
                    
                    result = self.run_improved_experiment(
                        source_piece, target_piece,
                        pair['coupling_strength'], "high_coupling",
                        edit_strength, edit_type, experiment_id
                    )
                    
                    all_results["experiments"].append(result)
        
        # 低耦合组实验
        print(f"\n🔵 低耦合组实验")
        low_coupling_mask = (
            (self.coupling_df['coupling_strength'] < 0.05) & 
            (self.coupling_df['is_same_hotpot'] == False) &
            (self.coupling_df['coupling_strength'] > -0.05)
        )
        low_coupling_candidates = self.coupling_df[low_coupling_mask].head(2)
        
        for i, (_, row) in enumerate(low_coupling_candidates.iterrows(), 1):
            source_piece = self.piece_map[row['piece_1_id']].copy()
            target_piece = self.piece_map[row['piece_2_id']].copy()
            
            source_piece['piece_id'] = row['piece_1_id']
            target_piece['piece_id'] = row['piece_2_id']
            
            for j, edit_strength in enumerate(edit_strengths, 1):
                for k, edit_type in enumerate(edit_types, 1):
                    experiment_id = f"L{i}-{j}-{k}"
                    
                    result = self.run_improved_experiment(
                        source_piece, target_piece,
                        row['coupling_strength'], "low_coupling",
                        edit_strength, edit_type, experiment_id
                    )
                    
                    all_results["experiments"].append(result)
        
        return all_results
    
    def analyze_results(self, results: Dict) -> Dict:
        """分析实验结果"""
        
        print(f"\n📊 分析改进实验结果")
        
        experiments = results["experiments"]
        
        # 按组和编辑类型分类
        analysis = {
            "by_coupling_type": {},
            "by_edit_type": {},
            "by_edit_strength": {},
            "overall_statistics": {}
        }
        
        # 按耦合类型分析
        for coupling_type in ["high_coupling", "low_coupling"]:
            coupling_experiments = [exp for exp in experiments if exp["pair_type"] == coupling_type and exp["edit_success"]]
            
            if coupling_experiments:
                ripples = [exp["ripple_effect"]["main_ripple_strength"] for exp in coupling_experiments]
                analysis["by_coupling_type"][coupling_type] = {
                    "count": len(coupling_experiments),
                    "mean_ripple": np.mean(ripples),
                    "std_ripple": np.std(ripples),
                    "min_ripple": np.min(ripples),
                    "max_ripple": np.max(ripples)
                }
        
        # 按编辑类型分析
        for edit_type in ["answer_based", "suppression", "random_control"]:
            type_experiments = [exp for exp in experiments if exp["edit_type"] == edit_type and exp["edit_success"]]
            
            if type_experiments:
                high_experiments = [exp for exp in type_experiments if exp["pair_type"] == "high_coupling"]
                low_experiments = [exp for exp in type_experiments if exp["pair_type"] == "low_coupling"]
                
                analysis["by_edit_type"][edit_type] = {
                    "high_coupling": {
                        "count": len(high_experiments),
                        "mean_ripple": np.mean([exp["ripple_effect"]["main_ripple_strength"] for exp in high_experiments]) if high_experiments else 0
                    },
                    "low_coupling": {
                        "count": len(low_experiments),
                        "mean_ripple": np.mean([exp["ripple_effect"]["main_ripple_strength"] for exp in low_experiments]) if low_experiments else 0
                    }
                }
                
                if high_experiments and low_experiments:
                    high_mean = analysis["by_edit_type"][edit_type]["high_coupling"]["mean_ripple"]
                    low_mean = analysis["by_edit_type"][edit_type]["low_coupling"]["mean_ripple"]
                    analysis["by_edit_type"][edit_type]["improvement_percentage"] = ((high_mean - low_mean) / low_mean * 100) if low_mean > 0 else 0
        
        # 总体统计
        successful_experiments = [exp for exp in experiments if exp["edit_success"]]
        analysis["overall_statistics"] = {
            "total_experiments": len(experiments),
            "successful_experiments": len(successful_experiments),
            "success_rate": len(successful_experiments) / len(experiments) if experiments else 0
        }
        
        return analysis

def main():
    """主函数"""
    print("🔬 启动改进的知识编辑实验")
    print("Let's make this editing more precise and effective!")
    
    # 初始化改进编辑器
    hf_token = "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"
    editor = ImprovedKnowledgeEditor(hf_token=hf_token)
    
    # 运行全面实验
    print(f"\n🎯 开始运行改进实验...")
    results = editor.run_comprehensive_experiments()
    
    # 分析结果
    analysis = editor.analyze_results(results)
    results["analysis"] = analysis
    
    # 保存结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"improved_experiment_results_{timestamp}.json"
    
    print(f"\n💾 保存结果到: {filename}")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 显示摘要
    print(f"\n📋 改进实验摘要:")
    print(f"   总实验次数: {analysis['overall_statistics']['total_experiments']}")
    print(f"   成功率: {analysis['overall_statistics']['success_rate']:.1%}")
    
    for edit_type, data in analysis["by_edit_type"].items():
        if "improvement_percentage" in data:
            print(f"   {edit_type} 编辑类型改善: {data['improvement_percentage']:.1f}%")
    
    print(f"\n✅ 改进实验完成，结果已保存")
    return filename

if __name__ == "__main__":
    main() 