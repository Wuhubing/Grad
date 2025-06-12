#!/usr/bin/env python3
"""
Knowledge Coupling MVP for Multi-Model Support (LLaMA-2 & GPT-2)

真正的知识片段间耦合度计算和涟漪效应分析系统
现在支持多种模型架构：LLaMA-2 7B/13B/70B 和 GPT-2 (small/medium/large/xl)

核心方法：
1. 梯度抽取：针对每个知识片段的真实答案token计算梯度
2. GradSim计算：知识片段间参数空间重叠度量
3. 自适应层选择：根据模型类型自动选择目标层
4. 统一接口：支持多种模型的统一分析接口

Formula: GradSim(i,j) = cos(∇_θ log P(a_i|q_i), ∇_θ log P(a_j|q_j))
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from transformers import (
    LlamaTokenizer, LlamaForCausalLM,
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForCausalLM
)
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class KnowledgePiece:
    """单个知识片段的表示"""
    def __init__(self, piece_id: str, question: str, answer: str, 
                 supporting_fact: str, category: str):
        self.piece_id = piece_id
        self.question = question  # 针对这个知识片段的cloze问题
        self.answer = answer      # 真实答案token
        self.supporting_fact = supporting_fact  # 支持事实
        self.category = category


class MultiModelKnowledgeCouplingMVP:
    """多模型知识耦合MVP分析器 - 支持LLaMA-2和GPT-2"""
    
    # 支持的模型配置
    SUPPORTED_MODELS = {
        'llama': {
            'patterns': ['llama', 'Llama'],
            'tokenizer_class': LlamaTokenizer,
            'model_class': LlamaForCausalLM,
            'target_layers': 'down_proj',  # LLaMA MLP层
            'layer_pattern': 'model.layers.{}.mlp.down_proj'
        },
        'gpt2': {
            'patterns': ['gpt2', 'GPT2'],
            'tokenizer_class': GPT2Tokenizer,
            'model_class': GPT2LMHeadModel,
            'target_layers': 'mlp.c_proj',  # GPT-2 MLP层
            'layer_pattern': 'transformer.h.{}.mlp.c_proj'
        }
    }
    
    def __init__(self, model_path: str = "gpt2", device: str = None, 
                 layer_range: Optional[Tuple[int, int]] = None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = self._detect_model_type(model_path)
        self.layer_range = layer_range  # 新增：指定层范围
        
        print(f"🤖 Initializing Multi-Model Knowledge Coupling MVP")
        print(f"Model: {model_path}")
        print(f"Type: {self.model_type.upper()}")
        print(f"Device: {self.device}")
        if layer_range:
            print(f"Layer range: {layer_range[0]}-{layer_range[1]} (focusing on high-level semantic layers)")
        
        # 根据模型类型加载相应的tokenizer和model
        self._load_model()
        
        # 存储结果
        self.knowledge_pieces = []
        self.gradients = {}
        self.coupling_matrix = None
        self.edit_results = []
    
    def _detect_model_type(self, model_path: str) -> str:
        """检测模型类型"""
        model_path_lower = model_path.lower()
        
        for model_type, config in self.SUPPORTED_MODELS.items():
            for pattern in config['patterns']:
                if pattern.lower() in model_path_lower:
                    return model_type
        
        # 默认尝试auto检测
        print(f"⚠️ Unknown model type for {model_path}, trying auto-detection...")
        return 'auto'
    
    def _load_model(self):
        """根据模型类型加载对应的模型和tokenizer"""
        if self.model_type == 'auto':
            # 使用AutoTokenizer和AutoModelForCausalLM
            print("🔍 Using auto-detection for model loading...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            # 尝试推断实际的模型类型
            self._infer_model_type_from_loaded_model()
        else:
            # 使用指定的模型类型
            config = self.SUPPORTED_MODELS[self.model_type]
            print(f"📦 Loading {self.model_type.upper()} model...")
            
            self.tokenizer = config['tokenizer_class'].from_pretrained(self.model_path)
            self.model = config['model_class'].from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.model.eval()
        print(f"✅ Model loaded successfully!")
    
    def _infer_model_type_from_loaded_model(self):
        """从已加载的模型推断模型类型"""
        model_class_name = self.model.__class__.__name__
        
        if 'Llama' in model_class_name:
            self.model_type = 'llama'
        elif 'GPT2' in model_class_name:
            self.model_type = 'gpt2'
        else:
            # 检查模型结构来推断
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                self.model_type = 'llama'
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                self.model_type = 'gpt2'
            else:
                print("⚠️ Could not infer model type, defaulting to gpt2")
                self.model_type = 'gpt2'
        
        print(f"🔍 Inferred model type: {self.model_type.upper()}")
    
    def _get_target_layers(self) -> List[str]:
        """获取目标层名称列表 - 支持指定层范围"""
        if self.model_type not in self.SUPPORTED_MODELS:
            # 通用方法：查找所有MLP相关层
            target_layers = []
            for name, param in self.model.named_parameters():
                if any(keyword in name.lower() for keyword in ['mlp', 'ffn', 'feed_forward']):
                    if any(keyword in name.lower() for keyword in ['proj', 'linear', 'dense']):
                        target_layers.append(name)
            return target_layers
        
        config = self.SUPPORTED_MODELS[self.model_type]
        target_layers = []
        
        # 获取层数
        if self.model_type == 'llama':
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                num_layers = len(self.model.model.layers)
            else:
                num_layers = 32  # 默认值
        elif self.model_type == 'gpt2':
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                num_layers = len(self.model.transformer.h)
            else:
                num_layers = 12  # 默认值
        else:
            num_layers = 12
        
        # 确定层范围
        if self.layer_range:
            start_layer, end_layer = self.layer_range
            start_layer = max(0, start_layer)
            end_layer = min(num_layers, end_layer + 1)  # +1 because range is exclusive
        else:
            # 默认使用最后4层（高层语义表示）
            if self.model_type == 'llama':
                start_layer = max(0, num_layers - 4)  # 最后4层
                end_layer = num_layers
            else:  # gpt2
                start_layer = max(0, num_layers - 4)  # 最后4层
                end_layer = num_layers
        
        # 生成目标层名称
        layer_pattern = config['layer_pattern']
        for i in range(start_layer, end_layer):
            layer_name = layer_pattern.format(i)
            target_layers.append(layer_name)
        
        print(f"🎯 Using layers {start_layer}-{end_layer-1} ({len(target_layers)} layers total)")
        return target_layers
    
    def extract_knowledge_pieces_from_hotpot(self, hotpot_data: List[Dict], 
                                           max_samples: int = 100) -> List[KnowledgePiece]:
        """从HotpotQA提取2-hop知识片段"""
        print(f"📚 Extracting knowledge pieces from HotpotQA...")
        
        knowledge_pieces = []
        processed = 0
        
        for sample in hotpot_data:
            if processed >= max_samples:
                break
                
            if len(sample['supporting_facts']) < 2:
                continue
                
            question = sample['question']
            answer = sample['answer']
            supporting_facts = sample['supporting_facts']
            
            # 取前两个支持事实作为hop-1和hop-2
            for hop_idx, (title, sent_id) in enumerate(supporting_facts[:2]):
                # 从context中找到对应的句子
                sent_text = None
                for context_title, context_sents in sample['context']:
                    if context_title == title and sent_id < len(context_sents):
                        sent_text = context_sents[sent_id]
                        break
                
                if sent_text:
                    # 创建针对这个knowledge piece的cloze问题
                    # 将supporting fact转换为cloze格式
                    cloze_question = self._create_cloze_question(sent_text, answer)
                    
                    # 确定类别
                    category = self._categorize_knowledge(sent_text, answer)
                    
                    piece = KnowledgePiece(
                        piece_id=f"{sample['_id']}_hop_{hop_idx}",
                        question=cloze_question,
                        answer=answer,
                        supporting_fact=sent_text,
                        category=category
                    )
                    
                    knowledge_pieces.append(piece)
            
            processed += 1
        
        self.knowledge_pieces = knowledge_pieces
        print(f"✅ Extracted {len(knowledge_pieces)} knowledge pieces from {processed} samples")
        
        return knowledge_pieces
    
    def _create_cloze_question(self, supporting_fact: str, answer: str) -> str:
        """将支持事实转换为cloze问题"""
        # 简单的替换策略：将答案替换为空白
        if answer.lower() in supporting_fact.lower():
            cloze = supporting_fact.replace(answer, "___", 1)
            return f"Fill in the blank: {cloze}"
        else:
            # 如果直接替换不行，创建一个基于事实的问题
            return f"Based on the fact '{supporting_fact}', what is the answer? ___"
    
    def _categorize_knowledge(self, supporting_fact: str, answer: str) -> str:
        """简单的知识类别分类"""
        fact_lower = supporting_fact.lower()
        answer_lower = answer.lower()
        
        # 基于关键词的简单分类
        if any(word in fact_lower for word in ['born', 'actor', 'director', 'person', 'he', 'she']):
            return 'Person'
        elif any(word in fact_lower for word in ['company', 'organization', 'corporation', 'founded']):
            return 'Organization'  
        elif any(word in fact_lower for word in ['city', 'country', 'located', 'state', 'region']):
            return 'Location'
        elif any(word in fact_lower for word in ['film', 'movie', 'book', 'album', 'song']):
            return 'Work'
        elif any(word in fact_lower for word in ['year', 'date', 'century', 'time', 'when']):
            return 'Time'
        else:
            return 'Science'
    
    def compute_knowledge_gradient(self, piece: KnowledgePiece) -> Optional[torch.Tensor]:
        """计算知识片段的梯度向量 - GPU优化版本，支持指定层范围"""
        try:
            prompt = piece.question + " Answer:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # 编码答案
            answer_tokens = self.tokenizer(piece.answer, add_special_tokens=False)['input_ids']
            if not answer_tokens:
                print(f"Warning: No tokens for answer '{piece.answer}'")
                return None
            
            target_token_id = answer_tokens[0]
            
            # 计算logits并设置梯度
            self.model.eval()
            self.model.zero_grad()
            
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # 计算目标答案的log probability
            log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
            target_logp = log_probs[target_token_id]
            
            # 反向传播计算梯度
            target_logp.backward()
            
            # 提取目标层的梯度并在GPU上拼接
            target_layers = self._get_target_layers()
            gradient_tensors = []
            
            for name, param in self.model.named_parameters():
                if any(target_layer in name for target_layer in target_layers):
                    if param.grad is not None:
                        # 保持在GPU上，展平为1D张量
                        flat_grad = param.grad.flatten()
                        gradient_tensors.append(flat_grad)
            
            if gradient_tensors:
                # 在GPU上拼接所有梯度
                full_gradient = torch.cat(gradient_tensors, dim=0)
                # 标准化
                full_gradient = F.normalize(full_gradient, p=2, dim=0)
                return full_gradient
            else:
                print(f"Warning: No gradients found for {piece.piece_id}")
                return None
                
        except Exception as e:
            print(f"Error computing gradient for {piece.piece_id}: {e}")
            return None
        finally:
            self.model.zero_grad()

    def compute_all_gradients(self) -> Dict[str, torch.Tensor]:
        """计算所有知识片段的梯度 - GPU优化版本"""
        if not self.knowledge_pieces:
            raise ValueError("No knowledge pieces loaded.")
        
        print(f"🔬 Computing gradients for all knowledge pieces using {self.model_type}...")
        
        gradients = {}
        
        # 使用tqdm显示进度，但保持张量在GPU上
        for piece in tqdm(self.knowledge_pieces, desc="Computing gradients"):
            gradient = self.compute_knowledge_gradient(piece)
            if gradient is not None:
                gradients[piece.piece_id] = gradient  # 保持GPU张量
        
        print(f"✅ Successfully computed {len(gradients)} gradients")
        
        # 更新类属性
        self.gradients = gradients
        
        # 显示目标层信息
        target_layers = self._get_target_layers()
        print(f"🎯 Target layers used: {len(target_layers)} layers")
        if len(target_layers) <= 5:
            for layer in target_layers[:5]:
                print(f"   - {layer}")
        else:
            print(f"   - {target_layers[0]} ... {target_layers[-1]} (showing first and last)")
        
        return gradients
    
    def compute_coupling_matrix(self) -> torch.Tensor:
        """计算知识片段间的耦合矩阵 - GPU优化版本"""
        if not self.gradients:
            raise ValueError("No gradients computed. Run compute_all_gradients() first.")
        
        print("🔗 Computing knowledge coupling matrix...")
        
        # 准备梯度矩阵 - 在GPU上操作
        piece_ids = list(self.gradients.keys())
        gradient_list = [self.gradients[pid] for pid in piece_ids]
        
        # 堆叠为矩阵 [n_pieces, gradient_dim]
        gradient_matrix = torch.stack(gradient_list, dim=0)
        
        print(f"📊 Gradient matrix shape: {gradient_matrix.shape}")
        print(f"🚀 Computing on device: {gradient_matrix.device}")
        
        # 计算余弦相似度矩阵 - 在GPU上
        # gradient_matrix已经标准化了，所以直接矩阵乘法
        coupling_matrix = torch.mm(gradient_matrix, gradient_matrix.t())
        
        # 将对角线设为1.0（自相似）
        coupling_matrix.fill_diagonal_(1.0)
        
        self.coupling_matrix = coupling_matrix
        self.piece_ids_order = piece_ids  # 保存顺序
        
        print(f"📊 Computed {coupling_matrix.shape[0]}×{coupling_matrix.shape[1]} coupling matrix on GPU")
        return coupling_matrix

    def save_analysis_results(self, output_dir: str = "results/coupling_analysis", dataset_info: Dict[str, Any] = None):
        """保存分析结果为跨pot分析所需格式"""
        import os
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"💾 Saving analysis results to {output_dir}/...")
        
        # 1. 保存知识片段为JSON
        pieces_data = []
        for piece in self.knowledge_pieces:
            pieces_data.append({
                'piece_id': piece.piece_id,
                'question': piece.question,
                'answer': piece.answer,
                'supporting_fact': piece.supporting_fact,
                'category': piece.category
            })
        
        pieces_file = os.path.join(output_dir, "knowledge_pieces.json")
        with open(pieces_file, 'w', encoding='utf-8') as f:
            json.dump(pieces_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(pieces_data)} knowledge pieces to knowledge_pieces.json")
        
        # 2. 保存耦合矩阵为numpy格式
        if self.coupling_matrix is not None:
            # 转移到CPU并转换为numpy
            coupling_np = self.coupling_matrix.detach().cpu().numpy()
            coupling_file = os.path.join(output_dir, "coupling_matrix.npy")
            np.save(coupling_file, coupling_np)
            print(f"✅ Saved coupling matrix ({coupling_np.shape}) to coupling_matrix.npy")
            
            # 3. 保存片段ID顺序
            order_file = os.path.join(output_dir, "piece_ids_order.json")
            with open(order_file, 'w', encoding='utf-8') as f:
                json.dump(self.piece_ids_order, f, indent=2)
            print(f"✅ Saved piece IDs order to piece_ids_order.json")
        
        # 4. 保存GPU性能信息和数据集信息
        analysis_metadata = {
            'model_info': {
                'model_path': self.model_path,
                'model_type': self.model_type,
                'layer_range': self.layer_range,
                'target_layers_count': len(self._get_target_layers())
            },
            'dataset_info': dataset_info or {},
            'analysis_timestamp': self._get_timestamp()
        }
        
        if torch.cuda.is_available():
            analysis_metadata['gpu_info'] = {
                'device': str(self.device),
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1e9,
                'gpu_memory_reserved': torch.cuda.memory_reserved() / 1e9,
                'gradient_computation': 'GPU',
                'coupling_computation': 'GPU'
            }
            
        metadata_file = os.path.join(output_dir, "analysis_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_metadata, f, indent=2)
        print(f"✅ Saved analysis metadata and dataset info to analysis_metadata.json")
        
        print(f"🎯 All results saved to {output_dir}/ - Ready for analysis!")

    def analyze_coupling_patterns(self) -> Dict[str, Any]:
        """分析耦合模式 - 兼容GPU张量"""
        if self.coupling_matrix is None:
            raise ValueError("Coupling matrix not computed.")
        
        # 转换为numpy进行分析（如果是GPU张量）
        if isinstance(self.coupling_matrix, torch.Tensor):
            coupling_np = self.coupling_matrix.detach().cpu().numpy()
        else:
            coupling_np = self.coupling_matrix
        
        # 去除对角线（自相似）
        n = coupling_np.shape[0]
        mask = ~np.eye(n, dtype=bool)
        off_diagonal = coupling_np[mask]
        
        # 分析耦合强度分布
        high_coupling = np.sum(off_diagonal >= 0.4)
        moderate_coupling = np.sum((off_diagonal >= 0.1) & (off_diagonal < 0.4))
        low_coupling = np.sum(off_diagonal < 0.1)
        
        total_pairs = len(off_diagonal)
        
        analysis = {
            'total_pairs': total_pairs,
            'high_coupling_pairs': high_coupling,
            'moderate_coupling_pairs': moderate_coupling,
            'low_coupling_pairs': low_coupling,
            'high_coupling_ratio': high_coupling / total_pairs,
            'mean_coupling': float(np.mean(off_diagonal)),
            'std_coupling': float(np.std(off_diagonal)),
            'max_coupling': float(np.max(off_diagonal)),
            'min_coupling': float(np.min(off_diagonal))
        }
        
        return analysis

    def predict_ripple_candidates(self, threshold: float = 0.4) -> List[Tuple[str, str, float]]:
        """预测高耦合的知识片段对（涟漪效应候选）- 兼容GPU张量"""
        if self.coupling_matrix is None:
            raise ValueError("Coupling matrix not computed.")
        
        # 转换为numpy进行分析（如果是GPU张量）
        if isinstance(self.coupling_matrix, torch.Tensor):
            coupling_np = self.coupling_matrix.detach().cpu().numpy()
        else:
            coupling_np = self.coupling_matrix
        
        piece_ids = [p.piece_id for p in self.knowledge_pieces]
        high_coupling_pairs = []
        
        n = len(piece_ids)
        for i in range(n):
            for j in range(i + 1, n):
                coupling_strength = coupling_np[i, j]
                if coupling_strength >= threshold:
                    high_coupling_pairs.append((piece_ids[i], piece_ids[j], float(coupling_strength)))
        
        # 按耦合强度排序
        high_coupling_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"🔥 Found {len(high_coupling_pairs)} high coupling pairs (threshold={threshold})")
        return high_coupling_pairs
    
    def simulate_memit_edit(self, target_piece_id: str, new_answer: str) -> Dict[str, Any]:
        """模拟MEMIT编辑操作（简化版）"""
        # 找到目标知识片段
        target_piece = None
        for piece in self.knowledge_pieces:
            if piece.piece_id == target_piece_id:
                target_piece = piece
                break
        
        if target_piece is None:
            raise ValueError(f"Knowledge piece {target_piece_id} not found.")
        
        print(f"🔧 Simulating MEMIT edit on {target_piece_id}")
        print(f"Original answer: {target_piece.answer}")
        print(f"New answer: {new_answer}")
        
        # 计算编辑前的logP
        original_logp = self._compute_logp(target_piece.question, target_piece.answer)
        
        # 模拟编辑后的logP变化（这里简化处理）
        # 在真实实现中，这里应该用MEMIT修改模型参数
        edited_logp = self._compute_logp(target_piece.question, new_answer)
        
        delta_logp = edited_logp - original_logp
        
        edit_result = {
            'target_piece_id': target_piece_id,
            'original_answer': target_piece.answer,
            'new_answer': new_answer,
            'original_logp': original_logp,
            'edited_logp': edited_logp,
            'delta_logp': delta_logp,
            'edit_success': delta_logp > 0  # 简化的成功判断
        }
        
        return edit_result
    
    def _compute_logp(self, question: str, answer: str) -> float:
        """计算问题-答案对的log probability"""
        try:
            prompt = f"{question} Answer:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            answer_tokens = self.tokenizer(answer, add_special_tokens=False)['input_ids']
            if not answer_tokens:
                return float('-inf')
            
            target_token_id = answer_tokens[0]
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
                return log_probs[target_token_id].item()
                
        except Exception as e:
            print(f"Error computing logP: {e}")
            return float('-inf')
    
    def _find_piece_by_id(self, piece_id: str) -> Optional[KnowledgePiece]:
        """根据ID查找知识片段"""
        for piece in self.knowledge_pieces:
            if piece.piece_id == piece_id:
                return piece
        return None
    
    def _analyze_cross_pot_patterns(self) -> Dict[str, Any]:
        """分析跨pot的耦合模式"""
        if not self.knowledge_pieces or self.coupling_matrix is None:
            return {}
        
        # 构建pot映射
        pot_to_pieces = {}
        piece_to_pot = {}
        
        for i, piece in enumerate(self.knowledge_pieces):
            pot_id = piece.piece_id.split('_hop_')[0]  # hotpot_train_X
            
            if pot_id not in pot_to_pieces:
                pot_to_pieces[pot_id] = []
            pot_to_pieces[pot_id].append(i)
            piece_to_pot[i] = pot_id
        
        # 转换为numpy进行分析
        if isinstance(self.coupling_matrix, torch.Tensor):
            coupling_np = self.coupling_matrix.detach().cpu().numpy()
        else:
            coupling_np = self.coupling_matrix
        
        # 分析跨pot vs 同pot耦合
        cross_pot_couplings = []
        same_pot_couplings = []
        
        for i in range(len(self.knowledge_pieces)):
            for j in range(i+1, len(self.knowledge_pieces)):
                coupling_strength = coupling_np[i, j]
                pot_i = piece_to_pot[i]
                pot_j = piece_to_pot[j]
                
                if pot_i != pot_j:
                    cross_pot_couplings.append(coupling_strength)
                else:
                    same_pot_couplings.append(coupling_strength)
        
        return {
            'unique_pots': len(pot_to_pieces),
            'cross_pot_pairs': len(cross_pot_couplings),
            'same_pot_pairs': len(same_pot_couplings),
            'cross_pot_mean': float(np.mean(cross_pot_couplings)) if cross_pot_couplings else 0.0,
            'same_pot_mean': float(np.mean(same_pot_couplings)) if same_pot_couplings else 0.0,
            'cross_pot_std': float(np.std(cross_pot_couplings)) if cross_pot_couplings else 0.0,
            'same_pot_std': float(np.std(same_pot_couplings)) if same_pot_couplings else 0.0,
            'high_cross_pot_pairs': sum(1 for c in cross_pot_couplings if c >= 0.4)
        }
    
    def _get_category_stats(self) -> Dict[str, int]:
        """获取知识片段类别统计"""
        category_counts = {}
        for piece in self.knowledge_pieces:
            category = piece.category
            category_counts[category] = category_counts.get(category, 0) + 1
        return category_counts
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def measure_ripple_effects(self, edit_result: Dict[str, Any], 
                             coupling_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """测量编辑的涟漪效应"""
        target_piece_id = edit_result['target_piece_id']
        
        # 找到与目标片段高耦合的其他片段
        target_idx = None
        piece_ids = [p.piece_id for p in self.knowledge_pieces]
        
        for i, pid in enumerate(piece_ids):
            if pid == target_piece_id:
                target_idx = i
                break
        
        if target_idx is None:
            return []
        
        ripple_effects = []
        
        for i, piece in enumerate(self.knowledge_pieces):
            if i != target_idx:
                coupling_strength = self.coupling_matrix[target_idx, i]
                
                if coupling_strength >= coupling_threshold:
                    # 测量这个片段的变化
                    original_logp = self._compute_logp(piece.question, piece.answer)
                    
                    # 在实际实现中，这里应该用编辑后的模型重新计算
                    # 这里简化处理：假设变化与耦合强度成正比
                    predicted_delta = edit_result['delta_logp'] * coupling_strength
                    
                    ripple_effect = {
                        'affected_piece_id': piece.piece_id,
                        'coupling_strength': coupling_strength,
                        'original_logp': original_logp,
                        'predicted_delta_logp': predicted_delta,
                        'category': piece.category
                    }
                    
                    ripple_effects.append(ripple_effect)
        
        # 按耦合强度排序
        ripple_effects.sort(key=lambda x: x['coupling_strength'], reverse=True)
        
        print(f"🌊 Measured {len(ripple_effects)} ripple effects")
        return ripple_effects
    
    def plot_coupling_heatmap(self, save_path: str = None, output_dir: str = "results/coupling_analysis"):
        """绘制耦合热图"""
        if self.coupling_matrix is None:
            raise ValueError("Coupling matrix not computed.")
        
        # 如果没有指定保存路径，使用output_dir
        if save_path is None:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "coupling_heatmap.png")
        
        plt.figure(figsize=(12, 10))
        
        # 创建标签
        labels = [f"K{i}" for i in range(len(self.knowledge_pieces))]
        
        # 转换GPU张量为numpy数组
        if isinstance(self.coupling_matrix, torch.Tensor):
            coupling_data = self.coupling_matrix.detach().cpu().numpy()
        else:
            coupling_data = self.coupling_matrix
        
        sns.heatmap(
            coupling_data,
            annot=False,
            cmap='RdYlBu_r',
            square=True,
            cbar_kws={'label': 'Knowledge Coupling Strength'},
            xticklabels=labels,
            yticklabels=labels
        )
        
        plt.title(f'Knowledge Coupling Matrix ({self.model_type.upper()})')
        plt.xlabel('Knowledge Pieces')
        plt.ylabel('Knowledge Pieces')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🎨 Coupling heatmap saved to: {save_path}")
        return save_path
    
    def generate_mvp_report(self, save_path: str = None, output_dir: str = "results/coupling_analysis") -> str:
        """生成MVP分析报告"""
        if self.coupling_matrix is None:
            raise ValueError("Analysis not completed. Run full analysis first.")
        
        # 如果没有指定保存路径，使用output_dir
        if save_path is None:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "mvp_report.md")
        
        coupling_analysis = self.analyze_coupling_patterns()
        high_coupling_pairs = self.predict_ripple_candidates()
        
        with open(save_path, 'w') as f:
            f.write("# Knowledge Coupling MVP Analysis Report\n\n")
            f.write("## 🎯 Research Goal\n")
            f.write(f"Quantify knowledge piece coupling in {self.model_type.upper()} using gradient similarity analysis.\n\n")
            
            f.write("## 📊 Key Findings\n\n")
            f.write(f"**Total Knowledge Pieces:** {len(self.knowledge_pieces)}  \n")
            f.write(f"**Mean Coupling Strength:** {coupling_analysis['mean_coupling']:.4f}  \n")
            f.write(f"**High Coupling Pairs:** {coupling_analysis['high_coupling_pairs']} ({coupling_analysis['high_coupling_ratio']:.2%})  \n")
            f.write(f"**Coupling Range:** [{coupling_analysis['min_coupling']:.4f}, {coupling_analysis['max_coupling']:.4f}]  \n\n")
            
            f.write("## 🔥 Top High-Coupling Pairs\n\n")
            f.write("| Knowledge Piece 1 | Knowledge Piece 2 | Coupling Strength |\n")
            f.write("|-------------------|-------------------|-------------------|\n")
            
            for pair in high_coupling_pairs[:10]:
                f.write(f"| {pair[0]} | {pair[1]} | {pair[2]:.4f} |\n")
            
            f.write("\n## 🔬 Methodology\n\n")
            f.write("1. **Knowledge Extraction:** HotpotQA 2-hop chains → cloze questions\n")
            f.write(f"2. **Gradient Computation:** ∇_θ log P(answer|question) targeting {self.model_type.upper()} layers\n")
            f.write("3. **Coupling Measurement:** Cosine similarity between gradient vectors\n")
            f.write("4. **Ripple Prediction:** High coupling (≥0.4) → potential ripple effects\n\n")
            
            f.write("## 🎯 Next Steps\n\n")
            f.write("1. **MEMIT Integration:** Implement actual knowledge editing\n")
            f.write("2. **Ripple Measurement:** ΔlogP and EM evaluation on edited model\n")
            f.write("3. **Validation:** GradSim vs actual ripple correlation analysis\n\n")
            
            f.write("---\n*Generated by Multi-Model Knowledge Coupling MVP*\n")
        
        print(f"📝 MVP report saved to: {save_path}")
        return save_path
    
    def save_coupling_results_for_validation(self, output_dir: str = "results/coupling_analysis", dataset_info: Dict[str, Any] = None):
        """保存耦合度计算结果用于验证 - 专注于核心数据"""
        import os
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"💾 Saving coupling results for validation to {output_dir}/...")
        
        # 转换耦合矩阵为numpy
        if isinstance(self.coupling_matrix, torch.Tensor):
            coupling_np = self.coupling_matrix.detach().cpu().numpy()
        else:
            coupling_np = self.coupling_matrix
        
        # 1. 保存详细的耦合度结果
        coupling_results = {
            'metadata': {
                'model_path': self.model_path,
                'model_type': self.model_type,
                'layer_range': self.layer_range,
                'total_knowledge_pieces': len(self.knowledge_pieces),
                'matrix_shape': coupling_np.shape,
                'timestamp': self._get_timestamp()
            },
            'dataset_info': dataset_info or {},
            'knowledge_pieces': [],
            'coupling_pairs': [],
            'statistics': {}
        }
        
        # 添加知识片段信息
        for i, piece in enumerate(self.knowledge_pieces):
            coupling_results['knowledge_pieces'].append({
                'index': i,
                'piece_id': piece.piece_id,
                'question': piece.question,
                'answer': piece.answer,
                'category': piece.category
            })
        
        # 添加所有耦合对（不只是高耦合的）
        piece_ids = [p.piece_id for p in self.knowledge_pieces]
        for i in range(len(piece_ids)):
            for j in range(i + 1, len(piece_ids)):
                coupling_strength = float(coupling_np[i, j])
                coupling_results['coupling_pairs'].append({
                    'piece_1_index': i,
                    'piece_2_index': j,
                    'piece_1_id': piece_ids[i],
                    'piece_2_id': piece_ids[j],
                    'coupling_strength': coupling_strength,
                    'is_same_hotpot': piece_ids[i].split('_hop_')[0] == piece_ids[j].split('_hop_')[0]
                })
        
        # 添加统计信息
        off_diagonal = coupling_np[~np.eye(coupling_np.shape[0], dtype=bool)]
        coupling_results['statistics'] = {
            'mean_coupling': float(np.mean(off_diagonal)),
            'std_coupling': float(np.std(off_diagonal)),
            'min_coupling': float(np.min(off_diagonal)),
            'max_coupling': float(np.max(off_diagonal)),
            'high_coupling_count': int(np.sum(off_diagonal >= 0.4)),
            'moderate_coupling_count': int(np.sum((off_diagonal >= 0.1) & (off_diagonal < 0.4))),
            'low_coupling_count': int(np.sum(off_diagonal < 0.1))
        }
        
        # 保存为JSON文件
        coupling_file = os.path.join(output_dir, "coupling_results_for_validation.json")
        with open(coupling_file, 'w', encoding='utf-8') as f:
            json.dump(coupling_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved coupling results for validation: coupling_results_for_validation.json")
        
        # 2. 保存简化的CSV格式用于快速验证
        csv_file = os.path.join(output_dir, "coupling_pairs.csv")
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("piece_1_id,piece_2_id,coupling_strength,is_same_hotpot,piece_1_answer,piece_2_answer\n")
            for pair in coupling_results['coupling_pairs']:
                p1_answer = coupling_results['knowledge_pieces'][pair['piece_1_index']]['answer']
                p2_answer = coupling_results['knowledge_pieces'][pair['piece_2_index']]['answer']
                f.write(f"{pair['piece_1_id']},{pair['piece_2_id']},{pair['coupling_strength']:.6f},{pair['is_same_hotpot']},{p1_answer},{p2_answer}\n")
        
        print(f"✅ Saved CSV format for quick analysis: coupling_pairs.csv")
        
        # 3. 保存高耦合对的详细信息
        high_coupling_pairs = [p for p in coupling_results['coupling_pairs'] if p['coupling_strength'] >= 0.4]
        high_coupling_pairs.sort(key=lambda x: x['coupling_strength'], reverse=True)
        
        high_coupling_file = os.path.join(output_dir, "high_coupling_pairs.json")
        with open(high_coupling_file, 'w', encoding='utf-8') as f:
            json.dump({
                'threshold': 0.4,
                'count': len(high_coupling_pairs),
                'pairs': high_coupling_pairs
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved high coupling pairs: high_coupling_pairs.json ({len(high_coupling_pairs)} pairs)")
        
        return coupling_file

    def run_mvp_analysis(self, hotpot_data: List[Dict], max_samples: int = 50, 
                         output_dir: str = "results/coupling_analysis", 
                         dataset_file_path: str = None,
                         generate_report: bool = False,
                         generate_heatmap: bool = False) -> Dict[str, Any]:
        """运行完整的MVP分析"""
        print("\n" + "🤖" + "="*50 + "🤖")
        print("MULTI-MODEL KNOWLEDGE COUPLING MVP ANALYSIS")
        print("="*52)
        
        # 准备数据集信息
        dataset_info = {
            'dataset_file_path': dataset_file_path,
            'total_samples_in_file': len(hotpot_data),
            'samples_processed': min(max_samples, len(hotpot_data)),
            'dataset_name': 'HotpotQA',
            'data_format': 'multi-hop QA'
        }
        
        # 分析数据集基本信息
        if hotpot_data:
            sample_fields = list(hotpot_data[0].keys()) if hotpot_data else []
            dataset_info['sample_fields'] = sample_fields
            
            # 统计类别分布
            categories = {}
            for sample in hotpot_data[:max_samples]:
                category = sample.get('category', 'Unknown')
                categories[category] = categories.get(category, 0) + 1
            dataset_info['category_distribution'] = categories
        
        # Step 1: 提取知识片段
        knowledge_pieces = self.extract_knowledge_pieces_from_hotpot(hotpot_data, max_samples)
        
        # Step 2: 计算梯度
        gradients = self.compute_all_gradients()
        
        # Step 3: 计算耦合矩阵
        coupling_matrix = self.compute_coupling_matrix()
        
        # Step 4: 分析耦合模式
        coupling_analysis = self.analyze_coupling_patterns()
        
        # Step 5: 预测涟漪候选
        high_coupling_pairs = self.predict_ripple_candidates()
        
        # Step 6: 保存分析结果（为跨pot分析准备）
        self.save_analysis_results(output_dir, dataset_info)
        
        # Step 7: 保存专门用于验证的耦合度结果
        coupling_validation_file = self.save_coupling_results_for_validation(output_dir, dataset_info)
        
        # Step 8: 可选的可视化和报告
        heatmap_path = None
        report_path = None
        
        if generate_heatmap:
            heatmap_path = self.plot_coupling_heatmap(output_dir=output_dir)
        
        if generate_report:
            report_path = self.generate_mvp_report(output_dir=output_dir)
        
        # 准备详细的知识片段信息
        knowledge_pieces_detailed = []
        for piece in self.knowledge_pieces:
            piece_info = {
                'piece_id': piece.piece_id,
                'question': piece.question,
                'answer': piece.answer,
                'supporting_fact': piece.supporting_fact,
                'category': piece.category
            }
            knowledge_pieces_detailed.append(piece_info)
        
        # 准备涟漪效应候选对的详细信息
        ripple_candidates_detailed = []
        for i, (source_id, target_id, coupling_strength) in enumerate(high_coupling_pairs[:10]):
            source_piece = self._find_piece_by_id(source_id)
            target_piece = self._find_piece_by_id(target_id)
            
            candidate_info = {
                'rank': i + 1,
                'source_piece_id': source_id,
                'target_piece_id': target_id,
                'coupling_strength': float(coupling_strength),
                'source_answer': source_piece.answer if source_piece else "Unknown",
                'target_answer': target_piece.answer if target_piece else "Unknown",
                'source_question': source_piece.question if source_piece else "Unknown",
                'target_question': target_piece.question if target_piece else "Unknown",
                'source_category': source_piece.category if source_piece else "Unknown",
                'target_category': target_piece.category if target_piece else "Unknown"
            }
            ripple_candidates_detailed.append(candidate_info)
        
        # 分析不同pot的耦合情况
        cross_pot_analysis = self._analyze_cross_pot_patterns()
        
        # GPU性能信息
        gpu_performance = {}
        if torch.cuda.is_available():
            gpu_performance = {
                'device': str(self.device),
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / 1e9,
                'model_type': self.model_type,
                'gradient_computation': 'GPU',
                'coupling_computation': 'GPU'
            }
        
        # 构建完整的results
        results = {
            # 基本信息
            'metadata': {
                'model_path': self.model_path,
                'model_type': self.model_type,
                'device': str(self.device),
                'layer_range': self.layer_range,
                'total_samples_processed': len(knowledge_pieces),
                'max_samples_requested': max_samples,
                'output_directory': output_dir,
                'analysis_timestamp': self._get_timestamp()
            },
            
            # 数据集信息
            'dataset_info': dataset_info,
            
            # 知识片段详细信息
            'knowledge_pieces': {
                'count': len(knowledge_pieces),
                'details': knowledge_pieces_detailed,
                'categories': self._get_category_stats()
            },
            
            # 梯度计算结果
            'gradient_analysis': {
                'total_gradients_computed': len(gradients),
                'target_layers_count': len(self._get_target_layers()),
                'target_layers': self._get_target_layers()[:5] + (['...'] if len(self._get_target_layers()) > 5 else []),
                'gradient_dimension': gradients[list(gradients.keys())[0]].shape[0] if gradients else 0
            },
            
            # 耦合分析结果
            'coupling_analysis': coupling_analysis,
            
            # 高耦合对详细信息
            'high_coupling_pairs': {
                'count': len(high_coupling_pairs),
                'threshold': 0.4,
                'top_10_detailed': ripple_candidates_detailed,
                'all_pairs': [(pair[0], pair[1], float(pair[2])) for pair in high_coupling_pairs]
            },
            
            # 跨pot分析
            'cross_pot_analysis': cross_pot_analysis,
            
            # 文件路径
            'generated_files': {
                'coupling_matrix': f"{output_dir}/coupling_matrix.npy",
                'knowledge_pieces_json': f"{output_dir}/knowledge_pieces.json",
                'piece_ids_order': f"{output_dir}/piece_ids_order.json",
                'analysis_metadata': f"{output_dir}/analysis_metadata.json",
                'coupling_validation_file': coupling_validation_file,
                'coupling_pairs_csv': f"{output_dir}/coupling_pairs.csv",
                'high_coupling_pairs': f"{output_dir}/high_coupling_pairs.json",
                'heatmap_visualization': heatmap_path if generate_heatmap else None,
                'analysis_report': report_path if generate_report else None
            },
            
            # GPU性能信息
            'gpu_performance': gpu_performance,
            
            # 实验准备信息
            'experiment_readiness': {
                'ripple_candidates_available': len(high_coupling_pairs) > 0,
                'recommended_candidates': min(5, len(high_coupling_pairs)),
                'next_steps': [
                    "Validate coupling results using generated files",
                    "Execute ripple effect validation",
                    "Validate GradSim hypothesis"
                ]
            }
        }
        
        print(f"\n✅ MVP Analysis completed!")
        print(f"📊 Mean coupling: {coupling_analysis['mean_coupling']:.4f}")
        print(f"🔥 High coupling pairs: {len(high_coupling_pairs)}")
        print(f"🧪 Ripple candidates: {len(ripple_candidates_detailed)}")
        print(f"💾 Results saved to: {output_dir}/")
        print(f"🎯 Coupling validation file: {coupling_validation_file}")
        if report_path:
            print(f"📝 Report: {report_path}")
        print(f"🎯 Detailed results available in returned dictionary")
        
        return results


def load_hotpot_data(file_path: str) -> List[Dict]:
    """加载HotpotQA数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Model Knowledge Coupling MVP")
    parser.add_argument("--hotpot_data", required=True, help="Path to HotpotQA JSON file")
    parser.add_argument("--model_path", default="gpt2", help="Model path")
    parser.add_argument("--max_samples", type=int, default=50, help="Maximum samples to process")
    parser.add_argument("--device", help="Device (cuda/cpu)")
    parser.add_argument("--output_dir", default="results/coupling_analysis", help="Output directory for results")
    parser.add_argument("--layer_start", type=int, help="Start layer for gradient computation (default: auto)")
    parser.add_argument("--layer_end", type=int, help="End layer for gradient computation (default: auto)")
    parser.add_argument("--no_report", action="store_true", help="Skip generating analysis report (default: skip)")
    parser.add_argument("--no_heatmap", action="store_true", help="Skip generating coupling heatmap (default: skip)")
    parser.add_argument("--generate_report", action="store_true", help="Generate analysis report")
    parser.add_argument("--generate_heatmap", action="store_true", help="Generate coupling heatmap")
    
    args = parser.parse_args()
    
    # 处理层范围参数
    layer_range = None
    if args.layer_start is not None and args.layer_end is not None:
        layer_range = (args.layer_start, args.layer_end)
        print(f"🎯 Using custom layer range: {args.layer_start}-{args.layer_end}")
    elif args.model_path and 'llama' in args.model_path.lower():
        # 对于LLaMA模型，推荐使用高层 (最后4层：28,29,30,31)
        layer_range = (28, 31)
        print(f"🎯 Auto-selected high semantic layers for LLaMA: 28-31 (recommended by boss)")
    elif args.model_path and 'gpt2' in args.model_path.lower():
        # 对于GPT2模型，使用最后几层
        if 'large' in args.model_path.lower() or 'xl' in args.model_path.lower():
            layer_range = (44, 47)  # GPT2-large/xl的最后几层
        else:
            layer_range = (8, 11)   # GPT2-small/medium的最后几层
        print(f"🎯 Auto-selected high semantic layers for GPT2: {layer_range}")
    
    # 处理报告和可视化生成选项
    generate_report = args.generate_report and not args.no_report
    generate_heatmap = args.generate_heatmap and not args.no_heatmap
    
    if generate_report:
        print("📝 Will generate analysis report")
    else:
        print("⏭️  Skipping analysis report generation (use --generate_report to enable)")
    
    if generate_heatmap:
        print("🎨 Will generate coupling heatmap")
    else:
        print("⏭️  Skipping heatmap generation (use --generate_heatmap to enable)")
    
    # 加载数据
    print(f"📚 Loading HotpotQA data from {args.hotpot_data}")
    hotpot_data = load_hotpot_data(args.hotpot_data)
    print(f"Loaded {len(hotpot_data)} samples")
    
    # 初始化分析器
    analyzer = MultiModelKnowledgeCouplingMVP(
        model_path=args.model_path,
        device=args.device,
        layer_range=layer_range
    )
    
    # 运行分析
    results = analyzer.run_mvp_analysis(
        hotpot_data, 
        args.max_samples, 
        args.output_dir,
        dataset_file_path=args.hotpot_data,  # 传递数据集文件路径
        generate_report=generate_report,
        generate_heatmap=generate_heatmap
    )
    
    print(f"\n🎯 MVP Analysis completed successfully!")
    print(f"\n📁 Key validation files generated:")
    print(f"   📊 coupling_results_for_validation.json - Complete coupling analysis")
    print(f"   📈 coupling_pairs.csv - Quick CSV format for analysis")
    print(f"   🔥 high_coupling_pairs.json - High coupling pairs (≥0.4)")
    print(f"   💾 coupling_matrix.npy - Raw coupling matrix")


if __name__ == "__main__":
    main() 