#!/usr/bin/env python3
"""
文件清理脚本
删除不必要的中间文件，保留核心工作流程文件
"""

import os
import shutil
from typing import List

def cleanup_files():
    """清理不必要的文件"""
    
    print("🧹 开始清理不必要的文件...")
    
    # 可以删除的文件列表
    files_to_delete = [
        # 调试和测试文件
        "debug_hotpot.py",
        "debug_model_structure.py", 
        "test_downloader.py",
        "test_multi_model.py",
        "test_hotpot_20.json",
        "test_mixed_dataset.json",
        
        # 早期/中间版本
        "demo_results.py",
        "demo_ripple_effects.py", 
        "detailed_ripple_demo.py",
        "final_ripple_experiment.py",
        "improved_ripple_experiment.py",
        "live_ripple_experiment.py",
        "ripple_effect_validation.py",
        "save_experiment_results.py",
        
        # 多余的转换脚本
        "convert_chains_to_hotpot.py",
        "download_hotpot_only.py",
        
        # 大量的中间耦合文件 (占空间太大)
        "hotpot_all_coupling.json",
        "hotpot_train_coupling.json", 
        "hotpot_dev_coupling.json",
        "hotpot_coupling_1000.json",
        "hotpot_coupling_500.json",
        "hotpot_coupling_100.json",
        "hotpot_coupling_50.json",
        "hotpot_coupling_20.json",
        
        # 旧的实验结果
        "experiment_results_20250610_085634.json",
        "experiment_results_20250610_085948.json"
    ]
    
    deleted_files = []
    deleted_size = 0
    
    for filename in files_to_delete:
        if os.path.exists(filename):
            try:
                file_size = os.path.getsize(filename)
                os.remove(filename)
                deleted_files.append(filename)
                deleted_size += file_size
                print(f"   ✅ 删除: {filename} ({file_size / (1024*1024):.1f}MB)")
            except Exception as e:
                print(f"   ❌ 删除失败: {filename} - {e}")
        else:
            print(f"   ⚠️  文件不存在: {filename}")
    
    print(f"\n📊 清理统计:")
    print(f"   删除文件数: {len(deleted_files)}")
    print(f"   释放空间: {deleted_size / (1024*1024*1024):.2f}GB")
    
    return deleted_files

def show_remaining_files():
    """显示保留的核心文件"""
    
    print(f"\n📁 保留的核心文件:")
    
    core_files = {
        "1. 数据下载": [
            "multihop_dataset_downloader.py",
            "mixed_multihop_100.json"
        ],
        "2. 数据转换": [
            "convert_hotpot_to_coupling_format.py"
        ],
        "3. 计算GradSim": [
            "knowledge_coupling_mvp.py"
        ],
        "4. 验证GradSim": [
            "coupling_validation/"
        ],
        "5. 知识编辑": [
            "improved_knowledge_editor.py"
        ],
        "6. 实验结果": [
            "improved_experiment_results_20250610_090950.json",
            "export_report_data.py", 
            "report_data_20250610_091336.json"
        ],
        "7. 工具脚本": [
            "cleanup_files.py"
        ]
    }
    
    for category, files in core_files.items():
        print(f"\n{category}:")
        for file in files:
            if os.path.exists(file):
                if os.path.isdir(file):
                    size = sum(os.path.getsize(os.path.join(file, f)) 
                              for f in os.listdir(file) if os.path.isfile(os.path.join(file, f)))
                    print(f"   ✅ {file} (目录, {size/1024:.1f}KB)")
                else:
                    size = os.path.getsize(file)
                    print(f"   ✅ {file} ({size/1024:.1f}KB)")
            else:
                print(f"   ❌ {file} (不存在)")

def get_current_disk_usage():
    """获取当前目录磁盘使用情况"""
    total_size = 0
    file_count = 0
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            try:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                file_count += 1
            except:
                pass
    
    return total_size, file_count

def main():
    """主函数"""
    print("🧹 文件清理工具")
    print("=" * 60)
    
    # 清理前的状态
    before_size, before_count = get_current_disk_usage()
    print(f"清理前: {before_count} 个文件，总大小 {before_size / (1024*1024*1024):.2f}GB")
    
    # 执行清理
    deleted = cleanup_files()
    
    # 清理后的状态  
    after_size, after_count = get_current_disk_usage()
    print(f"\n清理后: {after_count} 个文件，总大小 {after_size / (1024*1024*1024):.2f}GB")
    print(f"节省空间: {(before_size - after_size) / (1024*1024*1024):.2f}GB")
    
    # 显示保留的文件
    show_remaining_files()
    
    print(f"\n🎯 清理完成!")
    print(f"现在您有一个干净的工作目录，包含完整的核心工作流程文件。")
    
if __name__ == "__main__":
    main() 