#!/bin/bash

# 监控14B分类任务脚本

PID_FILE="logs/14b_classification.pid"
OUTPUT_DIR="results/optimized_high_coupling_classification"

function show_status() {
    echo "==============================================="
    echo "     14B模型分类任务监控 v1.0"
    echo "==============================================="
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 检查进程状态
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        
        if ps -p $PID > /dev/null; then
            echo "✅ 分类任务运行中 (PID: $PID)"
            
            # 显示运行时间
            start_time=$(ps -p $PID -o lstart= | xargs -I {} date -d "{}" +%s)
            current_time=$(date +%s)
            runtime=$((current_time - start_time))
            hours=$((runtime / 3600))
            minutes=$(((runtime % 3600) / 60))
            seconds=$((runtime % 60))
            echo "   运行时间: ${hours}h ${minutes}m ${seconds}s"
        else
            echo "❌ 分类任务未运行 (PID文件过期)"
        fi
    else
        echo "❌ 未找到分类任务"
    fi
    
    echo ""
    
    # GPU状态
    echo "🎮 GPU状态:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
    while IFS=, read -r idx name util mem_used mem_total temp; do
        mem_percent=$((mem_used * 100 / mem_total))
        echo "   GPU$idx: $name"
        echo "     利用率: ${util}%  显存: ${mem_used}MB/${mem_total}MB (${mem_percent}%)  温度: ${temp}°C"
    done
    
    echo ""
    
    # 内存状态
    echo "💾 内存使用:"
    free -h | grep -E "Mem|Swap" | while read -r line; do
        echo "   $line"
    done
    
    echo ""
    
    # 检查点状态
    if [ -d "$OUTPUT_DIR/checkpoints" ]; then
        checkpoint_count=$(ls -1 "$OUTPUT_DIR/checkpoints"/checkpoint_*.json 2>/dev/null | wc -l)
        if [ $checkpoint_count -gt 0 ]; then
            latest_checkpoint=$(ls -1t "$OUTPUT_DIR/checkpoints"/checkpoint_*.json 2>/dev/null | head -1)
            if [ -n "$latest_checkpoint" ]; then
                checkpoint_time=$(stat -c %y "$latest_checkpoint" | cut -d. -f1)
                processed=$(basename "$latest_checkpoint" .json | cut -d_ -f2)
                echo "📊 进度状态:"
                echo "   已完成检查点: $checkpoint_count 个"
                echo "   最新进度: $processed 个样本"
                echo "   最后更新: $checkpoint_time"
            fi
        fi
    fi
    
    echo ""
    
    # 输出文件
    if [ -d "$OUTPUT_DIR" ]; then
        echo "📁 输出文件:"
        ls -lah "$OUTPUT_DIR"/*.json "$OUTPUT_DIR"/*.csv "$OUTPUT_DIR"/*.md 2>/dev/null | \
        while read -r line; do
            echo "   $line"
        done
    fi
    
    echo ""
}

function show_live_log() {
    latest_log=$(ls -1t logs/14b_classification_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        echo "📄 实时日志 ($latest_log):"
        echo "==============================================="
        tail -f "$latest_log"
    else
        echo "❌ 未找到日志文件"
    fi
}

case "$1" in
    "status"|"")
        show_status
        ;;
    "log")
        show_live_log
        ;;
    "watch")
        while true; do
            clear
            show_status
            echo "🔄 30秒后自动刷新... (Ctrl+C 退出)"
            sleep 30
        done
        ;;
    "detailed-log")
        latest_log=$(ls -1t logs/14b_classification_*.log 2>/dev/null | head -1)
        if [ -n "$latest_log" ]; then
            echo "📄 最近100行日志:"
            tail -100 "$latest_log"
        else
            echo "❌ 未找到日志文件"
        fi
        ;;
    *)
        echo "用法: $0 [status|log|watch|detailed-log]"
        echo "  status      - 显示当前状态"
        echo "  log         - 实时查看日志"
        echo "  watch       - 持续监控(30秒刷新)"
        echo "  detailed-log - 查看最近100行日志"
        ;;
esac 