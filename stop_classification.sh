#!/bin/bash

# 停止14B分类任务脚本

PID_FILE="logs/14b_classification.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    
    if ps -p $PID > /dev/null; then
        echo "🛑 发现运行中的分类任务 (PID: $PID)"
        echo "🔄 正在优雅停止..."
        
        # 发送SIGINT信号，让程序优雅停止并保存检查点
        kill -SIGINT $PID
        
        # 等待最多30秒
        for i in {1..30}; do
            if ! ps -p $PID > /dev/null; then
                echo "✅ 任务已优雅停止，检查点已保存"
                rm -f "$PID_FILE"
                exit 0
            fi
            echo "   等待停止... ($i/30)"
            sleep 1
        done
        
        # 如果30秒后还没停止，强制停止
        echo "⚠️ 优雅停止超时，强制终止..."
        kill -SIGTERM $PID
        sleep 2
        
        if ps -p $PID > /dev/null; then
            echo "🔥 强制杀死进程..."
            kill -SIGKILL $PID
        fi
        
        rm -f "$PID_FILE"
        echo "✅ 任务已停止"
    else
        echo "❌ PID文件存在但进程不在运行"
        rm -f "$PID_FILE"
    fi
else
    echo "❌ 未找到PID文件，可能没有运行中的任务"
fi 