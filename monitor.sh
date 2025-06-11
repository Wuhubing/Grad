#!/bin/bash

# 增强监控脚本 v2.0
LOG_DIR="logs"
BATCH_LOG="logs/batch_processing.log"
PID_FILE="logs/batch_process.pid"
RESULTS_DIR="results"
CHECKPOINTS_DIR="checkpoints"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 获取进程状态
get_process_status() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "running:$PID"
        else
            echo "stopped:$PID"
        fi
    else
        echo "none:0"
    fi
}

# 显示状态
show_status() {
    clear
    echo -e "${CYAN}===============================================${NC}"
    echo -e "${CYAN}     知识耦合分析监控面板 v2.0${NC}"
    echo -e "${CYAN}===============================================${NC}"
    echo -e "${BLUE}时间: $(date)${NC}"
    echo ""
    
    # 检查进程状态
    STATUS=$(get_process_status)
    STATUS_TYPE=$(echo $STATUS | cut -d: -f1)
    PID=$(echo $STATUS | cut -d: -f2)
    
    case $STATUS_TYPE in
        running)
            echo -e "${GREEN}✅ 批处理进程运行中${NC} (PID: $PID)"
            # 显示运行时间
            if [ -n "$PID" ] && [ "$PID" != "0" ]; then
                ELAPSED=$(ps -o etime= -p $PID 2>/dev/null | xargs)
                echo -e "${BLUE}   运行时间: $ELAPSED${NC}"
            fi
            ;;
        stopped)
            echo -e "${RED}❌ 批处理进程已停止${NC} (最后PID: $PID)"
            ;;
        none)
            echo -e "${YELLOW}⚠️  未找到进程信息${NC}"
            ;;
    esac
    
    # 显示资源使用
    echo ""
    echo -e "${PURPLE}💾 内存使用:${NC}"
    free -h | grep -E "Mem:|Swap:" | while read line; do
        echo -e "${BLUE}   $line${NC}"
    done
    
    echo ""
    echo -e "${PURPLE}🎮 GPU状态:${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while IFS=',' read -r idx name util mem_used mem_total temp; do
            mem_used=$(echo $mem_used | xargs)
            mem_total=$(echo $mem_total | xargs)
            util=$(echo $util | xargs)
            temp=$(echo $temp | xargs)
            mem_pct=$((mem_used * 100 / mem_total))
            echo -e "${BLUE}   GPU$idx: $(echo $name | xargs)${NC}"
            echo -e "${BLUE}     利用率: ${util}%  内存: ${mem_used}MB/${mem_total}MB (${mem_pct}%)  温度: ${temp}°C${NC}"
        done
    else
        echo -e "${YELLOW}   未检测到GPU${NC}"
    fi
    
    # 显示进度信息（从检查点）
    echo ""
    echo -e "${PURPLE}📊 处理进度:${NC}"
    if [ -d "$CHECKPOINTS_DIR" ]; then
        CHECKPOINT_COUNT=$(find "$CHECKPOINTS_DIR" -name "*.json" 2>/dev/null | wc -l)
        echo -e "${BLUE}   已完成批次: $CHECKPOINT_COUNT${NC}"
        
        # 显示最新检查点
        LATEST_CHECKPOINT=$(find "$CHECKPOINTS_DIR" -name "*.json" -type f -exec stat --format='%Y %n' {} \; 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
        if [ -n "$LATEST_CHECKPOINT" ]; then
            CHECKPOINT_NAME=$(basename "$LATEST_CHECKPOINT" .json)
            CHECKPOINT_TIME=$(stat -c %y "$LATEST_CHECKPOINT" 2>/dev/null | cut -d. -f1)
            echo -e "${BLUE}   最新检查点: $CHECKPOINT_NAME${NC}"
            echo -e "${BLUE}   更新时间: $CHECKPOINT_TIME${NC}"
        fi
    else
        echo -e "${YELLOW}   检查点目录不存在${NC}"
    fi
    
    # 显示结果文件
    echo ""
    echo -e "${PURPLE}📁 输出文件:${NC}"
    if [ -d "$RESULTS_DIR" ]; then
        RESULT_COUNT=$(find "$RESULTS_DIR" -name "*.json" -o -name "*.csv" -o -name "*.npy" 2>/dev/null | wc -l)
        echo -e "${BLUE}   结果文件数: $RESULT_COUNT${NC}"
        
        # 显示最新结果文件大小
        find "$RESULTS_DIR" -type f \( -name "*.json" -o -name "*.csv" -o -name "*.npy" \) -exec ls -lh {} \; 2>/dev/null | tail -3 | while read -r line; do
            echo -e "${BLUE}   $line${NC}"
        done
    else
        echo -e "${YELLOW}   结果目录不存在${NC}"
    fi
    
    # 显示最新日志
    echo ""
    echo -e "${PURPLE}📝 最新日志 (最后5行):${NC}"
    echo -e "${CYAN}---------------------------------------${NC}"
    if [ -f "$BATCH_LOG" ]; then
        tail -5 "$BATCH_LOG" | while read line; do
            echo -e "${BLUE}$line${NC}"
        done
    else
        echo -e "${YELLOW}   日志文件不存在${NC}"
    fi
    echo -e "${CYAN}===============================================${NC}"
}

# 显示详细日志
show_detailed_log() {
    if [ -f "$BATCH_LOG" ]; then
        echo -e "${CYAN}========== 详细日志 (最后50行) ==========${NC}"
        tail -50 "$BATCH_LOG"
        echo -e "${CYAN}=======================================${NC}"
    else
        echo -e "${RED}日志文件不存在: $BATCH_LOG${NC}"
    fi
}

# 停止进程
stop_process() {
    STATUS=$(get_process_status)
    STATUS_TYPE=$(echo $STATUS | cut -d: -f1)
    PID=$(echo $STATUS | cut -d: -f2)
    
    if [ "$STATUS_TYPE" = "running" ]; then
        echo -e "${YELLOW}正在停止进程 $PID...${NC}"
        kill "$PID"
        sleep 3
        
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "${YELLOW}强制停止进程 $PID...${NC}"
            kill -9 "$PID"
            sleep 1
        fi
        
        if ! kill -0 "$PID" 2>/dev/null; then
            echo -e "${GREEN}✅ 进程已停止${NC}"
            rm -f "$PID_FILE"
        else
            echo -e "${RED}❌ 无法停止进程${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  没有运行的进程${NC}"
    fi
}

# 主逻辑
case "$1" in
    status)
        show_status
        ;;
    watch)
        while true; do
            show_status
            sleep 10
        done
        ;;
    log)
        tail -f "$BATCH_LOG"
        ;;
    detailed-log)
        show_detailed_log
        ;;
    stop)
        stop_process
        ;;
    *)
        echo -e "${CYAN}知识耦合分析监控脚本 v2.0${NC}"
        echo ""
        echo "用法: $0 {status|watch|log|detailed-log|stop}"
        echo ""
        echo -e "${YELLOW}命令说明:${NC}"
        echo "  status       - 显示当前状态"
        echo "  watch        - 持续监控状态（每10秒刷新）"
        echo "  log          - 查看实时日志"
        echo "  detailed-log - 显示详细日志（最后50行）"
        echo "  stop         - 停止批处理进程"
        echo ""
        echo -e "${YELLOW}快捷键（在watch模式下）:${NC}"
        echo "  Ctrl+C       - 退出监控"
        ;;
esac
