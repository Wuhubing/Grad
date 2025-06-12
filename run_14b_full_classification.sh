#!/bin/bash

# 14B模型全量高耦合对分类后台运行脚本
# 使用方式: ./run_14b_full_classification.sh [sample_size]

set -e

# 默认参数
SAMPLE_SIZE=${1:-5000}  # 默认5000，如果想全量用34924
MODEL="Qwen/Qwen2.5-14B-Instruct"
BATCH_SIZE=8
CHECKPOINT_INTERVAL=25

# 创建输出目录
OUTPUT_DIR="results/optimized_high_coupling_classification"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 设置日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/14b_classification_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/14b_classification.pid"

echo "🚀 启动14B模型全量分类 (后台运行)"
echo "📋 参数配置:"
echo "   样本大小: $SAMPLE_SIZE"
echo "   模型: $MODEL"
echo "   批处理大小: $BATCH_SIZE"
echo "   检查点间隔: $CHECKPOINT_INTERVAL"
echo "   日志文件: $LOG_FILE"
echo "   PID文件: $PID_FILE"

# 检查GPU状态
echo ""
echo "🎮 GPU状态检查:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

# 预估完成时间
if [ "$SAMPLE_SIZE" -eq 34924 ]; then
    echo ""
    echo "⏰ 全量分析预估:"
    echo "   总样本: 34,924"
    echo "   预估时间: ~22小时"
    echo "   预计完成: $(date -d '+22 hours' '+%Y-%m-%d %H:%M:%S')"
elif [ "$SAMPLE_SIZE" -eq 5000 ]; then
    echo ""
    echo "⏰ 中等规模分析预估:"
    echo "   总样本: 5,000"
    echo "   预估时间: ~3.2小时"
    echo "   预计完成: $(date -d '+3.2 hours' '+%Y-%m-%d %H:%M:%S')"
fi

echo ""
echo "🔄 启动分类任务..."

# 后台运行Python脚本
nohup python real_high_coupling_llm_classifier_optimized.py \
    --model "$MODEL" \
    --sample-size "$SAMPLE_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    > "$LOG_FILE" 2>&1 &

# 保存PID
PYTHON_PID=$!
echo $PYTHON_PID > "$PID_FILE"

echo "✅ 分类任务已启动 (PID: $PYTHON_PID)"
echo "📄 实时日志: tail -f $LOG_FILE"
echo "🛑 停止任务: kill $PYTHON_PID 或 ./stop_classification.sh"

# 等待几秒确保任务正常启动
sleep 5

if ps -p $PYTHON_PID > /dev/null; then
    echo "🎉 任务正常运行中..."
    echo ""
    echo "💡 监控命令:"
    echo "   查看实时日志: tail -f $LOG_FILE"
    echo "   监控GPU状态: watch -n 30 nvidia-smi"
    echo "   检查进程状态: ps -p $PYTHON_PID"
    echo "   查看输出文件: ls -la $OUTPUT_DIR/"
    echo ""
    echo "📊 结果文件将保存在: $OUTPUT_DIR/"
else
    echo "❌ 任务启动失败，请检查日志: $LOG_FILE"
    exit 1
fi 