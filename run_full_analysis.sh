#!/bin/bash

# =============================================================================
# 知识耦合分析完整自动化脚本
# 功能：环境配置 → 数据下载 → 数据转换 → 批处理分析
# =============================================================================

set -e  # 遇到错误立即退出

# 配置参数
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
DATA_DIR="$SCRIPT_DIR/datasets"
PROCESSED_DIR="$DATA_DIR/processed"
RESULTS_DIR="$SCRIPT_DIR/results"
CHECKPOINTS_DIR="$SCRIPT_DIR/checkpoints"

# 创建必要的目录
mkdir -p "$LOG_DIR" "$DATA_DIR" "$PROCESSED_DIR" "$RESULTS_DIR" "$CHECKPOINTS_DIR"

# 日志文件
MAIN_LOG="$LOG_DIR/full_analysis_$(date +%Y%m%d_%H%M%S).log"
BATCH_LOG="$LOG_DIR/batch_processing.log"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$MAIN_LOG"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$MAIN_LOG"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$MAIN_LOG"
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $1" | tee -a "$MAIN_LOG"
}

# 显示帮助信息
show_help() {
    cat << EOF
知识耦合分析完整自动化脚本

用法: $0 [选项]

选项:
    --env-only          只配置环境，不运行分析
    --data-only         只下载和转换数据，不运行分析
    --resume            从检查点恢复批处理（默认）
    --fresh             全新开始，忽略检查点
    --batch-size SIZE   批处理大小（默认: 2000）
    --sub-batch SIZE    子批次大小（默认: 10）
    --model MODEL       模型路径（默认: meta-llama/Llama-2-7b-hf）
    --layer-start N     起始层（默认: 28）
    --layer-end N       结束层（默认: 31）
    --help              显示此帮助信息

示例:
    $0                                    # 运行完整流程
    $0 --env-only                        # 只配置环境
    $0 --batch-size 1000 --sub-batch 5   # 自定义批次大小
    $0 --fresh                           # 重新开始分析

日志位置: $LOG_DIR
EOF
}

# 解析命令行参数
ENV_ONLY=false
DATA_ONLY=false
RESUME=true
BATCH_SIZE=2000
SUB_BATCH_SIZE=10
MODEL_PATH="meta-llama/Llama-2-7b-hf"
LAYER_START=28
LAYER_END=31

while [[ $# -gt 0 ]]; do
    case $1 in
        --env-only)
            ENV_ONLY=true
            shift
            ;;
        --data-only)
            DATA_ONLY=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --fresh)
            RESUME=false
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --sub-batch)
            SUB_BATCH_SIZE="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --layer-start)
            LAYER_START="$2"
            shift 2
            ;;
        --layer-end)
            LAYER_END="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查系统要求
check_system() {
    log "🔍 检查系统要求..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        error "Python3 未安装"
        exit 1
    fi
    
    # 检查GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
        log "🎮 GPU检测: $GPU_INFO"
    else
        warn "⚠️  未检测到NVIDIA GPU"
    fi
    
    # 检查内存
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    AVAIL_MEM=$(free -g | awk '/^Mem:/{print $7}')
    log "💾 系统内存: ${TOTAL_MEM}GB 总量, ${AVAIL_MEM}GB 可用"
    
    if [ "$AVAIL_MEM" -lt 50 ]; then
        warn "⚠️  可用内存较少，可能需要调整批次大小"
    fi
}

# 配置Python环境
setup_environment() {
    log "🔧 配置Python环境..."
    
    # 检查虚拟环境
    if [ ! -d "venv" ]; then
        log "创建Python虚拟环境..."
        python3 -m venv venv
    fi
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 升级pip
    log "升级pip..."
    pip install --upgrade pip
    
    # 安装依赖
    log "安装Python依赖包..."
    cat > requirements.txt << EOF
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
psutil>=5.9.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
ipywidgets>=8.0.0
huggingface-hub>=0.17.0
accelerate>=0.23.0
EOF
    
    pip install -r requirements.txt
    
    log "✅ Python环境配置完成"
}

# 下载HotpotQA数据
download_data() {
    log "📥 下载HotpotQA数据..."
    
    cd "$DATA_DIR"
    
    # 检查是否已存在数据
    if [ -f "hotpot_train_v1.1.json" ] && [ -f "hotpot_dev_distractor_v1.json" ]; then
        log "✅ HotpotQA数据已存在，跳过下载"
        return 0
    fi
    
    # 下载训练数据
    if [ ! -f "hotpot_train_v1.1.json" ]; then
        log "下载训练集..."
        wget -c "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json" || {
            error "训练集下载失败"
            exit 1
        }
    fi
    
    # 下载验证数据
    if [ ! -f "hotpot_dev_distractor_v1.json" ]; then
        log "下载验证集..."
        wget -c "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json" || {
            error "验证集下载失败"
            exit 1
        }
    fi
    
    cd "$SCRIPT_DIR"
    log "✅ 数据下载完成"
}

# 转换数据格式
convert_data() {
    log "🔄 转换数据格式..."
    
    # 检查转换脚本
    if [ ! -f "convert_hotpot_to_coupling_format.py" ]; then
        error "转换脚本不存在: convert_hotpot_to_coupling_format.py"
        exit 1
    fi
    
    # 检查是否已转换
    if [ -f "$PROCESSED_DIR/hotpotqa_all_converted.json" ]; then
        log "✅ 转换后的数据已存在，跳过转换"
        return 0
    fi
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 运行转换
    log "开始数据转换..."
    python3 convert_hotpot_to_coupling_format.py || {
        error "数据转换失败"
        exit 1
    }
    
    log "✅ 数据转换完成"
}

# 检查必要文件
check_files() {
    log "📋 检查必要文件..."
    
    local required_files=(
        "knowledge_coupling_mvp.py"
        "knowledge_coupling_batch_processor.py"
        "$PROCESSED_DIR/hotpotqa_all_converted.json"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            error "必要文件不存在: $file"
            exit 1
        fi
    done
    
    log "✅ 文件检查完成"
}

# 启动批处理分析
start_batch_processing() {
    log "🚀 启动批处理知识耦合分析..."
    
    # 检查必要文件
    check_files
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 构建命令
    local cmd="python3 knowledge_coupling_batch_processor.py"
    cmd="$cmd --model_path '$MODEL_PATH'"
    cmd="$cmd --batch_size $BATCH_SIZE"
    cmd="$cmd --sub_batch_size $SUB_BATCH_SIZE"
    cmd="$cmd --checkpoint_dir '$CHECKPOINTS_DIR'"
    cmd="$cmd --output_dir '$RESULTS_DIR/full_hotpotqa_analysis'"
    cmd="$cmd --layer_start $LAYER_START"
    cmd="$cmd --layer_end $LAYER_END"
    
    if [ "$RESUME" = false ]; then
        cmd="$cmd --no_resume"
    fi
    
    log "执行命令: $cmd"
    log "日志文件: $BATCH_LOG"
    
    # 创建启动脚本
    cat > start_batch.sh << EOF
#!/bin/bash
cd "$SCRIPT_DIR"
source venv/bin/activate
exec $cmd 2>&1 | tee -a "$BATCH_LOG"
EOF
    chmod +x start_batch.sh
    
    # 在后台运行
    nohup ./start_batch.sh > "$BATCH_LOG" 2>&1 &
    local pid=$!
    
    # 保存PID
    echo "$pid" > "$LOG_DIR/batch_process.pid"
    
    log "✅ 批处理已在后台启动"
    log "📊 进程ID: $pid"
    log "📝 日志文件: $BATCH_LOG"
    log "🔍 监控命令: tail -f $BATCH_LOG"
    log "⏹️  停止命令: kill $pid"
    
    # 等待几秒检查是否成功启动
    sleep 5
    if kill -0 "$pid" 2>/dev/null; then
        log "🎉 批处理进程运行正常"
    else
        error "❌ 批处理进程启动失败"
        exit 1
    fi
}

# 显示状态信息
show_status() {
    log "📊 系统状态信息:"
    log "   工作目录: $SCRIPT_DIR"
    log "   数据目录: $DATA_DIR"
    log "   结果目录: $RESULTS_DIR"
    log "   检查点目录: $CHECKPOINTS_DIR"
    log "   日志目录: $LOG_DIR"
    log "   批次大小: $BATCH_SIZE"
    log "   子批次大小: $SUB_BATCH_SIZE"
    log "   模型路径: $MODEL_PATH"
    log "   层范围: $LAYER_START-$LAYER_END"
    log "   恢复模式: $RESUME"
}

# 创建监控脚本
create_monitor_script() {
    cat > monitor.sh << EOF
#!/bin/bash

# 监控脚本
LOG_DIR="$LOG_DIR"
BATCH_LOG="$BATCH_LOG"
PID_FILE="$LOG_DIR/batch_process.pid"

show_status() {
    echo "==============================================="
    echo "知识耦合分析监控面板"
    echo "==============================================="
    echo "时间: \$(date)"
    echo ""
    
    # 检查进程状态
    if [ -f "\$PID_FILE" ]; then
        PID=\$(cat "\$PID_FILE")
        if kill -0 "\$PID" 2>/dev/null; then
            echo "✅ 批处理进程运行中 (PID: \$PID)"
        else
            echo "❌ 批处理进程已停止"
        fi
    else
        echo "⚠️  未找到进程信息"
    fi
    
    # 显示资源使用
    echo ""
    echo "💾 内存使用:"
    free -h | grep -E "Mem:|Swap:"
    
    echo ""
    echo "🎮 GPU状态:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while read line; do
            echo "   GPU: \$line"
        done
    else
        echo "   未检测到GPU"
    fi
    
    # 显示最新日志
    echo ""
    echo "📝 最新日志 (最后10行):"
    echo "---------------------------------------"
    if [ -f "\$BATCH_LOG" ]; then
        tail -10 "\$BATCH_LOG"
    else
        echo "   日志文件不存在"
    fi
    echo "==============================================="
}

case "\$1" in
    status)
        show_status
        ;;
    watch)
        watch -n 30 "\$0 status"
        ;;
    log)
        tail -f "\$BATCH_LOG"
        ;;
    stop)
        if [ -f "\$PID_FILE" ]; then
            PID=\$(cat "\$PID_FILE")
            echo "停止进程 \$PID..."
            kill "\$PID"
            rm -f "\$PID_FILE"
            echo "✅ 进程已停止"
        else
            echo "⚠️  未找到运行的进程"
        fi
        ;;
    *)
        echo "用法: \$0 {status|watch|log|stop}"
        echo ""
        echo "命令说明:"
        echo "  status  - 显示当前状态"
        echo "  watch   - 持续监控状态"
        echo "  log     - 查看实时日志"
        echo "  stop    - 停止批处理进程"
        ;;
esac
EOF
    chmod +x monitor.sh
    log "✅ 监控脚本已创建: ./monitor.sh"
}

# 主流程
main() {
    log "🚀 开始知识耦合分析完整流程"
    log "==============================================="
    
    # 显示配置
    show_status
    
    # 1. 检查系统
    check_system
    
    # 2. 配置环境
    setup_environment
    
    if [ "$ENV_ONLY" = true ]; then
        log "🎉 环境配置完成！"
        exit 0
    fi
    
    # 3. 下载数据
    download_data
    
    # 4. 转换数据
    convert_data
    
    if [ "$DATA_ONLY" = true ]; then
        log "🎉 数据准备完成！"
        exit 0
    fi
    
    # 5. 创建监控脚本
    create_monitor_script
    
    # 6. 启动批处理
    start_batch_processing
    
    log "🎉 完整流程启动完成！"
    log ""
    log "📋 后续操作:"
    log "   查看状态: ./monitor.sh status"
    log "   实时监控: ./monitor.sh watch"
    log "   查看日志: ./monitor.sh log"
    log "   停止处理: ./monitor.sh stop"
    log ""
    log "📁 重要目录:"
    log "   结果目录: $RESULTS_DIR"
    log "   日志目录: $LOG_DIR"
    log "   检查点目录: $CHECKPOINTS_DIR"
}

# 捕获中断信号
trap 'error "脚本被中断"; exit 1' INT TERM

# 运行主流程
main "$@" 