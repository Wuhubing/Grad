#!/bin/bash

# =============================================================================
# 知识耦合分析完整自动化脚本
# 功能：环境配置 → 数据下载 → 数据转换 → 批处理分析
# 版本：v2.0 - 增强版
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
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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

success() {
    echo -e "${CYAN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1" | tee -a "$MAIN_LOG"
}

# 显示帮助信息
show_help() {
    cat << EOF
${CYAN}知识耦合分析完整自动化脚本 v2.0${NC}

用法: $0 [选项]

${YELLOW}基本选项:${NC}
    --env-only          只配置环境，不运行分析
    --data-only         只下载和转换数据，不运行分析
    --resume            从检查点恢复批处理（默认）
    --fresh             全新开始，忽略检查点
    --help              显示此帮助信息

${YELLOW}分析参数:${NC}
    --batch-size SIZE   批处理大小（默认: 2000）
    --sub-batch SIZE    子批次大小（默认: 10）
    --model MODEL       模型路径（默认: meta-llama/Llama-2-7b-hf）
    --layer-start N     起始层（默认: 28）
    --layer-end N       结束层（默认: 31）

${YELLOW}高级选项:${NC}
    --hf-token TOKEN    HuggingFace访问令牌
    --no-gpu-check      跳过GPU内存检查
    --debug             启用调试模式
    --dry-run           只显示将要执行的命令，不实际运行

${YELLOW}示例:${NC}
    $0                                              # 运行完整流程
    $0 --env-only                                  # 只配置环境
    $0 --batch-size 1000 --sub-batch 5             # 自定义批次大小
    $0 --fresh --hf-token hf_xxx                   # 重新开始并指定HF token
    $0 --layer-start 24 --layer-end 27             # 自定义层范围

${YELLOW}监控命令:${NC}
    ./monitor.sh status    # 查看状态
    ./monitor.sh watch     # 实时监控
    ./monitor.sh log       # 查看日志
    ./monitor.sh stop      # 停止处理

${YELLOW}文件位置:${NC}
    日志: $LOG_DIR
    数据: $DATA_DIR
    结果: $RESULTS_DIR
    检查点: $CHECKPOINTS_DIR
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
HF_TOKEN=""
SKIP_GPU_CHECK=false
DEBUG_MODE=false
DRY_RUN=false

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
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --no-gpu-check)
            SKIP_GPU_CHECK=true
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
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

# 调试模式设置
if [ "$DEBUG_MODE" = true ]; then
    set -x
    log "🐛 调试模式已启用"
fi

# 检查HuggingFace Token
check_hf_token() {
    log "🔑 检查HuggingFace访问令牌..."
    
    # 优先级：命令行参数 > 环境变量 > 已保存的token
    if [ -n "$HF_TOKEN" ]; then
        log "✅ 使用命令行提供的HF token"
        export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
    elif [ -n "$HUGGINGFACE_HUB_TOKEN" ]; then
        log "✅ 使用环境变量中的HF token"
        HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
    elif command -v huggingface-cli &> /dev/null; then
        if huggingface-cli whoami &> /dev/null; then
            log "✅ 使用已登录的HF账户"
        else
            warn "⚠️  未检测到HuggingFace token，某些模型可能无法下载"
            warn "   请使用 --hf-token 参数提供token或运行: huggingface-cli login"
        fi
    else
        warn "⚠️  HuggingFace CLI未安装，无法验证token状态"
    fi
}

# 检查GPU详细信息
check_gpu_detailed() {
    if [ "$SKIP_GPU_CHECK" = true ]; then
        warn "⚠️  跳过GPU检查（--no-gpu-check）"
        return 0
    fi
    
    log "🎮 检查GPU详细信息..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        error "❌ nvidia-smi 未找到，请安装NVIDIA驱动"
        return 1
    fi
    
    # GPU基本信息
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    log "📊 检测到 $GPU_COUNT 个GPU"
    
    # 详细GPU信息
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | while IFS=',' read -r idx name memory_total memory_free util temp; do
        memory_total=$(echo $memory_total | xargs)
        memory_free=$(echo $memory_free | xargs)
        memory_used=$((memory_total - memory_free))
        memory_usage_pct=$((memory_used * 100 / memory_total))
        
        log "  GPU$idx: $(echo $name | xargs)"
        log "    内存: ${memory_used}MB/${memory_total}MB (${memory_usage_pct}%)"
        log "    利用率: $(echo $util | xargs)%  温度: $(echo $temp | xargs)°C"
        
        # 内存检查
        if [ "$memory_free" -lt 20000 ]; then
            warn "    ⚠️  GPU$idx 可用内存不足20GB，可能影响LLaMA2-7B运行"
        fi
        
        # 温度检查
        if [ "$(echo $temp | xargs)" -gt 80 ]; then
            warn "    ⚠️  GPU$idx 温度较高: $(echo $temp | xargs)°C"
        fi
    done
    
    # CUDA版本检查
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
        log "🔧 CUDA版本: $CUDA_VERSION"
    else
        warn "⚠️  nvcc未找到，无法确定CUDA版本"
    fi
}

# 检查系统要求
check_system() {
    log "🔍 检查系统要求..."
    
    # 检查Python版本
    if ! command -v python3 &> /dev/null; then
        error "Python3 未安装"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    log "🐍 Python版本: $PYTHON_VERSION"
    
    # 检查最低Python版本（3.8+）
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        log "✅ Python版本检查通过"
    else
        error "❌ 需要Python 3.8或更高版本"
        exit 1
    fi
    
    # 检查磁盘空间
    DISK_AVAIL=$(df -BG "$SCRIPT_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
    log "💿 可用磁盘空间: ${DISK_AVAIL}GB"
    
    if [ "$DISK_AVAIL" -lt 50 ]; then
        warn "⚠️  可用磁盘空间不足50GB，可能不够存储模型和结果"
    fi
    
    # GPU检查
    check_gpu_detailed
    
    # 检查内存
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    AVAIL_MEM=$(free -g | awk '/^Mem:/{print $7}')
    USED_MEM=$((TOTAL_MEM - AVAIL_MEM))
    MEM_USAGE_PCT=$((USED_MEM * 100 / TOTAL_MEM))
    
    log "💾 系统内存: ${USED_MEM}GB/${TOTAL_MEM}GB (${MEM_USAGE_PCT}%使用)"
    
    if [ "$AVAIL_MEM" -lt 50 ]; then
        warn "⚠️  可用内存不足50GB，建议调整批次大小"
        info "💡 建议: --batch-size 1000 --sub-batch 5"
    fi
    
    # 检查交换空间
    SWAP_TOTAL=$(free -g | awk '/^Swap:/{print $2}')
    SWAP_USED=$(free -g | awk '/^Swap:/{print $3}')
    if [ "$SWAP_TOTAL" -gt 0 ]; then
        log "🔄 交换空间: ${SWAP_USED}GB/${SWAP_TOTAL}GB"
        if [ "$SWAP_USED" -gt $((SWAP_TOTAL / 2)) ]; then
            warn "⚠️  交换空间使用过多，可能影响性能"
        fi
    fi
}

# 配置Python环境
setup_environment() {
    log "🔧 配置Python环境..."
    
    if [ "$DRY_RUN" = true ]; then
        info "💡 [DRY RUN] 将会配置Python虚拟环境和依赖"
        return 0
    fi
    
    # 检查虚拟环境
    if [ ! -d "venv" ]; then
        log "创建Python虚拟环境..."
        python3 -m venv venv || {
            error "虚拟环境创建失败"
            exit 1
        }
    fi
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 升级pip
    log "升级pip..."
    pip install --upgrade pip --quiet
    
    # 检查requirements.txt
    if [ ! -f "requirements.txt" ]; then
        warn "⚠️  requirements.txt不存在，将创建基本版本"
        create_basic_requirements
    fi
    
    # 安装依赖
    log "安装Python依赖包..."
    pip install -r requirements.txt --quiet || {
        error "依赖安装失败"
        exit 1
    }
    
    # 配置HuggingFace
    if [ -n "$HF_TOKEN" ]; then
        log "配置HuggingFace CLI..."
        echo "$HF_TOKEN" | huggingface-cli login --token || {
            warn "⚠️  HuggingFace CLI登录失败"
        }
    fi
    
    success "✅ Python环境配置完成"
}

# 创建基本requirements.txt
create_basic_requirements() {
    cat > requirements.txt << 'EOF'
# 核心深度学习框架
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.23.0

# 数据处理
datasets>=2.14.0
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0

# 模型相关（LLaMA2支持）
sentencepiece>=0.1.99
protobuf>=3.20.0
tokenizers>=0.14.0

# 科学计算和机器学习
scikit-learn>=1.3.0
scipy>=1.10.0

# 可视化
matplotlib>=3.7.0
seaborn>=0.12.0

# 系统监控
psutil>=5.9.0
gpustat>=1.1.0

# HuggingFace相关
huggingface-hub>=0.17.0
safetensors>=0.3.0
EOF
}

# 下载HotpotQA数据
download_data() {
    log "📥 下载HotpotQA数据..."
    
    if [ "$DRY_RUN" = true ]; then
        info "💡 [DRY RUN] 将会下载HotpotQA训练集和验证集"
        return 0
    fi
    
    cd "$DATA_DIR"
    
    # 检查是否已存在数据
    local train_file="hotpot_train_v1.1.json"
    local dev_file="hotpot_dev_distractor_v1.json"
    
    if [ -f "$train_file" ] && [ -f "$dev_file" ]; then
        # 验证文件完整性
        local train_size=$(stat -c%s "$train_file" 2>/dev/null || echo 0)
        local dev_size=$(stat -c%s "$dev_file" 2>/dev/null || echo 0)
        
        if [ "$train_size" -gt 500000000 ] && [ "$dev_size" -gt 40000000 ]; then
            success "✅ HotpotQA数据已存在且完整，跳过下载"
            cd "$SCRIPT_DIR"
            return 0
        else
            warn "⚠️  检测到不完整的数据文件，重新下载"
        fi
    fi
    
    # 下载训练数据
    if [ ! -f "$train_file" ] || [ "$(stat -c%s "$train_file" 2>/dev/null || echo 0)" -lt 500000000 ]; then
        log "下载训练集..."
        wget -c --timeout=30 --tries=3 --progress=bar \
            "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json" || {
            error "训练集下载失败"
            cd "$SCRIPT_DIR"
            exit 1
        }
    fi
    
    # 下载验证数据
    if [ ! -f "$dev_file" ] || [ "$(stat -c%s "$dev_file" 2>/dev/null || echo 0)" -lt 40000000 ]; then
        log "下载验证集..."
        wget -c --timeout=30 --tries=3 --progress=bar \
            "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json" || {
            error "验证集下载失败"
            cd "$SCRIPT_DIR"
            exit 1
        }
    fi
    
    # 验证下载完成
    log "验证下载文件..."
    for file in "$train_file" "$dev_file"; do
        if [ -f "$file" ]; then
            local size_mb=$(($(stat -c%s "$file") / 1024 / 1024))
            log "  ✅ $file: ${size_mb}MB"
        else
            error "  ❌ $file: 文件不存在"
            exit 1
        fi
    done
    
    cd "$SCRIPT_DIR"
    success "✅ 数据下载完成"
}

# 转换数据格式
convert_data() {
    log "🔄 转换数据格式..."
    
    if [ "$DRY_RUN" = true ]; then
        info "💡 [DRY RUN] 将会转换HotpotQA数据格式"
        return 0
    fi
    
    # 检查转换脚本
    if [ ! -f "convert_hotpot_to_coupling_format.py" ]; then
        error "转换脚本不存在: convert_hotpot_to_coupling_format.py"
        exit 1
    fi
    
    # 检查是否已转换
    local converted_file="$PROCESSED_DIR/hotpotqa_all_converted.json"
    if [ -f "$converted_file" ]; then
        local size_mb=$(($(stat -c%s "$converted_file") / 1024 / 1024))
        if [ "$size_mb" -gt 500 ]; then
            success "✅ 转换后的数据已存在且完整，跳过转换"
            return 0
        else
            warn "⚠️  检测到不完整的转换文件，重新转换"
        fi
    fi
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 运行转换
    log "开始数据转换..."
    python3 -c "
from convert_hotpot_to_coupling_format import HotpotQAConverter
converter = HotpotQAConverter(data_dir='$DATA_DIR', output_dir='$PROCESSED_DIR')
all_data = converter.convert_all_splits()
print(f'转换完成：{len(all_data)} 个样本')
" || {
        error "数据转换失败"
        exit 1
    }
    
    # 验证转换结果
    if [ -f "$converted_file" ]; then
        local size_mb=$(($(stat -c%s "$converted_file") / 1024 / 1024))
        success "✅ 数据转换完成: ${size_mb}MB"
    else
        error "❌ 转换后文件不存在"
        exit 1
    fi
}

# 检查必要文件
check_files() {
    log "📋 检查必要文件..."
    
    local required_files=(
        "knowledge_coupling_mvp.py"
        "knowledge_coupling_batch_processor.py"
        "$PROCESSED_DIR/hotpotqa_all_converted.json"
    )
    
    local missing_files=()
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        else
            local size_mb=$(($(stat -c%s "$file") / 1024 / 1024))
            log "  ✅ $file (${size_mb}MB)"
        fi
    done
    
    if [ ${#missing_files[@]} -gt 0 ]; then
        error "缺少必要文件:"
        for file in "${missing_files[@]}"; do
            error "  ❌ $file"
        done
        exit 1
    fi
    
    success "✅ 文件检查完成"
}

# 启动批处理分析
start_batch_processing() {
    log "🚀 启动批处理知识耦合分析..."
    
    if [ "$DRY_RUN" = true ]; then
        info "💡 [DRY RUN] 将会启动批处理分析:"
        info "    模型: $MODEL_PATH"
        info "    批次大小: $BATCH_SIZE"
        info "    子批次大小: $SUB_BATCH_SIZE"
        info "    层范围: $LAYER_START-$LAYER_END"
        info "    输出目录: $RESULTS_DIR/full_hotpotqa_analysis"
        return 0
    fi
    
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

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$SCRIPT_DIR:\$PYTHONPATH"

# 如果有HF token，设置环境变量
if [ -n "$HF_TOKEN" ]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

# 运行命令
exec $cmd 2>&1 | tee -a "$BATCH_LOG"
EOF
    chmod +x start_batch.sh
    
    # 在后台运行
    nohup ./start_batch.sh > "$BATCH_LOG" 2>&1 &
    local pid=$!
    
    # 保存PID
    echo "$pid" > "$LOG_DIR/batch_process.pid"
    
    success "✅ 批处理已在后台启动"
    log "📊 进程ID: $pid"
    log "📝 日志文件: $BATCH_LOG"
    log "🔍 监控命令: ./monitor.sh watch"
    log "⏹️  停止命令: ./monitor.sh stop"
    
    # 等待几秒检查是否成功启动
    sleep 5
    if kill -0 "$pid" 2>/dev/null; then
        success "🎉 批处理进程运行正常"
        
        # 显示前几行日志
        log "📋 最新日志片段:"
        tail -5 "$BATCH_LOG" | while read line; do
            info "    $line"
        done
    else
        error "❌ 批处理进程启动失败"
        error "📋 错误日志:"
        tail -10 "$BATCH_LOG" | while read line; do
            error "    $line"
        done
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
    
    if [ -n "$HF_TOKEN" ]; then
        log "   HF Token: 已配置"
    else
        log "   HF Token: 未配置"
    fi
    
    if [ "$DEBUG_MODE" = true ]; then
        log "   调试模式: 已启用"
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log "   试运行模式: 已启用"
    fi
}

# 创建增强监控脚本
create_monitor_script() {
    cat > monitor.sh << 'EOF'
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
EOF
    chmod +x monitor.sh
    success "✅ 增强监控脚本已创建: ./monitor.sh"
}

# 主流程
main() {
    log "🚀 开始知识耦合分析完整流程"
    log "==============================================="
    
    # 显示配置
    show_status
    
    if [ "$DRY_RUN" = true ]; then
        log "💡 试运行模式：仅显示将要执行的操作"
    fi
    
    # 1. 检查系统
    check_system
    
    # 2. 检查HF token
    check_hf_token
    
    # 3. 配置环境
    setup_environment
    
    if [ "$ENV_ONLY" = true ]; then
        success "🎉 环境配置完成！"
        exit 0
    fi
    
    # 4. 下载数据
    download_data
    
    # 5. 转换数据
    convert_data
    
    if [ "$DATA_ONLY" = true ]; then
        success "🎉 数据准备完成！"
        log "📁 数据位置: $PROCESSED_DIR"
        exit 0
    fi
    
    # 6. 创建监控脚本
    create_monitor_script
    
    # 7. 启动批处理
    start_batch_processing
    
    success "🎉 完整流程启动完成！"
    log ""
    log "📋 后续操作:"
    log "   查看状态: ./monitor.sh status"
    log "   实时监控: ./monitor.sh watch"
    log "   查看日志: ./monitor.sh log"
    log "   详细日志: ./monitor.sh detailed-log"
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