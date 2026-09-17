#!/bin/bash
# eval.sh - 评估 medix-r1_2b_dapo checkpoint 在 rad_vqa 上的表现
#
# 使用方式:
#   1. 先合并模型权重 (如果还没合并过): bash merge.sh
#   2. 运行评估: bash eval.sh
#
# rollout 模型: MediX-R1/training/checkpoints/medix-r1_2b_dapo/global_step_600 (本地 vLLM)
# 评分模型: config.py 中配置的 API 模型 (doubao)
# 评测任务: rad_vqa

set -e

# 环境变量
export HF_HUB_READ_TIMEOUT=600
export HF_HUB_CONNECTION_TIMEOUT=1200
export OMP_NUM_THREADS=5
export CUDA_VISIBLE_DEVICES=4
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# 取消 HF 镜像，直接访问官方 (避免 hf-mirror.com 代理超时)
unset HF_ENDPOINT

# 注意: 请确保已激活 conda 环境 (conda activate swarm7)

# 项目根目录 (eval2 的上两级)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 合并后的模型路径
MODEL_PATH="${PROJECT_ROOT}/MediX-R1/training/checkpoints/medix-r1_2b_dapo/global_step_600/actor/huggingface"

# 检查模型权重是否已合并
if ! ls "$MODEL_PATH"/*.safetensors 1>/dev/null 2>&1; then
    echo "错误: 模型权重尚未合并！"
    echo "请先运行: bash merge.sh"
    echo "路径: $MODEL_PATH 中没有找到 .safetensors 文件"
    exit 1
fi

echo "=========================================="
echo "评估配置:"
echo "  rollout 模型: $MODEL_PATH"
echo "  评分模型: config.py 中的 API 模型"
echo "  评测任务: rad_vqa"
echo "=========================================="

# 运行评估 (生成 + 评分 + 汇总)
python eval.py \
    --model "$MODEL_PATH" \
    --tasks rad_vqa \
    --num_workers 1 \
    --generate true \
    --evaluate true \
    --score true \
    --tensor_parallel_size 1 \
    --judge_server configapi

if [ $? -ne 0 ]; then
    echo ""
    echo "评估失败，请检查上方错误信息。"
    exit 1
fi

echo ""
echo "=========================================="
echo "评估完成！结果保存在 results/ 目录下。"
echo "=========================================="
