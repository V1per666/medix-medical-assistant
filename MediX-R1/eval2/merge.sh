#!/bin/bash
# merge.sh - 合并 FSDP 分片 checkpoint 为 HuggingFace 格式
#
# 将 training/checkpoints/medix-r1_2b_dapo/global_step_600/actor/ 下的
# 分片权重合并到 huggingface/ 子目录，供 vLLM 加载推理。
#
# 使用方式: bash merge.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ACTOR_DIR="${PROJECT_ROOT}/MediX-R1/training/checkpoints/medix-r1_2b_dapo/global_step_600/actor"
HF_DIR="${ACTOR_DIR}/huggingface"

# 检查是否已经合并过
if ls "$HF_DIR"/*.safetensors 1>/dev/null 2>&1; then
    echo "模型权重已存在于 $HF_DIR，跳过合并。"
    echo "如需重新合并，请先删除 $HF_DIR 下的 .safetensors 文件。"
    exit 0
fi

echo "开始合并模型权重..."
echo "  checkpoint 路径: $ACTOR_DIR"
echo "  输出路径: $HF_DIR"

cd "${PROJECT_ROOT}/MediX-R1/training"
python3 scripts/model_merger.py --local_dir "$ACTOR_DIR"

echo ""
echo "合并完成！模型已保存至: $HF_DIR"
