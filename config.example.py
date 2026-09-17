# LLM API config (OpenAI-compatible endpoint)
# 复制本文件为 config.py 并填入你自己的密钥后再运行。
LLM_CONFIG = {
    "api_key": "YOUR_DASHSCOPE_API_KEY",
    "model_name": "qwen3.8-max",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "temperature": 0.7,
    "max_tokens": 8192,
}

# Mem0 API config (Long-term memory)
MEM0_CONFIG = {
    "api_key": "",
}
