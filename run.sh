#!/bin/bash
# Agent 运行脚本
# 从 config.yaml 读取配置并设置环境变量

# 配置文件路径
CONFIG_FILE="config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

# 使用 Python 解析 YAML 并导出环境变量和参数
eval $(python3 << 'PYTHON_SCRIPT'
import yaml
import sys
import shlex

try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 导出环境变量
    api = config.get('api', {})
    if api.get('zhipu_api_key'):
        print(f"export ZHIPU_API_KEY='{api.get('zhipu_api_key')}'")
    if api.get('zhipu_api_base'):
        print(f"export ZHIPU_API_BASE='{api.get('zhipu_api_base')}'")
    if api.get('zhipu_model'):
        print(f"export ZHIPU_MODEL='{api.get('zhipu_model')}'")
    if api.get('serper_api_key'):
        print(f"export SERPER_API_KEY='{api.get('serper_api_key')}'")
    
    # 导出代理配置
    proxy = config.get('proxy', {})
    if proxy.get('http_proxy'):
        print(f"export HTTP_PROXY='{proxy.get('http_proxy')}'")
    if proxy.get('https_proxy'):
        print(f"export HTTPS_PROXY='{proxy.get('https_proxy')}'")
    
    # 构建命令行参数
    data = config.get('data', {})
    model = config.get('model', {})
    
    args = []
    args.append("--src_file")
    args.append(data.get('src_file', 'hotpotqa_10.jsonl'))
    args.append("--start_sample")
    args.append(str(data.get('start_sample', 0)))
    args.append("--end_sample")
    args.append(str(data.get('end_sample', 100000)))
    # args.append("--max_samples")
    # args.append(str(data.get('max_samples', 0)))
    args.append("--model_path")
    args.append(model.get('model_path', 'Qwen/Qwen2.5-7B-Instruct'))
    args.append("--prompt_type")
    args.append(model.get('prompt_type', 'v3'))
    args.append("--temp")
    args.append(str(model.get('temp', 0.0)))
    args.append("--gpu_id")
    args.append(model.get('gpu_id', '0'))
    # args.append("--gpu_memory_rate")
    # args.append(str(model.get('gpu_memory_rate', 0.95)))
    
    # 输出参数（用特殊标记）
    print("ARGS=" + shlex.quote(' '.join(args)))
    
    print("✓ 配置加载完成", file=sys.stderr)
    
except ImportError:
    print("错误: 需要安装 PyYAML: pip install pyyaml", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"错误: 解析配置文件失败: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_SCRIPT
)

if [ $? -ne 0 ]; then
    exit 1
fi

# 检查必要的 API Key
if [ -z "$ZHIPU_API_KEY" ]; then
    echo "⚠ 警告: ZHIPU_API_KEY 未设置"
fi

if [ -z "$SERPER_API_KEY" ]; then
    echo "⚠ 警告: SERPER_API_KEY 未设置，检索功能将无法使用"
fi

# 运行主程序
echo ""
echo "=========================================="
echo "开始运行 Agent..."
echo "配置文件: $CONFIG_FILE"
echo "运行参数: $ARGS"
echo "=========================================="
echo ""

python3 search_new.py $ARGS

