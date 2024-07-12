CONSTANT_YAML = """
  alias:
    - 7b-4bit
  project: vllm-chat
  service_config:
    name: qwen2
    traffic:
      timeout: 300
    resources:
      gpu: 1
      gpu_type: nvidia-rtx-3060
  engine_config:
    model: model/
    max_model_len: 2048
    quantization: awq
"""
