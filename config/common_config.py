
llm_model_dict = {
# "PULSE": {"name": "pulse",
#         "model_path": "/mnt/data/smart_health_02/yaoyujun/models/PULSE",
#         "template":"pulse",
#         "lora_target":"query_key_value",
#         "per_device_train_batch_size":2
#     },
# "PULSE-20B": {"name": "pulse_20b",
#         "model_path": "/mnt/data/smart_health_02/yaoyujun/models/PULSE-20b",
#         "template":"pulse",
#         "lora_target":"q_proj,v_proj",
#         "per_device_train_batch_size":2,
#         "quantization_bit":4
#     },
# "InternLM": {"name": "internlm",
#         "model_path": "/mnt/data/smart_health_02/yaoyujun/models/internlm-7b",
#         "template":"intern",
#         "lora_target":"q_proj,v_proj",
#         "per_device_train_batch_size":2,
#         "quantization_bit":4
#     },
"ChatGLM2": {"name": "chatglm2",
        "model_path": "/ailab/public/pjlab-smarthealth03/maokangkun/models/hf/chatglm2-6b",
        "template":"chatglm2",
        "lora_target":"query_key_value",
        "per_device_train_batch_size":4,
        "quantization_bit":4
    },
"ChatGLM3": {"name": "chatglm3",
        "model_path": "/ailab/user/yaoyujun/models/hf/chatglm3-6b",
        "template":"chatglm3",
        "lora_target":"query_key_value",
        "per_device_train_batch_size":4,
        "quantization_bit":4
    },
# "Baichuan2": {"name": "baichuan2",
#     "model_path": "/mnt/data/smart_health_02/yaoyujun/models/Baichuan2-7B-Base",
#     "template":"baichuan2",
#     "lora_target":"W_pack",
#     "per_device_train_batch_size":2
# },
"Llama3": {"name": "llama3",
        "model_path": "/ailab/public/pjlab-smarthealth03/maokangkun/models/hf/Meta-Llama-3-8B-Instruct",
        "template":"llama3",
        "lora_target":"all",
        "per_device_train_batch_size":1,
        "quantization_bit":4
    },
"Llama3.1": {"name": "llama3.1",
        "model_path": "/ailab/public/pjlab-smarthealth03/maokangkun/models/hf/Meta-Llama-3.1-8B-Instruct",
        "template":"llama3",
        "lora_target":"all",
        "per_device_train_batch_size":1,
        "quantization_bit":4
    },
"GLM4": {"name": "glm4",
        "model_path": "/ailab/public/pjlab-smarthealth03/maokangkun/models/hf/glm-4-9b-chat-1m",
        "template":"glm4",
        "lora_target":"all",
        "per_device_train_batch_size":1,
        "quantization_bit":4
    },
"Qwen2": {"name": "qwen2",
        "model_path": "/ailab/public/pjlab-smarthealth03/maokangkun/models/hf/Qwen2-7B-Instruct",
        "template":"qwen",
        "lora_target":"all",
        "per_device_train_batch_size":1,
        "quantization_bit":4
    },
"Qwen2-0.5B": {"name": "qwen2_0.5",
        "model_path": "/ailab/public/pjlab-smarthealth03/maokangkun/models/hf/Qwen2-0.5B-Instruct",
        "template":"qwen",
        "lora_target":"all",
        "per_device_train_batch_size":4,
        # "quantization_bit":4
    },
"MINICPM-2B": {"name": "minicpm2b",
        "model_path": "/ailab/public/pjlab-smarthealth03/maokangkun/models/hf/MiniCPM-2B-sft-bf16",
        "template":"cpm",
        "lora_target":"all",
        "per_device_train_batch_size":4,
        # "quantization_bit":4
    },
}

# 找到 profile.d/conda.sh 文件的绝对路径，填进来
conda_env_file = '/ailab/apps/anaconda/2022.10/etc/profile.d/conda.sh'

# 生成参数
max_length=1500
do_sample=True
temperature=0.1

default_model = "Llama3"
default_adaptor = "mirror_lake_4bit"