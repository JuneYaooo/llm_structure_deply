
llm_model_dict = {
"ChatGLM2": {"name": "chatglm2",
        "model_path": "/code/models/chatglm2-6b",
        "template":"chatglm2",
        "lora_target":"query_key_value",
        "per_device_train_batch_size":4,
        "quantization_bit":4
    },
"Llama3": {"name": "llama3",
        "model_path": "/code/models/Meta-Llama-3-8B-Instruct",
        "template":"llama3",
        "lora_target":"all",
        "per_device_train_batch_size":1,
        "quantization_bit":4
    },
}

# 找到 profile.d/conda.sh 文件的绝对路径，填进来
conda_env_file = '/etc/profile.d/conda.sh'

# 生成参数
max_length=1500
do_sample=True
temperature=0

default_model = "ChatGLM2"
default_adaptor = "mirror_lake_4bit_2"