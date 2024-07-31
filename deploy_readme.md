# 结构化模型部署说明

# 准备基本的模型文件，lora及配置文件
目前的目录结构是这样的，需要config，finetune,models放在某个文件夹中。config里面是配置文件，finetune里面是lora，models里面是需要用到的模型文件，注意模型文件名要和这俩匹配，还有个离线的docker镜像包llm_structure_image.tar（这个只要导入好就行，放哪里都行）。

```commandline
config
  ├── common_config.py
  ├── gunicorn_config.py
  ├── medical_logic.xlsx
finetune
  ├── chatglm2/
  │   └── checkpoints/
  │       └── mirror_lake_4bit_2
  ├── llama3/
  │   └── checkpoints/
  │       └── mirror_lake_4bit_2
  ├── llm_utils.py
models
  ├── chatglm2-6b
  └── Meta-Llama-3-8B-Instruct
llm_structure_image.tar
```

目前应该除了models里的模型文件是已经下载好传输过去了，其他的都可以在10.151.35.6下面：/zy路径下找到，并下载，包括一个docker离线镜像文件llm_structure_image.tar
把这些下载好到目标服务器上，按上面的形式放在一个文件夹中，比如/deploy

# 准备docker环境

## 通过docker pull的形式拉镜像
```
docker pull june666666/llm_structure_code:structure_20240730
```

## 通过docker镜像离线导入的方式
提前获取离线安装包llm_structure_image.tar，然后在目标服务器上导入
```
docker load -i llm_structure_image.tar
```

# 准备启动环境
注意CUDA需要是11.8以上的
## 先看看配置
检查一下config/commom_config.py 里面的两个模型的model_path，是不是/code/models/{模型名字}。然后default_model可以切换两个不同的模型，目前按默认的就行

## 参考启动命令
下面是参考启动命令，挂载的本地路径，/zy/...这些需要换成目标服务器对应文件的地址
PS：如果要部署两个卡，按以前的方法，每个卡上起一个docker?这个具体看以前怎么配置了
```
docker run -itd --gpus=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all -v /zy/config:/code/config -v /zy/models:/code/models -v /zy/finetune:/code/finetune --name llm_structure_code -p 46000:46000 june666666/llm_structure_code:structure_20240730
```