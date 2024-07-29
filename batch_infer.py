import os
import re
import pandas as pd
from pynvml import (nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, nvmlShutdown)
from config.common_config import *
import time
model_loaded = False
project_change = False
model_change = False
last_adapter_name = ''
last_model_name = ''


def get_available_gpu(threshold=20000):
    # Initialize NVML
    nvmlInit()
    # Get the number of GPU devices
    device_count = nvmlDeviceGetCount()

    # Find GPU devices with available memory greater than the threshold
    available_gpus = []
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        free_memory_mb = info.free / 1024 / 1024

        if free_memory_mb > threshold:
            available_gpus.append(i)

    # Shutdown NVML
    nvmlShutdown()
    # available_gpus = ['0']

    return available_gpus


def get_free_memory():
    nvmlInit()
    # Get the number of GPU devices
    device_count = nvmlDeviceGetCount()

    # Find GPU devices with available memory greater than the threshold
    free_memory_gpus = []
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        free_memory_mb = info.free / 1024 / 1024
        free_memory_gpus.append(free_memory_mb)

    # Shutdown NVML
    nvmlShutdown()
    return free_memory_gpus

def process_single_choice(value, value_domain):
            return value if value in value_domain else "未提及"

def process_multi_choice(values, value_domain):
    # Split the comma-separated string into a list, process it, and then join back into a string
    values_list = values.split(',')
    processed_values = [value for value in values_list if value in value_domain]
    return ','.join(processed_values)

class LLMPredict(object):
    """自动保存最新模型
    """
    def __init__(self,model_name,adapter_name):
        
        self.model = self.load_model(model_name,adapter_name)

    def load_model(self, model_name, adapter_name):
        global model_loaded, model, tokenizer, project_change, last_adapter_name, last_model_name, model_change
        current_directory = os.getcwd()
        model_file_name = llm_model_dict[model_name]['name']
        model_path = llm_model_dict[model_name]['model_path']
        template = llm_model_dict[model_name]['template']
        lora_target = llm_model_dict[model_name]['lora_target']
        quantization_bit = llm_model_dict[model_name]['quantization_bit'] if 'quantization_bit' in llm_model_dict[model_name] else None
        if adapter_name != last_adapter_name:
            project_change = True
        if model_name != last_model_name:
            model_change = True
        if not model_loaded or project_change or model_change:
            if model_loaded:
                model.model = model.model.to('cpu')
                del model
                import torch
                torch.cuda.empty_cache()
                model_loaded = False
            available_gpus = get_available_gpu(threshold=8000)
            if len(available_gpus)>0:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus[0])
                print('available_gpus[0]',available_gpus[0])
            else:
                return 'no enough GPU, please try it later!',''
            try:
                from src.llamafactory.chat.chat_model import ChatModel
                args = {
                    "model_name_or_path": model_path,
                    "template": template,
                    "finetuning_type": "lora",
                    # "lora_target": lora_target,
                    "adapter_name_or_path": f"{current_directory}/finetune/{model_file_name}/checkpoints/{adapter_name}",
                    "max_length":max_length,
                    "do_sample":do_sample,
                    "temperature":temperature,
                }
                if quantization_bit is not None:
                    quant_dict = {"quantization_bit":quantization_bit,
                    "quantization_method": "bitsandbytes"
                }
                    args.update(quant_dict)
                model = ChatModel(args)
                model_loaded = True
                last_adapter_name = adapter_name
                project_change = False
                last_model_name = model_name
                model_change = False
                # return model, tokenizer
            except Exception as e:
                print('error!! ', e)
                return e,''
        return model

    def clear_torch_cache(self):
        gc.collect()
        if self.llm_device.lower() != "cpu":
            if torch.has_mps:
                try:
                    from torch.mps import empty_cache
                    empty_cache()
                except Exception as e:
                    print(e)
                    print(
                        "如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。")
            elif torch.has_cuda:
                device_id = "0" if torch.cuda.is_available() else None
                CUDA_DEVICE = f"{self.llm_device}:{device_id}" if device_id else self.llm_device
                with torch.cuda.device(CUDA_DEVICE):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            else:
                print("未检测到 cuda 或 mps，暂不支持清理显存")


    def pred_res(self,Instruction,Input):
        answer = self.model.batch_chat([Instruction],[Instruction])
        return answer, 0
    
    # 硬控输出结果
    def get_result(self,project_name,field_en,field,context):
        prompt = ''
        
        medical_logic = pd.read_excel('./config/medical_logic.xlsx', sheet_name=project_name)
        row = medical_logic.loc[medical_logic['字段英文名称'] == field_en].to_dict('records')[0]
        # field = row['字段名'] if '字段名' in row and pd.notnull(row['字段名']) else field
        special_requirement = row['特殊要求'] if '特殊要求' in row and pd.notnull(row['特殊要求']) else ''
        if row['值域类型'] == '多选':
            prompt = f"""##结构化任务##根据下文中信息，判断{field}是什么？请在值域【{row['值域']}】中选择提到的所有内容。{special_requirement}"""
        elif row['值域类型'] == '单选':
            prompt = f"##结构化任务##根据下文中信息，判断{field}是什么？请在值域【{row['值域']}】中选择1个。{special_requirement}"
        elif row['值域类型'] == '提取':
            row['字段名'] = row['字段名'].replace('大小1','大小')
            prompt = f"""##结构化任务##根据下文中信息，判断{field}是什么？请提取文中对应的值。{special_requirement}"""
        else:
            return ''
        
        print('prompt',prompt)
        start_time = time.time()
        res, token_num = self.pred_res(prompt,context) #if self.model != 'no enough GPU' else 'no enough GPU',0
        end_time = time.time()
        print('token_num',token_num)
        infer_time = round(end_time-start_time,6)
        per_token = infer_time/token_num
        print('model infer time~~',infer_time)
        print('per_token',per_token)
        print('res',res)
        if row['值域类型']== '单选':
            processed_result = process_single_choice(res, row['值域'])
        elif row['值域类型'] == '多选':
            processed_result = process_multi_choice(res, row['值域'])
        else:
            processed_result = res
        return res, token_num


    # 硬控输出结果
    def get_batch_result(self, data_list):
        free_memory_gpus = get_free_memory()
        if free_memory_gpus[0] < 256 * len(data_list):
            import torch
            print('剩余空间', free_memory_gpus[0])
            print('清除缓存')
            torch.cuda.empty_cache()
        
        start_time = time.time()
        project_name = data_list[0]["project"]
        medical_logic = pd.read_excel('./config/medical_logic.xlsx', sheet_name=project_name)
        prompts, contexts = [], []
        for data_dict in data_list:
            project_name,field_en,field,context = data_dict["project"],data_dict["field_en"],data_dict["field"],data_dict["raw_text"]
            row = medical_logic.loc[medical_logic['字段英文名称'] == field_en].to_dict('records')[0]
            # field = row['字段名'] if '字段名' in row and pd.notnull(row['字段名']) else field
            special_requirement = row['特殊要求'] if '特殊要求' in row and pd.notnull(row['特殊要求']) else ''
            if row['值域类型'] == '多选':
                prompt = f"""##结构化任务##根据下文中信息，判断{field}是什么？请在值域【{row['值域']}】中选择提到的所有内容。{special_requirement}"""
            elif row['值域类型'] == '单选':
                prompt = f"##结构化任务##根据下文中信息，判断{field}是什么？请在值域【{row['值域']}】中选择1个。{special_requirement}"
            elif row['值域类型'] == '提取':
                row['字段名'] = row['字段名'].replace('大小1','大小')
                prompt = f"""##结构化任务##根据下文中信息，判断{field}是什么？请提取文中对应的值。{special_requirement}"""
            else:
                return ''
            print('prompt',prompt)
            prompts.append(prompt)
            contexts.append(context)
        res, token_num = self.pred_batch_res(prompts,contexts) # if self.model != 'no enough GPU' else 'no enough GPU',0
        end_time = time.time()
        infer_time = round(end_time - start_time, 6)
        per_token = infer_time / token_num
        print('model infer time~~', infer_time)
        print('per_token', per_token)
        print('res',res)

        # Process the results based on value domain type
        processed_results = []
        for i, row in enumerate(data_list):
            value_domain_type = medical_logic.loc[medical_logic['字段英文名称'] == row["field_en"], "值域类型"].values[0]
            value_range =  medical_logic.loc[medical_logic['字段英文名称'] == row["field_en"], "值域"].values[0]
            if value_domain_type == '单选':
                processed_result = process_single_choice(res[i], value_range)
            elif value_domain_type == '多选':
                processed_result = process_multi_choice(res[i], value_range)
            else:
                processed_result = res[i]

            processed_results.append(processed_result)

        return processed_results, token_num


    def pred_batch_res(self,Instructions,Inputs):
        batch_messages = []
        for i in range(len(Instructions)):
            messages = [{"role": 'user', "content": Instructions[i]+'\n\n'+Inputs[i]}]
            batch_messages.append(messages)
        responses = self.model.batch_chat(batch_messages)
        answers = [response.response_text for response in responses]
        tokens= [response.response_length+response.prompt_length for response in responses]
        token_sum = sum(tokens)
        return answers,token_sum

if __name__ == '__main__':
    fd_pred = LLMPredict("Llama3","mirror_lake_4bit")
    for i in range(15):
        start_time = time.time()
        data_list = [{
                "raw_text": """患者父親有前列腺癌史""",
                "field": "疾病名稱",
                "project": "澳门镜湖",
                "field_en": "FDNAM"
            },
            {
                "raw_text": "右側輸尿管結石術後反复血尿1月",
                "field": "症狀名稱",
                "project": "澳门镜湖",
                "field_en": "CCPSNAM"
            },
            {
                "raw_text": "月經婚育史(年齡&lt;=55y): LMP:2023/3/14,已婚未育.",
                "field": "末次月經日期",
                "project": "澳门镜湖",
                "field_en": "MOLMPDTC"
            }]
        res = fd_pred.get_batch_result(data_list)
        print('res',res)
        end_time = time.time()
        print('cost time:', round(end_time-start_time, 2))
