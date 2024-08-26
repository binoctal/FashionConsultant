#coding=utf-8
import os
from typing import Dict, Iterator, List, Optional, Union, Any

from modelscope_agent.llm.base import BaseChatModel, register_llm
from modelscope_agent.utils.logger import agent_logger as logger
from modelscope_agent.utils.retry import retry
from modelscope_agent.utils.tokenization_utils import count_tokens
import requests

from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import json
import datetime
import torch

from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

# 源大模型下载
from modelscope import snapshot_download

model_name: str = "Yuan2-2B-Mars-hf"  # 模型名称

model_dir = snapshot_download(f'IEITYuan/{model_name}', cache_dir='./bigmodel')

# 设置设备参数
# DEVICE = "cuda"  # 使用CUDA
# DEVICE_ID = "0"  # CUDA设备ID，如果未设置则为空
# CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合CUDA设备信息

CUDA_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 确定设备
device = torch.device(CUDA_DEVICE)

AutoModelForCausalLM
path = f'./bigmodel/IEITYuan/{model_name}' 
model = None
tokenizer = None

image_dir = 'images'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
    os.makedirs(image_dir + '/initial')
    os.makedirs(image_dir + '/final')
    os.makedirs('db')
    os.makedirs('data')

# 加载预训练的分词器和模型
print("Creat tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(path, legacy=True, use_fast=False)
#tokenizer = AutoTokenizer.from_pretrained(path, torch_dtype="auto", device_map="auto", legacy=True, use_fast=False)
tokenizer.add_tokens(['<|im_start|>', '<|im_end|>', '<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

print("Creat model...")
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

@register_llm('yuan')
class YuanLLM(BaseChatModel):  
    context_window: int = 8192  # 上下文窗口大小
    num_output: int = 1000  # 输出的token数量
    _support_fn_call = None

    # 清理GPU内存函数
    @staticmethod
    def torch_gc():
        if torch.cuda.is_available():  # 检查是否可用CUDA
            with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
                torch.cuda.empty_cache()  # 清空CUDA缓存
                torch.cuda.ipc_collect()  # 收集CUDA内存碎片

    def __init__(self,
                 model: str,
                 model_server: str,
                 is_chat: bool = True,
                 is_function_call: Optional[bool] = None,
                 support_stream: Optional[bool] = None,
                 **kwargs):

        # 调用 BaseChatModel 的构造函数
        super().__init__(model, model_server)
    
        self.client = None
        self.is_chat = is_chat
        self.support_stream = support_stream 

    def _chat_stream(self,
                     messages: List[Dict],
                     stop: Optional[List[str]] = None,
                     **kwargs) -> Iterator[str]:
        return None

    def _chat_no_stream(self,
                        messages: List[Dict],
                        stop: Optional[List[str]] = None,
                        **kwargs) -> str:
        return None

    def support_function_calling(self):
        return False

    def support_raw_prompt(self) -> bool:
        return True
    
    def build_raw_prompt(self, messages: list):
        prompt = ''
        # messages.append({'role': 'assistant', 'content': ''})
        im_start = '<|im_start|>'
        im_end = '<|im_end|>'
        if messages[0]['role'] == 'system':
            sys = messages[0]['content']
            system_prompt = f'{im_start}system:\n{sys}{im_end}'
        else:
            system_prompt = f'{im_start}system\nYou are a helpful assistant.{im_end}'

        used_length = count_tokens(system_prompt)

        for message in reversed(messages):
            if message['role'] == 'user':
                query = message['content'].lstrip('\n').rstrip()
                local_prompt = f'\n{im_start}user:\n{query}{im_end}'
            elif message['role'] == 'assistant':
                response = message['content'].lstrip('\n').rstrip()
                local_prompt = f'\n{im_start}assistant\n{response}{im_end}'

            if message['role'] != 'system':
                cur_content_length = count_tokens(local_prompt)
                if used_length + cur_content_length > self.max_length:
                    break
                used_length += cur_content_length
                prompt = local_prompt + prompt

        prompt = system_prompt + prompt

        # add one empty reply for the last round of assistant
        # ensure the end of prompt is assistant
        if not prompt.endswith(f'\n{im_start}assistant\n{im_end}'):
            prompt += f'\n{im_start}assistant\n{im_end}'
        prompt = prompt[:-len(f'{im_end}')]
        return prompt

    @retry(max_retries=3, delay_seconds=0.5)
    def chat(self,
             prompt: Optional[str] = None,
             messages: Optional[List[Dict]] = None,
             stop: Optional[List[str]] = None,
             stream: bool = False,
             **kwargs) -> Union[str, Iterator[str]]:

        prompt += "<sep>"
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt")

        # 移动输入张量到与模型相同的设备
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 如果模型的`pad_token_id`与`eos_token_id`相同，则需要手动设置`attention_mask`
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            # 创建一个与输入长度相同的mask数组，1表示有效位置，0表示需要忽略的位置
            attention_mask = (inputs["input_ids"] != tokenizer.pad_token_id).long()

            # 更新inputs字典
            inputs['attention_mask'] = attention_mask
        else:
            # 如果`pad_token_id`与`eos_token_id`不同，则不需要手动设置`attention_mask`
            pass  # `tokenizer()` 默认已经处理了`attention_mask`

        # 使用模型进行生成
        outputs = model.generate(**inputs, max_length=4000, num_return_sequences=1)
        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output = tokenizer.decode(outputs[0])
        response = output.split("<sep>")[-1]
        response = response.split("<eod>")[0]

        YuanLLM.torch_gc()  # 执行GPU内存清理

        print('[xin]: yuanllm response: ', response)
        
        return response  # 返回响应

    def _out_generator(self, response):
        for chunk in response:
            if hasattr(chunk.choices[0], 'text'):
                yield chunk.choices[0].text

    def chat_with_raw_prompt(self,
                             prompt: str,
                             stream: bool = True,
                             **kwargs) -> str:
        return None

    def chat_with_functions(self,
                            messages: List[Dict],
                            functions: Optional[List[Dict]] = None,
                            **kwargs) -> Dict:
        return None

class YuanLammaLLM(CustomLLM):
    context_window: int = 8192  # 上下文窗口大小
    num_output: int = 1000  # 输出的token数量

    # 清理GPU内存函数
    @staticmethod
    def torch_gc():
        if torch.cuda.is_available():  # 检查是否可用CUDA
            with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
                torch.cuda.empty_cache()  # 清空CUDA缓存
                torch.cuda.ipc_collect()  # 收集CUDA内存碎片

    def __init__(self):
        super().__init__()
        
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        # 得到LLM的元数据
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=model_name,
        )

    @llm_completion_callback()  # 回调函数
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # 完成函数
        # print("完成函数")

        pdb.set_trace()

        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt")

        # 移动输入张量到与模型相同的设备
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 如果模型的`pad_token_id`与`eos_token_id`相同，则需要手动设置`attention_mask`
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            # 创建一个与输入长度相同的mask数组，1表示有效位置，0表示需要忽略的位置
            attention_mask = (inputs["input_ids"] != tokenizer.pad_token_id).long()

            # 更新inputs字典
            inputs['attention_mask'] = attention_mask
        else:
            # 如果`pad_token_id`与`eos_token_id`不同，则不需要手动设置`attention_mask`
            pass  # `tokenizer()` 默认已经处理了`attention_mask`

        # 获取 input_ids 和 attention_mask
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        outputs = model.generate(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      max_length=self.num_output,  # 输出的最大长度
                                      num_return_sequences=1,  # 返回序列的数量
                                      no_repeat_ngram_size=2,  # 不重复 n-gram 的大小
        )

        response = tokenizer.decode(outputs[0])

        YuanLammaLLM.torch_gc()

        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        # 流式完成函数
        print("流式完成函数")

        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt")

        # 移动输入张量到与模型相同的设备
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 如果模型的`pad_token_id`与`eos_token_id`相同，则需要手动设置`attention_mask`
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            # 创建一个与输入长度相同的mask数组，1表示有效位置，0表示需要忽略的位置
            attention_mask = (inputs["input_ids"] != tokenizer.pad_token_id).long()

            # 更新inputs字典
            inputs['attention_mask'] = attention_mask
        else:
            # 如果`pad_token_id`与`eos_token_id`不同，则不需要手动设置`attention_mask`
            pass  # `tokenizer()` 默认已经处理了`attention_mask`

        outputs = model.generate(inputs, max_length=self.num_output)
        response = tokenizer.decode(outputs[0])

        YuanLammaLLM.torch_gc()
        
        for token in response:
            yield CompletionResponse(text=token, delta=token)
