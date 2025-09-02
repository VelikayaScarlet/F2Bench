# filename: load_model.py
from utils import *
from zhipuai import ZhipuAI
from openai import OpenAI
import os
import torch
import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams

GLM4_API_KEY = "your_API_KEY"
DS_API_KEY = "your_API_KEY"
GPT_API_KEY = "your_API_KEY"

def Loader(model_name):
    model_dir = ""
    if model_name == 'llama2-7b':
        model_dir = "path/to/llama2-7b/" 
    elif model_name == 'llama2-13b':
        model_dir = "path/to/llama2-13b/" 
    elif model_name == 'gemma2-2b':
        model_dir = "path/to/gemma2-2b/" 
    elif model_name == 'gemma2-9b':
        model_dir = "path/to/gemma2-9b/" 
    elif model_name == 'qwen2_5-0.5b':
        model_dir = "path/to/qwen2_5-0_5b/"
    elif model_name == 'qwen2_5-1.5b':
        model_dir = "path/to/Qwen2__5_1__5B/"
    elif model_name == 'qwen2_5-7b':
        model_dir = "path/to/qwen2_5-7b/" 
    elif model_name == 'qwen2_5-32b':
        model_dir = "path/to/qwen2_5-32b/"
    elif model_name == 'gpt':
        print("This is a Model from API, which can't be load by vLLM.")
        return None
    else:
        print("Cannot find the model called " + model_name)
        return None

    try:
        llm = LLM(model=model_dir,
                  dtype="bfloat16",
                  gpu_memory_utilization=0.8,
                #   quantization="fp8",
                 )
        return llm
    except Exception as e:
        print(f"Failed to load model from local path: {model_dir} with Exception: {e}")
        return None

def inference_vllm(llm, prompt):
    assert llm is not None, "llm_model is None! Loader failed."
    from vllm import LLM
    assert isinstance(llm, LLM), f"llm_model is not a vLLM LLM instance: {type(llm)}"

    # sampling_params = SamplingParams(max_new_tokens=512)
    sampling_params = SamplingParams(max_tokens=256+len(prompt.split(" ")))

    outputs = llm.generate([prompt], sampling_params,use_tqdm=False)
    response = outputs[0].outputs[0].text
    return response.strip()


def glm4(prompt):
    client = ZhipuAI(api_key=GLM4_API_KEY)
    try:
        response = client.chat.completions.create(
            model="glm-4-flash", 
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
        )
        response_content = response.choices[0].message.content
        return response_content
    except Exception as e:
        if hasattr(e, 'response') and e.response.status_code == 400:
            print(f"警告: 检测到敏感内容，跳过当前请求。错误信息: {e.response.text}")
            return "-0-"

def deepseek(prompt):
    client = OpenAI(api_key=DS_API_KEY, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        max_tokens=512,
    )
    return response.choices[0].message.content

def gpt(prompt):
    client = OpenAI(
        base_url='https://xiaoai.plus/v1', 
        api_key=GPT_API_KEY
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return response.choices[0].message.content