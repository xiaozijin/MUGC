import os
import json
import requests
import openai
import traceback
import time
import timeout_decorator
from tqdm import tqdm
from multiprocessing import Process as mp
from multiprocessing import Pool
import argparse
import re
from openai import AzureOpenAI


TIMEOUT = 10000


deployment_name='captioneval'
client = AzureOpenAI(
  azure_endpoint = "https://openai.azure.com/", 
  api_key="your_api_key",  
  api_version="2023-07-01-preview"
)



def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def call_chatgpt_azure(caption1,caption2):
    prompt = '''
    为您提供两个视频的caption。gold caption是标准答案，已解析为JSON格式，test caption是生成的结果,已解析为JSON格式。现在，我们需要参考标准答案，并对测试标题的准确性进行评分。请按照以下步骤进行评估。比较两个JSON中描述的backgroud、object,culture之间的相似性，在score属性下给出background,object和culture之间的分数，并给出相似的总分数，总分数为20%的background得分加上百分之60的object得分加百分之20的cultrue得分，注意分数区间在0-100，此外为每个分数提供简要的评分原因，解释 test caption 与 gold caption 的差异点，score属性字段中只需要有四个分数，此外不需要把任何JSON放进去,所有分数最高分数均为100。\n \
    gold caption的json如下:\n ''' + caption1 + ''' \n
    test caption如下:\n ''' + caption2 + '''\n
    score输出格式:\n \nbackground_score:[background_score here]\n object_score: [object_score here]\n  culture_score: [culture_score here]\n total_score: [total_score here]\n \
    explain输出格式：\n \nbackground:[简要说明原因]\n object:[简要说明原因]\n culture:[简要说明原因]\n total:[简要说明原因]\n
    score属性字段中只需要有四个分数，此外不需要把任何JSON放进去
    explain属性字段只需要有1-2句话的打分原因分析
    '''
    retries = 0
    max_retries = 5
    retry_delay = 2

    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content  

        except Exception as e:
            print(f"Error occurred: {e}")
            
            if "429" in str(e):
                print(f"Rate limit exceeded. Retrying after {retry_delay} seconds...")
            else:
                print(f"An error occurred. Retrying after {retry_delay} seconds...")

            retries += 1  
            time.sleep(retry_delay)  

    print("Max retries exceeded. Exiting...")
    return None  

def save_progress(result, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


def load_progress(save_path):
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def process_caption_pair(ref_item, gen_item):
    ref_caption = ref_item['caption']
    gen_response = gen_item['response']
    ref_response = ref_item['response']
    score_response = call_chatgpt_azure(ref_response, gen_response)
    return {
        "ref_caption": ref_caption,
        "ref_response": ref_response,
        "gen_response": gen_response,
        "response": score_response
    }
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  
    return data

def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

class APIStatusError(Exception):
    def __init__(self, response, body):
        self.response = response
        self.body = body
        super().__init__(f"APIStatusError: {response}, {body}")

    def __reduce__(self):
        return (self.__class__, (self.response, self.body))




def main():
    result = []
    ref_data = read_jsonl('/share/project/MCCU/datasets/video/ref.jsonl')
    gen_data = read_jsonl('/share/project/MCCU/datasets/image/extract/format/IU/xcomposer2-vl.jsonl')
    save_path = "/share/project/MCCU/datasets/image/score/xcomposer_vl-all/score_undo.json"

    file_mappings_image_IU = [
        ("Aquila-vl-2b.jsonl","Aquila"),
        ("Cogvlm-llama3-Chinese.jsonl","cogvlm-llama3-all"),
        ("Deepseek-vl-7b.jsonl","deepseek-all"),
        ("Emu3.jsonl","Emu3"),
        ("glm4v.jsonl","glm-all"),
        ("InternVL2-8B.jsonl","InternVL2-8B"),
        ("llava-mistral.jsonl","llava-mistral-all"),
        ("llava-next.jsonl","llava-next-all"),
        ("MiniCPMv2.6.jsonl","MiniCPM-all"),
        ("Molmo.jsonl","Molmo"),
        ("monkey-chat.jsonl","monkey-chat-all"),
        ("mplug.jsonl","mplug-all"),
        ("mplug3.jsonl","mplug3-all"),
        ("qwen-vl-chat.jsonl","qwen-all"),
        ("qwen2_vl.jsonl","qwen2-vl"),
        ("xcomposer2-vl.jsonl", "xcomposer_vl-all"),

    ]
    file_mappings_image_T2I = [
        ("dream-like-photoreal","dreamlike-photoreal-all"),
        ("Emu3","Emu3-image"),
        ("flux","flux"),
        ("Kandinsky3","Kandinsky3"),
        ("openjourney","openjourney-all"),
        ("sdv1.4","sdv1.4-all"),
        ("sdv1.5","sdv1.5-all"),
        ("sdv3.5","sdv3.5"),
        ("sdxl-lightning","sdxl-lightning"),
        ("sdxl-turbo","sdxl-turbo"),
        ("sdxl","sdxl"),

    ]
    file_mappings_image_VU = [
        ("llava-next-video","llava-next-video-all"),
        ("MiniCPM-video","MiniCPM-all"),
        ("mplug-video","mplug-all"),
        ("video-llama2","video-llama2-all"),
    ]
    file_mappings_video_T2V = [
        ("video_Dream Machine","Dream Machine-all"),
        ("video_Gen2","Gen2"),
        ("video_hailuo","hailuo"),
        ("video_kling","kling"),
        ("video_Pika","Pika"),
        ("video_PixVerse","PixVerse"),
        ("video_tongyi","tongyi"),
        ("video_VIDU","VIDU"),
        ("video_zhipu","zhipu"),

    ]
    # 遍历文件映射并处理
    for input_file, output_folder in file_mappings_image_VU:
        gen_data = read_jsonl(f"/share/project/MCCU/datasets/video/extract/format/VU/{input_file}.jsonl")
        save_path = f"/share/project/MCCU/datasets/video/score/{output_folder}/score_undo.json"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print("开始处理:",{output_folder})
        
        completed_results = load_progress(save_path)
        completed_count = len(completed_results)

        remaining_ref_data = ref_data[completed_count:]
        remaining_gen_data = gen_data[completed_count:]


        with Pool(processes=5,maxtasksperchild=10) as pool:  
            for i, result in enumerate(pool.starmap(process_caption_pair, zip(remaining_ref_data, remaining_gen_data))):
                completed_results.append(result)
                save_progress(completed_results, save_path)
                print(f'已处理 {i + 1} / {len(remaining_ref_data)} 个caption pair')

        print("全部处理完成")

    
if __name__ == '__main__':
    main()
