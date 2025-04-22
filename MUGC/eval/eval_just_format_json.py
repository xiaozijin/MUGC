# 比较图片与图片是否对应正确
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
# openai.api_type = "azure"
# openai.api_base = "https://baaisailing-ce.openai.azure.com/"
# openai.api_version = "2023-07-01-preview"
# openai.api_key = '4c933ac1b1be461287d5f7bfe45041c2'

# openai.api_type = "azure"
# openai.api_base = "https://baaisailing-ae.openai.azure.com/"
# openai.api_version = "2023-07-01-preview"
# openai.api_key = 'ef58de5b8cca4916a8662761ba3fc830'

# openai.api_type = "azure"
# openai.api_base = "https://baaimrnd-ae.openai.azure.com/"
# openai.api_version = "2023-07-01-preview"
# openai.api_key = 'cd156f683dfc47078bbf54305337a0eb'

# openai.api_type = "azure"
# openai.api_base = "https://baaiaquila.openai.azure.com/"
# openai.api_version = "2023-07-01-preview"
# openai.api_key = '314c94b92ba04eda88ad930d51c1ae69'

deployment_name='captioneval'
client = AzureOpenAI(
  azure_endpoint = "https://baaisailing-ae.openai.azure.com/", 
  api_key="ef58de5b8cca4916a8662761ba3fc830",  
  api_version="2023-07-01-preview"
)



def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

#给出background,object和text之间的分数，给出两者中国风元素间的分数
#background_score:[background_score here]\n object_score: [object_score here]\n text_score: [text_score here]\n culture_score: [culture_score here]
def call_chatgpt_azure(caption1,caption2):
    # prompt = '''
    # 为您提供两个视频的caption。gold caption是标准答案，已解析为JSON格式，test caption是生成的结果。现在，我们需要参考标准答案，并对测试标题的准确性进行评分。请按照以下步骤进行评估。首先，请从test caption的描述中提取backgroud、object和text部分，形成JSON。JSON字段包括：对backgroud、object和text。object是指标题中的对象，这些对象需要包括但不限于标题中描述的以下属性，如外观、动作、位置。请随时添加任何相关信息。如果没有关于某些属性的描述，则相关字段将填充为空白。backgroud是指出现在标题中的背景，text是指标题中双引号内的文本信息（如果没有双引号，则填写为空白）。\ntest caption转成JSON文件后，可以比较两个JSON中描述的backgroud、object和text之间的相似性，并给出测试标题的分数，最高分数为100。\n \
    # {
    #     "objects": [
    #         {
    #             "name": "[name here]",
    #             "features": {
    #                     "feature1": "[feature1 here]",
    #                     "feature2": "[feature2 here]",
    #                     ......
    #             }
    #         }
    #     ],
    #     "background": "[background here]",
    #     "text": "[text here]" 
    # }\n
    # gold caption的json如下:\n ''' + caption1 + ''' \n
    # test caption如下:\n ''' + caption2 + '''\n
    # 分数输出格式:\n score: [score here]\n \
    # 直接生成json和score，不要输出分析
    # '''
    prompt = '''
    为您提供两个视频的对应caption。gold caption是标准答案，已解析为JSON格式，test caption是生成的结果。现需要从test caption的描述中提取backgroud、object和text部分，形成JSON。JSON字段包括：对backgroud、object和text。object是指标题中的对象，这些对象需要包括但不限于标题中描述的以下属性，如外观、动作、位置。请随时添加任何相关信息。如果没有关于某些属性的描述，则相关字段将填充为空白。backgroud是指出现在标题中的背景，text是指标题中双引号内的文本信息（如果没有双引号，则填写为空白）。\
    {
        "objects": [
            {
                "name": "[name here]",
                "features": {
                        "feature1": "[feature1 here]",
                        "feature2": "[feature2 here]",
                        ......
                }
            }
        ],
        "background": "[background here]",
        "text": "[text here]" 
    }\n
    gold caption的json如下:\n ''' + caption1 + ''' \n
    test caption如下:\n ''' + caption2 + '''\n
    请输出并保存格式化后的内容
    '''
        # 直接生成json和每项score，不要输出分析
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
    gen_caption = gen_item['caption']
    score_response = call_chatgpt_azure(ref_caption, gen_caption)
    return {
        "response": score_response
    }
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 使用 json.load() 读取整个 JSON 文件
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
    ref_data = read_jsonl('/share/project/zpf/code/MCCU/datasets/image/ref.jsonl')
    gen_data = read_json('/share/project/zpf/code/MCCU/datasets/image/First-inference/sdxl_clean_caption.json')
    save_path = "/share/project/zpf/code/MCCU/datasets/image/format/sdxl.json"
    completed_results = load_progress(save_path)
    completed_count = len(completed_results)

    # 剩余需要处理的pair
    remaining_ref_data = ref_data[completed_count:]
    remaining_gen_data = gen_data[completed_count:]


    with Pool(processes=5,maxtasksperchild=10) as pool:  # 使用多进程池，4个并发进程
        for i, result in enumerate(pool.starmap(process_caption_pair, zip(remaining_ref_data, remaining_gen_data))):
            completed_results.append(result)
            # 实时保存进度
            save_progress(completed_results, save_path)
            print(f'已处理 {i + 1} / {len(remaining_ref_data)} 个caption pair')

    print("全部处理完成")
if __name__ == '__main__':
    main()
