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


def get_parsers():
    parser = argparse.ArgumentParser(description="Caption eval.")
    parser.add_argument("--gold-file", type=str, default='/share/project/MCCU/datasets/video/ref.jsonl')
    parser.add_argument("--test-file", type=str, default='/share/project/MCCU/datasets/video/format/llava-next-video/all.jsonl')
    parser.add_argument("--process-num", type=int, default=5)
    parser.add_argument("--output_path", type=str, default='/share/project/MCCU/datasets/video/score/llava-next-video')
    args= parser.parse_args()
    return args

args = get_parsers()

def get_caption_data_mplug_image():
    test_data = []
    test_dict = {}
    test_list = []
    with open(args.test_file, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            test_data.append(data)
    # for line in open(args.test_file + "/captions_00.json",'r'):
    #     test_data.append(json.loads(line))
    for item in test_data:
        test_dict[item["image_path"].split("/")[-1].split('.')[0]] = [item["caption"].strip()]
    for line in open(args.gold_file, 'r'):
        item = json.loads(line)
        path = item["path"].split("/")[-1].split(".")[0]
        if path not in test_dict:
            continue
        test_dict[path].append(item["text"])
        test_dict[path].append(item["path"])
        test_dict[path].append(item["caption"])
    for key, value in test_dict.items():
        if len(value) == 4:
            test_list.append(value)
    print("total length: {}".format(len(test_list)))
    return test_list

def get_caption_data_mplug_video():
    test_data = []
    with open(args.test_file, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            test_data.append(data)
    test_dict = {}
    test_list = []
    
    for item in test_data:
        print(item["video_path"].split("/")[-1])
        test_dict[item["video_path"].split("/")[-1]] = [item["caption"].strip()]
    for line in open(args.gold_file, 'r'):
        item = json.loads(line)
        path = item["path"].split("/")[-1]
        print(path)
        if path not in test_dict:
            continue
        test_dict[path].append(item["text"])
        test_dict[path].append(item["path"])
        test_dict[path].append(item["caption"])
    for key, value in test_dict.items():
        if len(value) == 4:
            test_list.append(value)
    print("total length: {}".format(len(test_list)))
    return test_list

def contains_english(text):
    pattern = re.compile(r'[a-zA-Z]')
    return bool(pattern.search(text))

def get_caption_data():
    test_data = json.load(open(args.test_file , 'r')) 
    test_dict = {}
    test_list = []
    for item in test_data:
        if "sorry" in item["caption"]:
            continue
        test_dict[item["path"].split("/")[-1].split(".")[0]] = [item["caption"]]
    for line in open(args.gold_file, 'r'):
        item = json.loads(line)
        path = item["path"].split("/")[-1]
        if path not in test_dict:
            continue
        test_dict[path].append(item["text"])
        test_dict[path].append(item["path"])
        test_dict[path].append(item["caption"])
    for key, value in test_dict.items():
        if len(value) == 4:
            test_list.append(value)
    print("total length: {}".format(len(test_list)))
    return test_list

def get_caption_data_3():
    test_data = json.load(open(args.test_file , 'r')) 
    test_dict = {}
    test_list = []
    for item in test_data:
        if "sorry" in item["caption"]:
            continue
        test_dict[item["video_path"].split("/")[-1].split(".")[0]] = [item["caption"]]
    for line in open(args.gold_file, 'r'):
        item = json.loads(line)
        path = item["path"].split("/")[-1].split(".")[0]
        if path not in test_dict:
            continue
        test_dict[path].append(item["text"])
        test_dict[path].append(item["path"])
        test_dict[path].append(item["caption"])
    for key, value in test_dict.items():
        if len(value) == 4:
            test_list.append(value)
    print("total length: {}".format(len(test_list)))
    return test_list

def extract_score(string):
    string = string.lower().replace("\n", "").replace("\"", "")
    match = re.search(r'score:\s*([\d]+)', string)
    print(string)
    if match:
        return match.group(1)
    else:
        match =  re.search(r'(\d+)\s+out\s+of\s+100', string)
        if match:
            return match.group(1)
        else:
            match = re.search(r'score: \(.+?\) = ([\d]+)', string)
            if match:
                return match.group(1)
            else:
                match = re.search(r'the score is (\d+)', string)
                if match:
                    return match.group(1)
                else:
                    match = re.search(r'The score for the test caption is (\d+)', string)
                    if match:
                        return match.group(1)
                    else:
                        match = re.search(r'a score of (\d+)', string)
                        if match:
                            return match.group(1)
                        else:
                            string = string[-5:] if len(string) > 5 else string
                            match = re.search(r'[\d]+', string)
                            if match:
                                return match.group(0)
                            else:
                                match = re.search(r'The score for the test caption is (\d+)', string)
                                if match:
                                    return match.group(1)
                                else:
                                    return None
def get_caption(text):
    score = extract_score(text)
    return score

def call_chatgpt_azure(query):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
    )
    # print(response['choices'][0]['message']['content'])
    return response.choices[0].message.content


@timeout_decorator.timeout(TIMEOUT)
def call_chatgpt_azure_ceval(sen):
  template_bustm = '''
    为您提供两个视频的对应caption。gold caption是标准答案，已解析为JSON格式，test caption是生成的结果。现在，我们需要参考标准答案，并对test caption的准确性进行评分。请按照以下步骤进行评估。首先，请从test caption的描述中提取backgroud、object和text部分，形成JSON。JSON字段包括：对backgroud、object和text。object是指标题中的对象，这些对象需要包括但不限于标题中描述的以下属性，如外观、动作、位置。请随时添加任何相关信息。如果没有关于某些属性的描述，则相关字段将填充为空白。backgroud是指出现在标题中的背景，text是指标题中双引号内的文本信息（如果没有双引号，则填写为空白）。\ntest caption转成JSON文件后，可以比较两个JSON中描述的backgroud、object和text之间的相似性，给出background,object和text之间的分数，给出两者中国风元素间的分数，并给出测试标题的分数，所有分数最高分数均为100。\n \
    gold caption的json如下:\n ''' + sen[1] + ''' \n
    test caption如下:\n ''' + sen[0] + '''\n
    score输出格式:\n \nbackground_score:[background_score here]\n object_score: [object_score here]\n text_score: [text_score here]\n culture_score: [culture_score here]\n total_score: [total_score here]\n \
    score属性字段中只需要有上面各个分数字段及相应分数，此外不用任何东西
    '''
    return call_chatgpt_azure(template_bustm)

def call_multiprocess(ipt_list, save_dir, filename, meta_func):
    save_path = os.path.join(save_dir, filename)
    if os.path.exists(save_path):
        raise Exception(f'Error: save_path [{save_path}] Already Exists!')
    f = open(save_path, 'a+')
    for i in range(len(ipt_list)):
        try_count = 1
        # opt = meta_func(ipt_list[i])
        while True:
            try:
                opt = meta_func(ipt_list[i])
                break
            except Exception as e:
                print(e)
                # traceback.print_exc()
                print(f'INFO: retry [i], try count [{try_count}]')
                try_count += 1
                if try_count == 1000:
                    opt = ''
                    break
        temp = {}
        temp["score"] = get_caption(opt)
        temp["response"] = opt
        temp["test_caption"] = ipt_list[i][0]
        temp["gold_caption"] = ipt_list[i][3]
        temp["path"] = ipt_list[i][2]
        f.write(json.dumps(temp, ensure_ascii=False) + '\n')
    f.close()
    print(f'INFO: [{save_path}] done.')
    return
  
def merge_results(output_dir, process_num):
    f = open(os.path.join(output_dir, 'all.jsonl'), 'w+')
    for idx in range(process_num):
        filepath = os.path.join(output_dir, f'{idx}.jsonl')
        lines = open(filepath).readlines()
        lines = [item.strip() for item in lines if len(item.strip()) > 0]
        for line in lines:
            f.write(line + '\n')
    f.close()

def run_multiprocess(ipt_func, meta_func, output_dir, process_num):
    ipt_list = ipt_func()
    print(f'INFO: data size: [{len(ipt_list)}]')
    split_size = len(ipt_list) // process_num
    print(split_size)
    if len(ipt_list) % process_num != 0:
        split_size += 1
    p = Pool(process_num)
    for i in range(process_num):
        sub_data = ipt_list[
            split_size * i:
            split_size * (i+1)
        ]
        filename = f'{i}.jsonl'
        p.apply_async(call_multiprocess, args=(sub_data, output_dir, filename, meta_func))
        call_multiprocess(sub_data, output_dir, filename, meta_func)
    print('waiting for all subprocesses done ...')
    p.close()
    p.join()
    print('all processed done.')

    print('start merge ...')
    merge_results(output_dir, process_num)
    print('merge done.')


def main():
    
    run_multiprocess(
        get_caption_data_mplug_video,  
        call_chatgpt_azure_ceval,  
        args.output_path, 
        args.process_num
    )
main()
