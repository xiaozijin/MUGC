import torch
import sys
sys.path.append("/share/project/zpf/code/MCCU/inference/MPLUG-OWL/")
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM
from torchvision import transforms
import os
import argparse
from pathlib import Path
import json
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser
from torch.nn.utils.rnn import pad_sequence
parser = ArgumentParser()
parser.add_argument("--output_dir", type=str,default="/share/project/zpf/code/MCCU/datasets/image")
parser.add_argument("--pretrained_ckpt", type=str,default="/share/project/zpf/code/MCCU/inference/MPLUG-OWL/checkpoints/glm-4v-9b")
parser.add_argument("--input_file", type=str,default="/share/project/zpf/code/MCCU/datasets/image/test.json")
parser.add_argument("--dataset_root", type=str)
parser.add_argument("--mode", type=str, choices=["CN", "EN"], default="CN")
args = parser.parse_args()

def perfect_image_path(image_path):
    if image_path.endswith(".jpg") or image_path.endswith(".png"):
        return image_path
    return image_path+".jpg"

class ImageMetadataDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, dataset_root=None):
        with open(file_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if dataset_root is not None:
            for i in range(len(self.metadata)):
                self.metadata[i]["image_path"] = os.path.join(dataset_root, self.metadata[i]["image_path"])
            
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        info = self.metadata[idx]
        image_path = perfect_image_path(info["image_path"])
        # image = Image.open(image_path).resize((500,500)).convert("RGB")
        # transform = transforms.Pad((500,500))
        # transform = transforms.ToTensor()
        # image = transform(image)
        # print(f"Processed image size: {image.size()}") m
        return {
                "image_path": image_path
            }
        # return {
        #     "image_path": perfect_image_path(info["image_path"]),
        # }

def collate_fn(data):
    ret={
        "image_path" : [x["image_path"] for x in data],
    }
    return ret

# def custom_collate_fn(batch):
#     images = [item['image'] for item in batch]
#     paths = [item['image_path'] for item in batch]
    
#     # 找到批次中最大的图像尺寸
#     max_height = max([img.shape[1] for img in images])
#     max_width = max([img.shape[2] for img in images])
    
#     # 对所有图像进行填充，使它们的尺寸一致
#     padded_images = []
#     for img in images:
#         padding = (0, 0, max_width - img.shape[2], max_height - img.shape[1])
#         padded_img = transforms.functional.pad(img, padding)
#         padded_images.append(padded_img)
    
#     # 转换为张量
#     padded_images = torch.stack(padded_images)
#     return {
#         "image": padded_images,
#         "image_path": paths
#     }

def main():
    seed = 0
    output_dir = args.output_dir # {name}_p{version}_f{num_frames}
    pretrained_ckpt = args.pretrained_ckpt
    input_file = args.input_file
    dataset_root = args.dataset_root
    save_interval = 10
    batch_size = 1 
    if args.mode=="EN":
        base_prompt = "Please describe this image in detail.response in english"
    else:
        base_prompt = "请详细描述图片内容。请用中文回答。"

    accelerator = Accelerator()
    set_seed(seed)
    if accelerator.is_main_process:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"test_captions_{accelerator.process_index:02d}.json"
    )

    dataset = ImageMetadataDataset(input_file, dataset_root)
    # assert batch_size==1
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    dataloader = accelerator.prepare(dataloader)

    
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_ckpt,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to("cuda").eval()
    model.to(accelerator.device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt, trust_remote_code=True)


    progress_bar = tqdm(
        range(0, len(dataloader)),
        initial=0,
        desc="Batches",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    results = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
            
    processed_set = set()
    
    def add_to_processed(processed_set, info_str):
        processed_set.add(info_str)
        
    def in_processed(processed_set, info_str):
        return info_str in processed_set
    
    def construct_info_str(item):
        return f'{item["image_path"]}'
    
    for result in results:
        add_to_processed(processed_set, construct_info_str(result))
        
    for step, batch in enumerate(dataloader, start=1):
        image_paths = batch["image_path"]
        for image in image_paths:
            image = Image.open(image).convert("RGB")

        
        info_str_list = [construct_info_str({"image_path":image_path}) for image_path in image_paths]
        batch_in_processed = [in_processed(processed_set, info_str) for info_str in info_str_list]
        if all(batch_in_processed):
            progress_bar.update(1)
            continue
        
        # import pdb ; pdb.set_trace()
        inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": base_prompt}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)
        gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
        inputs = inputs.to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            res = tokenizer.decode(outputs[0])

        for i in range(len(image_paths)):
            if batch_in_processed[i]:
                continue
            results.append(
                {
                    "image_path": image_paths[i],
                    "caption": res,
                }
            )
        progress_bar.update(1)
        if step % save_interval == 0 or step == len(dataloader):
            json.dump(
                results,
                open(output_path, "w", encoding="utf-8"),
                indent=4,
                ensure_ascii=False,
            )
    progress_bar.close()
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()