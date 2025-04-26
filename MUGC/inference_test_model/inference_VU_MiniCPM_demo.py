import torch
import sys
sys.path.append("/share/project/MCCU/inference/MPLUG-OWL/")
from transformers import AutoTokenizer,AutoModel,AutoModelForCausalLM

import os
import argparse
from torchvision import transforms
from pathlib import Path
import json
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from decord import VideoReader
from PIL import Image
from argparse import ArgumentParser
from typing import Tuple, Dict
import numpy as np
from decord import VideoReader, cpu
parser = ArgumentParser()
parser.add_argument("--output_dir", type=str,default="/share/project/MCCU/datasets/video")
parser.add_argument("--pretrained_ckpt", type=str,default="/share/project/MCCU/inference/MPLUG-OWL/checkpoints/MiniCPM")
parser.add_argument("--input_file", type=str,default="/share/project/MCCU/datasets/video/test.json")
parser.add_argument("--dataset_root", type=str)
parser.add_argument("--mode", type=str, choices=["CN", "EN"], default="CN")
args = parser.parse_args()

def get_index(num_segments, frames_start_end:Tuple[int,int]):
    frames_start, frames_end = frames_start_end
    num_frames = frames_end - frames_start

    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array(
        [start + int(np.round(seg_size * idx)) for idx in range(num_segments)]
    )
    indices = frames_start + offsets
    return indices

def read_specific_frames(video_path, num_segments, frames_start_end):
    try:
        vr = VideoReader(video_path, height=512, width=512)
        if tuple(frames_start_end) == (0,0):
            frames_start_end = (0, len(vr))
        frame_indices = get_index(num_segments, frames_start_end)
        loaded_frames = vr.get_batch(frame_indices).asnumpy()
        images_group = [Image.fromarray(frame).convert('RGB') for frame in loaded_frames]
    except:
        return False
    return images_group

class VideoMetadataDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, dataset_root=None):
        with open(file_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        if dataset_root is not None:
            for i in range(len(self.metadata)):
                self.metadata[i]["video_path"] = os.path.join(dataset_root, self.metadata[i]["video_path"])
            
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        info = self.metadata[idx]
        if info["start"] == 0 and info["end"] == 0:
            item = self.get_video_clip_metainfo(info["video_path"], (0,0))
        return {
            "video_path": info["video_path"],
            "frames_start_end": (item["start"], item["end"]),
            "start": item["start"],
            "end": item["end"]
        }
    

    def get_video_clip_metainfo(self, video_path, frames_start_end=(0,0)):
        vr = VideoReader(video_path)
        if tuple(frames_start_end)==(0,0):
            frames_start_end = (0, len(vr))
        total_frames = frames_start_end[1] - frames_start_end[0]
        fps = round(vr.get_avg_fps(), 0)
        duration = round(total_frames / fps, 2)
        print(frames_start_end)
        return {
            "total_frames": total_frames,
            "fps": fps,
            "duration": duration,
            "frames_start_end": frames_start_end,
            "start": frames_start_end[0],
            "end": frames_start_end[1],
        }

MAX_NUM_FRAMES=64
def collate_fn(data):
    ret={
        "video_path" : [x["video_path"] for x in data],
        "frames_start_end" : [x["frames_start_end"] for x in data],
    }
    return ret
def encode_video(video_paths):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    for video_path in video_paths:
        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)  # FPS
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        print('num frames:', len(frames))
    return frames

def main():
    seed = 42
    output_dir = args.output_dir # {name}_p{version}_f{num_frames}
    pretrained_ckpt = args.pretrained_ckpt
    input_file = args.input_file
    dataset_root = args.dataset_root
    num_segments = 10
    save_interval = 50
    batch_size = 1 # be sure
    if args.mode=="EN":
        base_prompt = "Please describe this video in detail.response in english"
    else:
        base_prompt = "请详细描述视频内容。请用中文回答。"

    accelerator = Accelerator()
    set_seed(seed)
    if accelerator.is_main_process:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"test_captions_{accelerator.process_index:02d}.json"
    )

    dataset = VideoMetadataDataset(input_file, dataset_root)
    # frams = read_specific_frames(video_path=video_paths,num_segments=num_segments,frames_start_end=frames_start_ends)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    dataloader = accelerator.prepare(dataloader)
    
    model = AutoModel.from_pretrained(pretrained_ckpt, trust_remote_code=True,
        attn_implementation='sdpa', torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt, trust_remote_code=True)
    model.to(accelerator.device)

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
        return f'{item["video_path"]}_{item["frames_start_end"][0]}_{item["frames_start_end"][1]}'
    
    for result in results:
        add_to_processed(processed_set, construct_info_str(result))
        
    for step, batch in enumerate(dataloader, start=1):
        video_paths = batch["video_path"]
        frames = encode_video(video_paths)
        msgs = [
    {'role': 'user', 'content': frames + [base_prompt]}, 
]
        frames_start_ends = batch["frames_start_end"]
        params={}
        params["use_image_id"] = False
        params["max_slice_nums"] = 2 # use 1 if cuda OOM and video resolution >  448*448
        
        info_str_list = [construct_info_str({"video_path":video_path, "frames_start_end":frames_start_end}) for video_path, frames_start_end in zip(video_paths, frames_start_ends)]
        batch_in_processed = [in_processed(processed_set, info_str) for info_str in info_str_list]
        print(batch_in_processed)
        if all(batch_in_processed):
            progress_bar.update(1)
            continue
        with torch.cuda.amp.autocast():
           answer = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                **params
)

        for i in range(len(video_paths)):
            if batch_in_processed[i]:
                continue
            results.append(
                {
                    "video_path": video_paths[i],
                    "frames_start_end": frames_start_ends[i],
                    "start": frames_start_ends[i][0],
                    "end": frames_start_ends[i][1],
                    "caption": answer,
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
