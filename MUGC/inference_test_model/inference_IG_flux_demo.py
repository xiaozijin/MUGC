import torch
from diffusers import StableDiffusionPipeline
import os
from diffusers import FluxPipeline

# 设置模型和设备
model_id = "/share/project/zpf/code/MCCU/inference/MPLUG-OWL/checkpoints/flux"
device = "cuda"
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

pipe = pipe.to(device)

# 确保输出文件夹存在
output_dir = "/share/project/zpf/code/MCCU/datasets/ge_image/flux"
os.makedirs(output_dir, exist_ok=True)

# 读取 prompt 文件
with open("/share/project/zpf/code/MCCU/evaluate/gpt4v/video/T2I/scoreca/image/clean_en/all.txt", "r") as f:
    prompts = f.readlines()

for i, prompt in enumerate(prompts):
    prompt = prompt.strip()  # 移除多余空白字符
    if prompt:  # 确保 prompt 非空
        # 使用生成器生成图像
        image = pipe(
            prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(0)  # 保持随机种子一致
        ).images[0]
        # 保存图像
        image.save(os.path.join(output_dir, f"image_{i+1}.png"))