import torch
from diffusers import StableDiffusionPipeline
import os
from diffusers import FluxPipeline

model_id = "/share/project/zpf/code/MCCU/inference/MPLUG-OWL/checkpoints/flux"
device = "cuda"
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

pipe = pipe.to(device)

output_dir = "/share/project/MCCU/datasets/ge_image/flux"
os.makedirs(output_dir, exist_ok=True)

with open("/share/project/MCCU/evaluate/image/clean_en/all.txt", "r") as f:
    prompts = f.readlines()

for i, prompt in enumerate(prompts):
    prompt = prompt.strip()  
    if prompt: 
        image = pipe(
            prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(0)  
        ).images[0]
        image.save(os.path.join(output_dir, f"image_{i+1}.png"))
