# MUGC
The implementation about the paper MUGC:Holistic Evaluation of Multimodal Understanding and Generation for Chinese Cultural Contexts.

## Introduction
we propose MUGC (Multimodal Understanding and Generation in Chinese Cultural Contexts), the first unified benchmark designed around Chinese cultural scenarios—an application domain rich in symbolic and aesthetic elements, and well-suited as a case study for evaluating multimodal models' ability to capture cultural semantics across image, video, and text modalities. We begin by assembling a diverse collection of Chinese‑style images and videos, generating initial captions with a pretrained vision–language model and refining them manually to form high‑quality image‑text and video‑text pairs. Next, we define a set of cultural attributes—such as object identity, background composition, and symbolic features—and employ an LLM‑based information extractor to convert free‑form captions into structured representations along these dimensions. An instruction‑tuned judge model then assigns interpretable scores for each attribute, and performance is quantified by win‑rate comparisons against a high‑quality reference model, yielding composite scores that jointly reflect semantic fidelity and cultural appropriateness.

## Experiment
We evaluated multiple MLLM models on both generation tasks (image understanding, video understanding) and understanding tasks (image generation,video generation) to verify their performance in comprehending and generating content in a Chinese style.

The image understanding (IU) open-source models we evaluated include Aquila-vl-2B, Cogvlm2-llama3-chinese-19B , Deepseek-vl-7B , Emu3-Chat-8B ,GLM4.1v-9B-Thinking ,GLM-4.6V-10B-Flash (Hong et al., 2025), Intern-s1-mini-9B , MiniCPMV-4.5-9B , Onevision-qwen2-7B , Ministral3-8B-Instruct, Nemotron-nano-12B-2-
VL , Qwen3.5-9B ,Qwen3-vl-8B-Instruct  and Janus-
Pro-7B.

The video understanding (VU) open-source models we evaluated include GLM-4.6V, Nemotron-nano-12B-2-VL,
VideoLLaMA2-7B, Llava-Next-Video-7B , Llava-Video-7B-Qwen2 , Qwen3.5-9B, Qwen3-vl-8B-Instruct and MiniCPMV-4..

The image generation (IG) open-source models we evaluated include qwen-image, z-image-turbo, flux2, dream-like photo real, OpenJourney, Emu3-Gen, Kandinsky3, sdxl, sdxl-turbo, flux, Janus-image, Playground V2.5.

The closed-source video generation (VG) models we evaluated include Dream Machine, hailuo, tongyi, Gen2, Pika, Pixverse, kling, VIDU, and zhipu.



