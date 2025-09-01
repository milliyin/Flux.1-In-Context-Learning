# Flux.1 In-Context Learning

This repository contains experiments and training resources for **Flux.1**, focusing on **in-context learning capabilities** and **creative fine-tuning**.  
It includes workflows, datasets, and example code for applying Flux.1 to custom domains such as **pixel art characters**.

---

## 🔥 Overview

- **Base Model**: [`black-forest-labs/FLUX.1-dev`](https://huggingface.co/black-forest-labs/FLUX.1-dev)  
- **Adapter**: LoRA (Low-Rank Adaptation) applied to attention layers  
- **Custom LoRA Model**: [`milliyin/pixel_art_characters_lora_flux_nf4`](https://huggingface.co/milliyin/pixel_art_characters_lora_flux_nf4)  
- **Goal**: Extend Flux.1 beyond realism into **stylized domains** (e.g., pixel art, retro sprites, custom avatars)  
- **Training Hardware**: NVIDIA A100  

This repo demonstrates how **Flux.1** can adapt to new artistic styles using **in-context learning** with small, curated datasets.

---

## ✨ Features

- LoRA training scripts for Flux.1  
- Example dataset structures (image + caption pairs)  
- Sample inference notebooks for **text-to-image** generation  
- In-context editing & style transfer examples  
- Integration with Hugging Face **Diffusers**  

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/milliyin/Flux.1-In-Context-Learning.git
cd Flux.1-In-Context-Learning
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Inference with Pixel-Art LoRA

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16
).to("cuda")

# Load LoRA weights from Hugging Face
pipe.load_lora_weights("milliyin/pixel_art_characters_lora_flux_nf4")

prompt = "pixel art RPG adventurer, green cloak, facing forward, clean outline"
negative = "blurry, low quality, distorted"

image = pipe(
    prompt=prompt,
    negative_prompt=negative,
    height=512,
    width=512,
    guidance_scale=7.5,
    num_inference_steps=50
).images[0]

image.save("sample.png")
```

---

## 📜 License

This project is released under **CC BY-NC 4.0**.  
You may use it for research and personal projects, but **commercial usage is not allowed**.

---

## 📬 Contact

Created by **Muhammad Illiyin** (@milliyin)  
For inquiries or collaborations: [milliyin.vercel.app](https://milliyin.vercel.app)
