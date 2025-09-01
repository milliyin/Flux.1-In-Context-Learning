import torch
from diffusers import FluxPipeline
import os

# Configuration - Edit these values directly
LORA_PATH = "/home/bilal/illiyin/art_characters_lora_flux_nf4/checkpoint-700"
BASE_MODEL = "black-forest-labs/FLUX.1-dev"
OUTPUT_DIR = "./generated_images"

# Generation settings
PROMPTS = [
    "pixel art character, with blue hair, wearing no clothes, holding no weapon, facing forward",
    "pixel art character, with blue hair, wearing purple apron, holding no weapon, facing forward",
    "pixel art character, with red hair, wearing gray armor, holding no weapon, facing forward",
    "pixel art character, with black hair, wearing gray armor, holding no weapon, facing forward",
    "pixel art character, with red hair, wearing brown armor, holding no weapon, facing forward",
    "pixel art character, with blue hair, wearing pink scarf, holding no weapon, facing forward",
    "pixel art character, with brown hair, wearing blue clothing, holding no weapon, facing forward"
]

NEGATIVE_PROMPT = "blurry, low quality, distorted, ugly"
HEIGHT = 512
WIDTH = 512
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 50
SEED = 42

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading base model...")
    # Load the base FLUX pipeline
    pipe = FluxPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16
    )
    
    print(f"Loading LoRA weights from {LORA_PATH}...")
    # Load LoRA weights
    pipe.load_lora_weights(LORA_PATH)
    
    # Move to GPU
    pipe = pipe.to("cuda")
    
    # Set seed for reproducible results
    torch.manual_seed(SEED)
    print(f"Using seed: {SEED}")
    
    print("Generating images...")
    print(f"Size: {WIDTH}x{HEIGHT}")
    print(f"Steps: {NUM_INFERENCE_STEPS}")
    print(f"Guidance Scale: {GUIDANCE_SCALE}")
    print("-" * 50)
    
    # Generate images for each prompt
    for i, prompt in enumerate(PROMPTS):
        print(f"Generating image {i+1}/{len(PROMPTS)}...")
        print(f"Prompt: {prompt}")
        
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=HEIGHT,
            width=WIDTH,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_INFERENCE_STEPS,
        ).images[0]
        
        # Save image
        output_path = os.path.join(OUTPUT_DIR, f"pixel_art_{i+1:02d}.png")
        image.save(output_path)
        print(f"Saved: {output_path}")
        print("-" * 30)
    
    print("Generation complete!")
    print(f"All images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()