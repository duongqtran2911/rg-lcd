import torch
from diffusers import (
    AutoencoderKL,
    LCMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

# =====================
# CONFIG
# =====================
model_name = "jiachenli-ucsb/RG-LCM-SD-2.1-768-HPSv2.1"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

prompt = "A photo of a girl holding a banana to her mouth"
num_inference_steps = 4
guidance_scale = 7.5
width = 768
height = 768
seed = 42

# =====================
# SET SEED
# =====================
torch.manual_seed(seed)

# =====================
# LOAD MODEL
# =====================
if "SD-2.1-base" in model_name:
    pretrained_teacher_model = "stabilityai/stable-diffusion-2-1-base"
else:
    pretrained_teacher_model = "sd2-community/stable-diffusion-2-1"

vae = AutoencoderKL.from_pretrained(
    pretrained_teacher_model,
    subfolder="vae",
    torch_dtype=dtype,
)

scheduler = LCMScheduler.from_pretrained(
    pretrained_teacher_model,
    subfolder="scheduler"
)

unet = UNet2DConditionModel.from_pretrained(
    model_name,
    torch_dtype=dtype
)

pipe = StableDiffusionPipeline.from_pretrained(
    pretrained_teacher_model,
    vae=vae,
    unet=unet,
    scheduler=scheduler,
    torch_dtype=dtype,
).to(device)

# Optional speed-up
pipe.enable_xformers_memory_efficient_attention() if device == "cuda" else None

# =====================
# INFERENCE
# =====================
images = pipe(
    prompt=prompt,
    width=width,
    height=height,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    num_images_per_prompt=1,
).images

# =====================
# SAVE OUTPUT
# =====================
output_path = "output.png"
images[0].save(output_path)

print(f"Saved image to {output_path}")