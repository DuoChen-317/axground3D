import os
from diffusers import StableDiffusionXLInpaintPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image


DIFFUSER_MODEL = "./model/RealVisXL_V4.0_inpainting"
DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
PIPE = StableDiffusionXLInpaintPipeline.from_pretrained(
        DIFFUSER_MODEL, revision="fp16", variant="fp16"
    )
# 2) swap in DPM++ 2M Karras scheduler
PIPE.scheduler = DPMSolverMultistepScheduler.from_config(
    PIPE.scheduler.config, use_karras_sigmas=True
)
PIPE = PIPE.to(DEVICE)
PIPE.enable_attention_slicing()  # reduce VRAM usage




def fill_in(init_image: Image.Image, mask_image: Image.Image) -> Image.Image:
    # Inpaint
    result = PIPE(
        prompt="a indoor room",  # empty prompt to rely solely on surrounding context
        image=init_image,
        mask_image=mask_image,
        guidance_scale=7.5,  # adjust as needed
        num_inference_steps=25,
    ).images[0]

    return result
