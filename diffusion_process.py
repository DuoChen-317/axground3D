import os
from diffusers import StableDiffusionXLInpaintPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import gc


class ViewDiffuser:
    def __init__(self,):

        self.DIFFUSER_MODEL = "./model/RealVisXL_V4.0_inpainting"
        self.DEVICE = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
        self.PIPE = StableDiffusionXLInpaintPipeline.from_pretrained(
            self.DIFFUSER_MODEL, revision="fp16", variant="fp16",
        )
        # 2) swap in DPM++ 2M Karras scheduler
        self.PIPE.scheduler = DPMSolverMultistepScheduler.from_config(
            self.PIPE.scheduler.config, use_karras_sigmas=True
        )
        self.PIPE = self.PIPE.to(self.DEVICE)
        self.PIPE.enable_attention_slicing()  # reduce VRAM usage

    def fill_in(self, init_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        # Inpaint
        blurred_mask = self.PIPE.mask_processor.blur(mask_image, blur_factor=5)
        result = self.PIPE(
            prompt="a indoor room space, with furnitures ",  # empty prompt to rely solely on surrounding context
            image=init_image,
            mask_image=blurred_mask,
            mask_blur=0,
            height=512,
            width=512,
            guidance_scale=7.5,  # adjust as needed
            num_inference_steps=25,
        ).images[0]
        # gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return result
