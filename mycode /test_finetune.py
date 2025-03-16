import torch

from PIL import Image
from lavis.models import load_model_and_preprocess
from diffusers import StableDiffusionPipeline
import os

# 环境配置（根据网页1、5建议的私有部署方案优化）
os.environ.update({
    "TORCH_HOME": "/data/torch_cache",
    "TRANSFORMERS_CACHE": "/data/transformers_cache",
    "HUGGINGFACE_HUB_CACHE": "/data/huggingface_cache",
    "HF_HOME": "/data/huggingface_cache"
})

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
pipe = pipe.to("cuda")

prompt = "A futuristic city with flying cars"
image = pipe(prompt).images[0]  
image.save("generated_image.png")