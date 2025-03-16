import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import os

# 设置 PyTorch 缓存路径
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["TORCH_HOME"] = "/data/torch_cache"
# 设置 Hugging Face 模型缓存路径（LAVIS 依赖）
os.environ["TRANSFORMERS_CACHE"] = "/data/transformers_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/huggingface_cache"
os.environ["HF_HOME"] = "/data/huggingface_cache"  

dip_model, dip_vis_processors, _ = load_model_and_preprocess(
    name="blip_caption", 
    model_type="base_coco",
    is_eval=True,
    device="cuda"
)

# 处理输入图像
image1 = Image.open("./images/cat-sofa.png").convert("RGB")
processed_img1 = dip_vis_processors["eval"](image1).unsqueeze(0).cuda()

# 生成描述文本
caption = dip_model.generate({"image": processed_img1})[0]

# 提取主题关键词（示例输出："a golden retriever playing in the park"）
theme = " ".join([word for word in caption.split() if word not in ["a", "an", "the"]][:3])
model, vis_preprocess, txt_preprocess = load_model_and_preprocess("blip_diffusion", "base", device="cuda", is_eval=True)

cond_subject = theme
tgt_subject = theme
# prompt = "painting by van gogh"
text_prompt = "swimming underwater"

cond_subjects = [txt_preprocess["eval"](cond_subject)]
tgt_subjects = [txt_preprocess["eval"](tgt_subject)]
text_prompt = [txt_preprocess["eval"](text_prompt)]

cond_image = Image.open("./images/cat-sofa.png").convert("RGB")

cond_images = vis_preprocess["eval"](cond_image).unsqueeze(0).cuda()
samples = {
    "cond_images": cond_images,
    "cond_subject": cond_subjects,
    "tgt_subject": tgt_subjects,
    "prompt": text_prompt,
}


iter_seed = 88888
guidance_scale = 7.5
num_inference_steps = 50
negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

output = model.generate(
        samples,
        seed=iter_seed,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        neg_prompt=negative_prompt,
        height=512,
        width=512,
    )

output[0].save("output.png")