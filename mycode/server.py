from flask import Flask, request, send_file, jsonify
from PIL import Image
import torch
from lavis.models import load_model_and_preprocess
import os
import uuid
from io import BytesIO
from flask_cors import CORS
from diffusers import StableDiffusionPipeline
import base64

# 初始化Flask应用
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",         # 允许所有域名
        "methods": ["GET", "POST", "PUT", "DELETE"],  # 允许的HTTP方法[6](@ref)
        "allow_headers": "Content-Type,Authorization" # 允许的自定义请求头[1](@ref)
    }
})

# 配置缓存路径（需提前创建目录）
os.environ.update({
    "TORCH_HOME": "/data/torch_cache",
    "TRANSFORMERS_CACHE": "/data/transformers_cache",
    "HUGGINGFACE_HUB_CACHE": "/data/huggingface_cache"
})

# 全局加载模型（避免重复加载）
model, vis_preprocess, txt_preprocess = None, None, None
try:
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
    pipe = pipe.to("cuda") if torch.cuda.is_available() else pipe
except Exception as e:
    app.logger.error(f"模型加载失败: {str(e)}")

def load_models():
    global model, vis_preprocess, txt_preprocess
    if model is None:
        model, vis_preprocess, txt_preprocess = load_model_and_preprocess(
            "blip_diffusion", "base", device="cuda", is_eval=True
        )

load_models()  # 应用启动时加载模型

# 通用错误处理
@app.errorhandler(500)
def handle_error(e):
    return jsonify(error=str(e)), 500

# 文生图接口
@app.route('/text2img', methods=['POST'])
def text_to_image():
    try:
        # 参数校验
        text_prompt = request.form.get('text_prompt', '').strip()
        if not text_prompt:
            abort(400, description="text_prompt参数不能为空")
        
        # 生成图像
        image = pipe(text_prompt,
                    height=512,
                    width=512).images[0]
        
        # 内存流处理
        img_io = BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        img_data = base64.b64encode(img_io.getvalue()).decode('utf-8')
        return jsonify({ "image": f"data:image/png;base64,{img_data}" })
    
    except AttributeError:
        abort(500, description="模型未正确初始化")
    except torch.cuda.OutOfMemoryError:
        abort(503, description="GPU内存不足，请稍后重试")
    except Exception as e:
        app.logger.error(f"生成失败: {str(e)}")
        abort(500)

# 图生图接口
@app.route('/img2img', methods=['POST'])
def image_to_image():
    try:
        # 接收文件与参数
        cond_image = request.files['cond_image']
        src_image = request.files['src_image']
        data = request.form
        
        # 参数校验
        required_params = ['text_prompt', 'cond_subject', 'src_subject', 'tgt_subject']
        if not all(data.get(p) for p in required_params):
            raise ValueError("Missing required parameters")

        # 处理输入图像
        cond_img = Image.open(cond_image.stream).convert('RGB')
        src_img = Image.open(src_image.stream).convert('RGB')

        # 预处理流程
        cond_img = vis_preprocess["eval"](cond_img).unsqueeze(0).cuda()
        src_subject = txt_preprocess["eval"](data['src_subject'])
        tgt_subject = txt_preprocess["eval"](data['tgt_subject'])
        cond_subject = txt_preprocess["eval"](data['cond_subject'])
        text_prompt = [txt_preprocess["eval"](data['text_prompt'])]

        # 构建输入样本
        samples = {
            "cond_images": cond_img,
            "cond_subject": cond_subject,
            "src_subject": src_subject,
            "tgt_subject": tgt_subject,
            "prompt": text_prompt,
            "raw_image": src_img,
        }

        # 调用模型生成
        output = model.edit(
            samples,
            seed=int(data.get('seed', 88871)),
            guidance_scale=float(data.get('guidance_scale', 7.5)),
            num_inference_steps=int(data.get('num_steps', 50)),
            num_inversion_steps=int(data.get('num_inversion_steps', 50)),
            neg_prompt=data.get('negative_prompt', "")
        )

        # 返回生成图像
        img_io = BytesIO()
        output[1].save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)