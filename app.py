import logging
import os
import random
import re
import sys
import warnings
import gradio as gr
import torch
import math

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modeling.t2i_pipeline import BitDanceT2IPipeline
except ImportError:
    print("Warning: Could not import BitDanceT2IPipeline. Please ensure 'modeling' folder is present.")

# ==================== Environment Variables ==================================
MODEL_PATH = "models/BitDance-14B-64x"

# =============================================================================
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ==================== Resolution Settings ====================================
RAW_RESOLUTIONS = [
    [2048, 512],
    [1920, 512],
    [1536, 640],
    [1280, 768],
    [1152, 896],
    [1024, 1024],
    [896, 1152],
    [768, 1280],
    [640, 1536],
    [512, 1920],
    [512, 2048],
]

RESOLUTION_CHOICES = []
for w, h in RAW_RESOLUTIONS:
    divisor = math.gcd(w, h)
    ratio_w = w // divisor
    ratio_h = h // divisor
    label = f"{w}x{h} ({ratio_w}:{ratio_h})"
    RESOLUTION_CHOICES.append(label)

DEFAULT_RES = "1024x1024 (1:1)"

EXAMPLE_PROMPTS = [
    ["一幅具有电影感的胶片肖像，一位美丽的中国女生，凌乱的黑发在风中飘动遮住脸庞，眼神灵动地看着镜头。她在画面的左1/3处。她围着一条厚实的鲜红色针织围巾，穿着一件破旧的米色羊羔毛外套。背景是日落时分寒冷、干枯的荒野和远山。强烈的金色逆光直射镜头，产生巨大的镜头眩光和朦胧的光晕效果，空气中有尘埃感。胶片颗粒质感，浅景深，自然原始的风格。"],
    ["一位穿着粉色吊带罗纹长裙的亚洲少女，外搭一件米白色毛绒短开襟衫，在阳光洒落的森林小径上侧身回眸。她拥有淡粉色薰衣草发色的甜美脸庞，发间别着一朵白色小花。黄金时段的光线穿过浓密的树叶，在深绿色的背景上形成美丽的景深光斑 和柔和光晕。电影级肖像摄影，超高画质，细腻的皮肤纹理，强调少女的温柔与唯美浪漫的日系氛围。"],
    ["一个半人半机械的黑客，坐在充满全息屏幕的黑暗房间里，绿色的代码光映照在他的脸上，赛博朋克风格，高科技细节，锐利的焦点。"],
    ["A surreal double exposure portrait that blends a woman’s face with a beautiful seascape. The overall mood is dreamy and mystical, with rich colors and intricate details."],
    ["A close-up, macro photography stock photo of a strawberry intricately sculpted into the shape of a hummingbird in mid-flight, its wings a blur as it sips nectar from a vibrant, tubular flower. The backdrop features a lush, colorful garden with a soft, bokeh effect, creating a dreamlike atmosphere. The image is exceptionally detailed and captured with a shallow depth of field, ensuring a razor-sharp focus on the strawberry-hummingbird and gentle fading of the background. The high resolution, professional photographers style, and soft lighting illuminate the scene in a very detailed manner, professional color grading amplifies the vibrant colors and creates an image with exceptional clarity. The depth of field makes the hummingbird and flower stand out starkly against the bokeh background."],
    ["网红咖啡店内部，透过钢化玻璃拍摄，中景平视角度；玻璃表面有环境反光与色彩叠影，人物面部柔光打亮，坐着看向镜头，穿着带大毛领的宽松上衣；白天咖啡店，太阳光线打在人物脸上，玻璃反光清透自然，ccd质感。"],
    ["室内中景人像摄影，复古胶片风格，电影叙事感画面。一位清纯气质的年轻女性，留着黑色齐刘海长直发，妆容清透伪素颜，皮肤白皙透亮。她身穿一件质地柔软、淡绿色的马海毛（Mohair）绒毛毛衣，质感毛绒蓬松，下身搭配淡青色棉麻长裙。人物慵懒地蜷缩/侧卧在沙发角落，身体姿态放松柔软，呈现自然的C型曲线。一只手轻轻拿着一颗鲜红的番茄靠近脸颊和下巴，眼神迷离、温柔且深情地直视镜头，表情处于放空与凝视之间，极具故事感。复古文艺的室内一角，沙发上铺着淡雅的复古碎花布艺沙发罩，身旁放着一盘红色的番茄作为前景点缀。背景虚化，隐约可见室内的陈设与绿植，整体环境色调偏向青绿色的胶片感。极具艺术感的局部自然光（丁达尔效应光斑）。一束明亮的午后阳光精准地照射在手部、手中的番茄以及面部一侧，形成强烈的明暗对比（Chiaroscuro）。高光部分带有光晕（Bloom），阴影部分呈现胶片特有的青蓝色调，光影层次丰富。。慵懒、静谧、梦幻、日系文艺、情绪感强、高级且富有夏末秋初的诗意。。模拟胶片相机（如Contax T3或Pentax 67）拍摄，使用50mm标准定焦镜头，大光圈（f/1.8）制造柔和的背景虚化。后期加入明显的粗颗粒胶片滤镜（Heavy Film Grain）和色彩偏移，增强模拟摄影的真实感与年代感。极度真实的皮肤质感，保留面部微小的毛孔和纹理，拒绝过度磨皮；马海毛毛衣在逆光下呈现出清晰的绒毛光晕边缘；番茄表面光滑的高光反射；碎花布料的褶皱细节；整体画面覆盖一层复古的胶片噪点。"],
]

def get_resolution(resolution_str):
    match = re.search(r"(\d+)\s*[×x]\s*(\d+)", resolution_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 1024, 1024

def load_models(model_path):
    print(f"Loading BitDance model from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"Warning: Model path {model_path} does not exist locally. Attempting to load anyway (or handle download logic here).")

    pipe = BitDanceT2IPipeline(model_path=model_path, device="cuda")
    return pipe

def generate_image(
    pipe,
    prompt,
    resolution,
    seed=42,
    guidance_scale=7.5,
    num_inference_steps=50,
):
    width, height = get_resolution(resolution)
    
    images = pipe.generate(
        prompt=prompt,
        height=height,
        width=width,
        num_sampling_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images=1,
        seed=seed
    )
    
    return images[0]

pipe = None

def init_app():
    global pipe
    try:
        pipe = load_models(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        pipe = None

def generate(
    prompt,
    resolution,
    seed=42,
    steps=50,
    guidance_scale=7.5,
    random_seed=True,
    gallery_images=None,
    progress=gr.Progress(track_tqdm=True),
):
    if random_seed:
        new_seed = random.randint(1, 1000000)
    else:
        new_seed = seed if seed != -1 else random.randint(1, 1000000)

    if pipe is None:
        raise gr.Error("Model not loaded.")

    print(f"Generating: Prompt='{prompt[:20]}...', Res={resolution}, Seed={new_seed}, Steps={steps}, CFG={guidance_scale}")

    try:
        image = generate_image(
            pipe=pipe,
            prompt=prompt,
            resolution=resolution,
            seed=new_seed,
            guidance_scale=guidance_scale,
            num_inference_steps=int(steps),
        )
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")

    if gallery_images is None:
        gallery_images = []
    
    gallery_images = [image] + gallery_images 
    
    return gallery_images, str(new_seed), int(new_seed)

init_app()

# ==================== Gradio UI ====================

with gr.Blocks(title="BitDance Demo") as demo:
    gr.Markdown(
        """<div align="center">

### BitDance: Scaling Autoregressive Generative Models with Binary Tokens

[🕸️ Project Page](TBD) • [📄 Paper](TBD) • [💻 Code](https://github.com/shallowdream204/BitDance) • [📦 Model](https://huggingface.co/collections/shallowdream204/bitdance)

</div>""",
    elem_id="title",
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(label="Prompt", lines=3, placeholder="Enter your prompt here...")
            
            resolution = gr.Dropdown(
                value=DEFAULT_RES, 
                choices=RESOLUTION_CHOICES, 
                label="Resolution (Width x Height)"
            )

            with gr.Row():
                seed = gr.Number(label="Seed", value=42, precision=0)
                random_seed = gr.Checkbox(label="Random Seed", value=True)

            with gr.Row():
                steps = gr.Slider(label="Diffusion Sampling Steps", minimum=10, maximum=100, value=50, step=1)
                guidance_scale = gr.Slider(label="CFG Guidance Scale", minimum=1.0, maximum=15.0, value=7.5, step=0.5)

            generate_btn = gr.Button("Generate", variant="primary")

            gr.Markdown("### 📝 Example Prompts")
            gr.Examples(examples=EXAMPLE_PROMPTS, inputs=prompt_input, label=None)

        with gr.Column(scale=1):
            output_gallery = gr.Gallery(
                label="Generated Images",
                columns=2,
                rows=2,
                height=600,
                object_fit="contain",
                format="png",
                interactive=False,
            )
            used_seed = gr.Textbox(label="Seed Used", interactive=False)

    generate_btn.click(
        generate,
        inputs=[prompt_input, resolution, seed, steps, guidance_scale, random_seed, output_gallery],
        outputs=[output_gallery, used_seed, seed],
        api_visibility="public",
    )

css = """
.fillable{max-width: 1230px !important}
"""

if __name__ == "__main__":
    demo.launch(css=css, share=True)
