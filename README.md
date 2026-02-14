# BitDance: Scaling Autoregressive Generative Models with Binary Tokens

<p align="center">
  <a href="TBD">
    <img
      src="https://img.shields.io/badge/Project-Page-0A66C2?logo=chromewebstore&logoColor=0A66C2"
      alt="Project Page"
    />
  </a>
  <a href="TBD">
    <img
      src="https://img.shields.io/badge/arXiv paper-TBD-red?logo=arxiv&logoColor=red"
      alt="BitDance Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/collections/shallowdream204/bitdance">
    <img 
        src="https://img.shields.io/badge/Weights-BitDance-yellow?logo=huggingface&logoColor=yellow" 
        alt="BitDance Model"
    />
  </a>
  <a href="https://huggingface.co/spaces/shallowdream204/BitDance-14B-64x">
    <img 
        src="https://img.shields.io/badge/HF Space-Demo-orange?logo=huggingface&logoColor=yellow" 
        alt="BitDance Demo"
    />
  </a>
</p>

<p align="center"><img src="assets/speed.webp" width=90%"></p>


> [Yuang Ai*](https://shallowdream204.github.io/), [Jiaming Han*](https://csuhan.com/), [Shaobin Zhuang*](https://scholar.google.com/citations?user=PGaDirMAAAAJ), [Weijia Mao](https://scholar.google.com/citations?user=S7bGBmkyNtEC), [Xuefeng Hu](https://xuefenghu.me/), [Ziyan Yang](https://ziyanyang.github.io/), [Zhenheng Yang](https://zhenheny.github.io/), [Huaibo Huang†](https://hhb072.github.io/), [Xiangyu Yue†](https://xyue.io/), [Hao Chen*†‡](https://haochen-rye.github.io/)
>
> <sup>*</sup> Equal Contribution&nbsp;&nbsp;<sup>†</sup> Corresponding Author&nbsp;&nbsp;<sup>‡</sup> Project Lead
>
> For visual generation, discrete autoregressive models often struggle with poor tokenizer reconstruction, difficulties in sampling from large vocabularies, and slow token-by-token generation speeds. We present **BitDance**, which addresses these challenges via a large-vocabulary binary tokenizer, a binary diffusion head for sampling in large discrete space, and a next-patch diffusion paradigm that enables efficient multitoken prediction. BitDance is an open-source discrete autoregressive foundation model with 14B parameters, trained on large-scale multimodal tokens. While maintaining the standard language modeling paradigm for text tokens, BitDance employs a next-patch diffusion paradigm for visual tokens to predict multiple tokens in parallel—up to 64 per step. This unified multimodal framework is simple, scalable, and capable of efficiently generating high-resolution, photorealistic images.

<p align="center"><img src="assets/teaser.webp" width="90%"></p>

## 🔥 News
- **2026.2.14**: T2I inference code and models are released.


## ⚡ Quick Start

1️⃣ Create Conda Environment and Install Package
```bash
git clone shallowdream204/BitDance.git
cd BitDance
conda create -n bitdance python=3.11 -y
conda activate bitdance
pip install -r requirements.txt
pip install flash_attn==2.8.2 --no-build-isolation
```

2️⃣ Download Model Weights

We offer two models, BitDance-14B-64x and BitDance-14B-16x, which can predict 64 and 16 tokens in parallel at each step, respectively.
|  Model  | #Token per Step | Step (1024px) | Supported Size | Huggingface |
|:-------:|:----:|:----:|:-----------:|:----:|
| BitDance-14B-64x| 64 | 64 |1024px       | [BitDance-14B-64x](https://huggingface.co/shallowdream204/BitDance-14B-64x) |
| BitDance-14B-16x| 16 | 256 |512&1024px       | [BitDance-14B-16x](https://huggingface.co/shallowdream204/BitDance-14B-16x) |


```python
from huggingface_hub import snapshot_download

save_dir = "models/BitDance-14B-64x"
repo_id = "shallowdream204/BitDance-14B-64x"
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)

save_dir = "models/BitDance-14B-16x"
repo_id = "shallowdream204/BitDance-14B-16x"
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)

```

3️⃣ T2I Inference (check [here](modeling/t2i_pipeline.py#L22) for the supported image resolution)
```python
# example_t2i.py
from modeling.t2i_pipeline import BitDanceT2IPipeline

model_path = 'models/BitDance-14B-64x'
# model_path = 'models/BitDance-14B-16x'
device = 'cuda'

pipe = BitDanceT2IPipeline(model_path=model_path, device=device)

prompt = "A close-up portrait in a cinematic photography style, capturing a girl-next-door look on a sunny daytime urban street. She wears a khaki sweater, with long, flowing hair gently draped over her shoulders. Her head is turned slightly, revealing soft facial features illuminated by realistic, delicate sunlight coming from the left. The sunlight subtly highlights individual strands of her hair. The image has a Canon film-like color tone, evoking a warm nostalgic atmosphere."

image = pipe.generate(
    prompt=prompt,
    height=1024,
    width=1024,
    num_sampling_steps=50, # may adjust to 25 steps for faster inference, but may slightly reduce quality
    guidance_scale=7.5,
    num_images=1,
    seed=42
)[0]

image.save("example.png")
```

## 🤗 Demo

🔥 Try the Huggingface Space demo to start playing with BitDance: [BitDance-Demo](https://huggingface.co/spaces/shallowdream204/BitDance-14B-64x)

You can also run the demo locally:
```bash
python app.py
```

## 📸 Evaluation
We provide the scripts for evaluation on DPG Bench and GenEval. More benchmark evaluation scripts are coming soon.

1️⃣ Evaluation of BitDance-14B-64x Model
```bash
bash scripts/eval/eval_bitdance_14b_64x.sh
```
2️⃣ Evaluation of BitDance-14B-16x Model

```bash
bash scripts/eval/eval_bitdance_14b_16x.sh
```

Note you still need to follow the instructions in [DPG Bench](https://github.com/TencentQQGYLab/ELLA#-dpg-bench) and [GenEval](https://github.com/djghosh13/geneval) to evaluate the results.

## 🎰 Train
We are organizing the code related to data loading. The training instruction of BitDance is coming soon.

## 🪪 License

BitDance is licensed under the [Apache 2.0 license](LICENSE).

## 📖 Citation
If you find our work useful for your research, please consider citing our paper:
```bibtex
@article{ai2026bitdance,
  title   = {BitDance: Scaling Autoregressive Generative Models with Binary Tokens},
  author  = {Ai, Yuang and Han, Jiaming and Zhuang, Shaobin and Hu, Xuefeng and Yang, Ziyan and Yang, Zhenheng and Huang, Huaibo and Yue, Xiangyu and Chen, Hao},
  journal = {TBD},
  year    = {2026}
}
```