import subprocess
command = 'pip install git+https://github.com/snekkenull/diffusers.git'
subprocess.run(command, shell=True)

import os
import gradio as gr
import torch
import numpy as np
import random
from diffusers import StableDiffusion3Pipeline, AutoencoderKL, SD3Transformer2DModel, StableDiffusion3Img2ImgPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils import load_image
from PIL import Image
import requests
import transformers
from transformers import AutoTokenizer, T5EncoderModel
from translatepy import Translator


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
translator = Translator()
HF_TOKEN = os.environ.get("HF_TOKEN", None)
# Constants
model = "stabilityai/stable-diffusion-3-medium"
repo= "stabilityai/stable-diffusion-3-medium-diffusers"
MAX_SEED = np.iinfo(np.int32).max

CSS = """
.gradio-container {
  max-width: 690px !important;
}
footer {
    visibility: hidden;
}
"""

JS = """function () {
  gradioURL = window.location.href
  if (!gradioURL.endsWith('?__theme=dark')) {
    window.location.replace(gradioURL + '?__theme=dark');
  }
}"""


vae = AutoencoderKL.from_pretrained(
    repo,
    subfolder="vae",
    torch_dtype=torch.float16,
 )

transformer = SD3Transformer2DModel.from_pretrained(
    repo, 
    subfolder="transformer",
    torch_dtype=torch.float16,
)

# Ensure model and scheduler are initialized in GPU-enabled function
if torch.cuda.is_available():
    pipe = StableDiffusion3Pipeline.from_pretrained(
        repo,
        vae=vae,
        transformer=transformer,
        torch_dtype=torch.float16).to("cuda")
    pipe2 = StableDiffusion3Img2ImgPipeline.from_pretrained(
        repo,
        vae=vae,
        transformer=transformer,
        torch_dtype=torch.float16).to("cuda")



pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe2.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe2.scheduler.config)

print(pipe.tokenizer_max_length)

# Function 
def generate_image(
    prompt,
    negative="low quality",
    width=1024,
    height=1024,
    scales=5,
    steps=30,
    strength=0.7,
    seed: int =-1,
    nums=1,
    progress=gr.Progress(track_tqdm=True)):

    if seed == -1:
        seed = random.randint(0, MAX_SEED)
    seed = int(seed)
    print(f'prompt:{prompt}')
    
    text = str(translator.translate(prompt['text'], 'English'))

    
    if prompt['files']:
        #images = Image.open(prompt['files'][-1]).convert('RGB')
        init_image = load_image(prompt['files'][-1]).resize((height, width))
    else:
        init_image = None
    generator = torch.Generator().manual_seed(seed)
    

    if init_image:
        image = pipe2(
            prompt=text,
            image=init_image,
            negative_prompt=negative, 
            guidance_scale=scales,
            num_inference_steps=steps,
            strength=strength,
            generator = generator,
            num_images_per_prompt = nums,
        ).images
    else:
        image = pipe(
            prompt=text,
            negative_prompt=negative, 
            width=width,
            height=height,
            guidance_scale=scales,
            num_inference_steps=steps,
            generator = generator,
            num_images_per_prompt = nums,
        ).images

    print(image)
    print(seed)
    return image, seed


examples = [
        [{"text": "a female character with long, flowing hair that appears to be made of ethereal, swirling patterns resembling the Northern Lights or Aurora Borealis. The background is dominated by deep blues and purples, creating a mysterious and dramatic atmosphere. The character's face is serene, with pale skin and striking features. She wears a dark-colored outfit with subtle patterns. The overall style of the artwork is reminiscent of fantasy or supernatural genres", "files": []}],
        [{"text": "Digital art, portrait of an anthropomorphic roaring Tiger warrior with full armor, close up in the middle of a battle, behind him there is a banner with the text \"Open Source\".", "files": []}],
        [{"text": "photo of a dog and a cat both standing on a red box, with a blue ball in the middle with a parrot standing on top of the ball. The box has the text \"SD3\"", "files": []}],
		[{"text": "selfie photo of a wizard with long beard and purple robes, he is apparently in the middle of Tokyo. Probably taken from a phone.", "files": []}],
		[{"text": "A vibrant street wall covered in colorful graffiti, the centerpiece spells \"SD3 MEDIUM\", in a storm of colors", "files": []}],
		[{"text":  "photo of a young woman with long, wavy brown hair tied in a bun and glasses. She has a fair complexion and is wearing subtle makeup, emphasizing her eyes and lips. She is dressed in a black top. The background appears to be an urban setting with a building facade, and the sunlight casts a warm glow on her face.", "files": []}],
		[{"text": "anime art of a steampunk inventor in their workshop, surrounded by gears, gadgets, and steam. He is holding a blue potion and a red potion, one in each hand", "files": []}],
		[{"text": "photo of picturesque scene of a road surrounded by lush green trees and shrubs. The road is wide and smooth, leading into the distance. On the right side of the road, there's a blue sports car parked with the license plate spelling \"SD32B\". The sky above is partly cloudy, suggesting a pleasant day. The trees have a mix of green and brown foliage. There are no people visible in the image. The overall composition is balanced, with the car serving as a focal point.", "files": []}],
		[{"text": "photo of young man in a black suit, white shirt, and black tie. He has a neatly styled haircut and is looking directly at the camera with a neutral expression. The background consists of a textured wall with horizontal lines. The photograph is in black and white, emphasizing contrasts and shadows. The man appears to be in his late twenties or early thirties, with fair skin and short, dark hair.", "files": []}],
		[{"text": "photo of a woman on the beach, shot from above. She is facing the sea, while wearing a white dress. She has long blonde hair", "files": []}],
]



# Gradio Interface

with gr.Blocks(css=CSS, js=JS, theme="soft") as demo:
    gr.HTML("<h1><center>SD3M🐉T5</center></h1>")
    gr.HTML("<p><center><a href='https://huggingface.co/stabilityai/stable-diffusion-3-medium'>sd3m</a> text/image-to-image generation<br><b>Update</b>: fix diffuser to support 512 token</center></p>")
    with gr.Group():
        with gr.Row():
            prompt = gr.MultimodalTextbox(label='Enter Your Prompt (Multi-Languages)', interactive=True, placeholder="Enter prompt, add one image.", file_types=['image'], scale=1)
    img = gr.Gallery(label='SD3M Generated Image',columns = 1, preview=True)
    with gr.Accordion("Advanced Options", open=False):
        with gr.Row():
            negative = gr.Textbox(label="Negative prompt", value="low quality, ugly, blurry, poor face, bad anatomy")
        with gr.Row():
            width = gr.Slider(
                label="Width",
                minimum=512,
                maximum=1280,
                step=8,
                value=1024,
            )
            height = gr.Slider(
                label="Height",
                minimum=512,
                maximum=1280,
                step=8,
                value=1024,
            )
        with gr.Row():
            scales = gr.Slider(
                label="Guidance",
                minimum=3.5,
                maximum=7,
                step=0.1,
                value=5,
            )
            steps = gr.Slider(
                label="Steps",
                minimum=1,
                maximum=50,
                step=1,
                value=30,
            )
            strength = gr.Slider(
                label="Strength",
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=0.7,
            )
        with gr.Row():
            seed = gr.Slider(
                label="Seed (-1 Get Random)",
                minimum=-1,
                maximum=MAX_SEED,
                step=1,
                value=-1,
                scale=2,
            )
            nums = gr.Slider(
                label="Image Numbers",
                minimum=1,
                maximum=4,
                step=1,
                value=1,
                scale=1,
            )  
    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=[img, seed],
        fn=generate_image,
        cache_examples="lazy",
        examples_per_page=4,
    )

    prompt.submit(fn=generate_image,
                 inputs=[prompt, negative, width, height, scales, steps, strength, seed, nums],
                 outputs=[img, seed],
                 )

    
demo.queue().launch()