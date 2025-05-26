import sys
from pathlib import Path

project_root = Path().absolute().parent
sys.path.append(str(project_root))

from flask import Flask, render_template, request, send_from_directory
import threading
import time
from src.diffusion.model import ConditionedDiffusionModelWithAction
from src.utils import load_config

import torch
import torchvision
import numpy as np
from PIL import Image

app = Flask(__name__)

# Store current prompt or other input
user_input = {"prompt": ""}

config = load_config("../config.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n\n")

print("Loading model...")
model = ConditionedDiffusionModelWithAction(image_size=config['game']['image_size'],
            in_channels=config['game']['in_channels'],
            out_channels=config['game']['out_channels'],
            device=device,
            timesteps=config['game']['timesteps'],
            beta_schedule=config['game']['beta_schedule'],           
            action_embedding_size=config['game']['action_embedding_size'],
            num_actions=config['game']['num_actions'],
            unet_weights_path=config['game']['model_path']
        )    

print(f"Model loaded successfully with type {type(model)}!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    user_input["prompt"] = data.get("prompt", "")
    return {"status": "received"}

@app.route('/keypress', methods=['POST'])
def keypress():
    data = request.json
    print(f"Key received: {data.get('key')}")
    return {"status": "ok"}

@app.route('/image')
def image():
    return send_from_directory('static', 'output.png')

def make_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.asarray(grid_im).astype(np.uint8))
    return grid_im
    
def generate_loop():
   
    print("Running up to date version of app.py")

    try:
        context_length = config['game']['context_length']

        frames = torch.load("init/init_frames.pt").to(device)
        actions = torch.load("init/init_actions.pt").to(device)

        while True:
            # prompt = user_input["prompt"]
                # Run your diffusion model here with the prompt
            # print(f"Generating for prompt: {prompt}")
            print("Generating new image")
            sample = model.generate_next_frame(frames, actions, num_inference_steps=30)
            print("Next frame shape:\t", sample.shape)
            image = make_images(sample)
            
            print("Saving new frame")
            image.save("./static/output.png", "PNG")

            next_c_frame = sample.unsqueeze(1)          # [B, 1, H, W]
            frames = torch.cat([frames[:, 1:, :, :], next_c_frame], dim=1).to(device)  # [B, C, H, W]
            
                # Save result to 'static/output.png'
                # model.generate_image(prompt, save_path="static/output.png")
            time.sleep(2)  # Adjust based on your model's speed
    except Exception as e:
        print(f"Error in generate_loop: {e}")

if __name__ == '__main__':
    threading.Thread(target=generate_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=7860)
