import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from typing import List
import io
import os
from pyngrok import ngrok
import uvicorn
import nest_asyncio




DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ALLOW_CUDA = False
# ALLOW_MPS = False

# if torch.cuda.is_available() and ALLOW_CUDA:
#     DEVICE = "cuda"
# elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
#     DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer("/content/Pytorch/stable_diffusion/data/vocab.json",
                          merges_file="/content/Pytorch/stable_diffusion/data/merges.txt")
model_file = "/content/Pytorch/stable_diffusion/data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

## TEXT TO IMAGE
do_cfg = True
cfg_scale = 8  # min: 1, max: 14


# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
strength = 0.9

## SAMPLER
sampler = "ddpm"
num_inference_steps = 50
seed = 42


app = FastAPI()


@app.post("/generate")
async def generate(promptStr: str):
    """ Create API endpoint to send image to and specify
     what type of file it'll take

    promptStr: Get and String and generate the image (text --> image)
    """

    output_image = pipeline.generate(
    prompt=promptStr,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
)
    im = Image.fromarray(output_image)
    im.save("result.png")
    return StreamingResponse(io.BytesIO(output_image.tobytes()),
                                 media_type="image/png")


auth_token = "" #put your token auth from your ngok account in the site

# Authenticate ngrok
# https://dashboard.ngrok.com/signup
os.system(f"ngrok authtoken {auth_token}")

# Create tunnel
public_url = ngrok.connect(8000)


# Allow for asyncio to work within the Jupyter notebook cell
nest_asyncio.apply()


# Run the FastAPI app using uvicorn
print(public_url)
uvicorn.run(app)

#disconnect the tunnel
#ngrok.disconnect(public_url=public_url)
