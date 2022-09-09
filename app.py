from operator import mod
import torch
from torch import autocast
from diffusers import StableDiffusionInpaintPipeline
import base64
from io import BytesIO
import os
import PIL

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    HF_AUTH_TOKEN = ""
    
    model = StableDiffusionInpaintPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16,use_auth_token=HF_AUTH_TOKEN).to("cuda")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    init_image_base64 = model_inputs.get('init_image_base64', None)
    if init_image_base64==None:
        return {'message': "No init_image provided"}
    mask_image_base64 = model_inputs.get('mask_image_base64', None)
    if mask_image_base64==None:
        return {'message': "No mask_image provided"}
    strength = model_inputs.get("strength",0.8)
    guidance_scale = model_inputs.get("guidance_scale",7.5)
    steps = model_inputs.get("steps",50)

    init_image_encoded = init_image_base64.encode('utf-8')
    init_image_bytes = BytesIO(base64.b64decode(init_image_encoded))
    init_image = PIL.Image.open(init_image_bytes)

    mask_image_encoded = mask_image_base64.encode('utf-8')
    mask_image_bytes = BytesIO(base64.b64decode(mask_image_encoded))
    mask_image = PIL.Image.open(mask_image_bytes)

    # Run the model
    with autocast("cuda"):
        image = model(prompt,init_image=init_image,mask_image=mask_image,strength=strength,num_inference_steps=steps,guidance_scale=guidance_scale).images[0]
    
    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'image_base64': image_base64}
