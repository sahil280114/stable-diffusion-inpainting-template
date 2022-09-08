import banana_dev as banana
import base64
from io import BytesIO
from PIL import Image

def imgToBase64String(filename):
    img = Image.open(filename)
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(im_bytes).decode('utf-8')
    return im_b64

init_image_string = imgToBase64String("init_image.jpg")
mask_image_string = imgToBase64String("mask_image.jpg")  
model_inputs = {"prompt":"","init_image_base64":init_image_string,"mask_image_base64":mask_image_string,"strength":0.6,"guidance_scale":75,"steps":50}


#Call model deployed on banana
api_key = "api_key"
model_key = "model_key"
output = banana.run(api_key,model_key,model_inputs)
output_image_string = output["modelOutputs"][0]["image_base64"]
image_encoded = output_image_string.encode('utf-8')
image_bytes = BytesIO(base64.b64decode(image_encoded))
image = Image.open(image_bytes)
image.save("output.jpg")

#Call the model locally
import requests
res = requests.post('http://localhost:8000/', json = model_inputs)

print(res.json())