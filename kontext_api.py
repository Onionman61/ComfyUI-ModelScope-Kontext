import requests
import time
import json
from PIL import Image
from io import BytesIO
import numpy as np
import torch

class ModelScopeUniversalAPI:
    """
    A ComfyUI custom node to call various ModelScope image generation APIs.
    This version supports both FLUX.1-Kontext-Dev (single image) and FLUX.2-dev (multi-image).
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # List of supported models for the dropdown menu
        s.supported_models = ['black-forest-labs/FLUX.2-dev', 'MusePublic/FLUX.1-Kontext-Dev']
        
        # Define the maximum seed value for the Kontext model
        MAX_SEED = 2147483647
        
        return {
            "required": {
                "model": (s.supported_models, {"default": s.supported_models[0]}),
                "image_1": ("IMAGE",),
                "api_key": ("STRING", {"multiline": False, "default": "your_modelscope_api_key"}),
                "prompt": ("STRING", {"multiline": True, "default": "Give the dog in the picture a birthday hat"}),
                
                # These parameters are specific to the Kontext model, but we keep them for convenience.
                # The code will ignore them if FLUX.2-dev is selected.
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 1.5, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "ModelScope" # Updated category for clarity

    def tensor_to_pil(self, tensor):
        image_np = tensor.cpu().numpy().squeeze()
        if image_np.ndim == 3 and image_np.shape[0] in [1, 3, 4]:
             image_np = image_np.transpose(1, 2, 0)
        image_np = (image_np * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    def pil_to_tensor(self, pil_image):
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

    def upload_image_to_host(self, pil_image, image_number):
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        buffered.seek(0)
        url = 'https://freeimage.host/api/1/upload'
        api_key = '6d207e02198a847aa98d0a2a901485a5'
        payload = {'key': api_key, 'action': 'upload', 'format': 'json'}
        files = {'source': (f'image_{image_number}.png', buffered, 'image/png')}
        try:
            print(f"Uploading image #{image_number} to freeimage.host...")
            response = requests.post(url, data=payload, files=files, timeout=60)
            response.raise_for_status()
            result = response.json()
            if result.get("status_code") == 200 and result.get("image"):
                image_url = result["image"].get("url")
                print(f"Successfully uploaded image #{image_number}: {image_url}")
                return image_url
            else:
                raise Exception(f"Failed to upload image #{image_number}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Image hosting service connection error for image #{image_number}: {e}")

    def generate_image(self, model, image_1, api_key, prompt, width, height, seed, steps, guidance, image_2=None, image_3=None):
        
        # --- Step 1: Upload all provided images ---
        image_urls = []
        all_images = [(1, image_1), (2, image_2), (3, image_3)]
        
        for i, img_tensor in all_images:
            if img_tensor is not None:
                pil_img = self.tensor_to_pil(img_tensor)
                url = self.upload_image_to_host(pil_img, i)
                image_urls.append(url)
        
        if not image_urls:
            raise Exception("No valid images provided to upload.")

        # --- Step 2: Build the API payload based on the selected model ---
        payload = { "model": model, "prompt": prompt }

        if model == 'MusePublic/FLUX.1-Kontext-Dev':
            # This is the old Kontext model, which takes one image and extra parameters.
            print("Using FLUX.1-Kontext-Dev model. Applying size, seed, steps, and guidance.")
            MAX_SEED = 2147483647
            seed_to_use = seed % (MAX_SEED + 1)
            
            payload["image_url"] = image_urls[0] # Takes only the first image URL as a string
            payload["size"] = f"{width}x{height}"
            payload["seed"] = seed_to_use
            payload["steps"] = steps
            payload["guidance"] = guidance
        
        elif model == 'black-forest-labs/FLUX.2-dev':
            # This is the new model, which takes a list of images and has fewer parameters.
            print("Using FLUX.2-dev model. Ignoring size, seed, steps, and guidance parameters.")
            payload["image_url"] = image_urls # Takes all image URLs as a list
        
        else:
            raise Exception(f"Unknown model selected: {model}")

        print(f"Sending payload to ModelScope: {payload}")

        # --- Step 3: Call the API and poll for results (this part is the same for both) ---
        base_url = 'https://api-inference.modelscope.cn/'
        common_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                f"{base_url}v1/images/generations",
                headers={**common_headers, "X-ModelScope-Async-Mode": "true"},
                data=json.dumps(payload, ensure_ascii=False).encode('utf-8')
            )
            response.raise_for_status()
            task_id = response.json()["task_id"]
            print(f"ModelScope task started with ID: {task_id}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to start ModelScope task: {e}")
        
        while True:
            try:
                result_response = requests.get(
                    f"{base_url}v1/tasks/{task_id}",
                    headers={**common_headers, "X-ModelScope-Task-Type": "image_generation"},
                )
                result_response.raise_for_status()
                data = result_response.json()

                if data["task_status"] == "SUCCEED":
                    output_image_url = data["output_images"][0]
                    print(f"Image generation successful: {output_image_url}")
                    image_response = requests.get(output_image_url)
                    image_response.raise_for_status()
                    result_image = Image.open(BytesIO(image_response.content)).convert("RGB")
                    return (self.pil_to_tensor(result_image),)
                
                elif data["task_status"] == "FAILED":
                    raise Exception(f"ModelScope image generation failed. Reason: {data.get('message', 'Unknown error')}")

            except requests.exceptions.RequestException as e:
                raise Exception(f"Error while checking ModelScope task status: {e}")

            time.sleep(5)

# Required by ComfyUI to map the node class
NODE_CLASS_MAPPINGS = {
    "ModelScopeUniversalAPI": ModelScopeUniversalAPI
}

# Required by ComfyUI to display a friendly name
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelScopeUniversalAPI": "ModelScope Universal API"
}