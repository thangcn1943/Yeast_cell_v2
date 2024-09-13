from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import base64
import numpy as np
from process_image import cut_unecessary_img, resize_image, predict_cell, split_image, merge_images
from PIL import Image
import io
import json
import os
from models.unet_model import unet
from models.cnn_model import cnn

image_router = APIRouter()

def predict_mask(image_data):
    try:
        # Decode and preprocess image
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)
        
        # Perform image processing
        image = cut_unecessary_img(image)
        image = resize_image(image, image[0][0].tolist())
        image_normalize = image.astype(np.float32) / 255.0
        
        # Split image into patches
        image_array = split_image(image_normalize)
        
        # Predict using U-Net model
        predictions = []
        batch_size = 1  

        for i in range(0, len(image_array), batch_size):
            batch = np.array(image_array[i:i+batch_size])
            batch_predictions = unet.predict(batch)
            predictions.extend(batch_predictions)
        
        predictions = np.array(predictions)
        merge_mask = merge_images(image,predictions)
        merge_mask = (merge_mask > 0.5).astype(np.uint8) * 255
        
        return image, merge_mask
    except Exception as e:
        raise RuntimeError(f"Error in image prediction: {e}")
    
@image_router.post("/request_image/")
async def upload_image(request: Request):
    try:
        # Read JSON data from request body
        body = await request.json()
        
        if "base64_image" not in body:
            raise HTTPException(status_code=400, detail="base64_image field is required")
        
        base64_image = body["base64_image"]
        image_id = body["image_id"]
        # Decode Base64 image
        image_data = base64.b64decode(base64_image)
        
        # Predict image and mask
        image, mask = predict_mask(image_data)
        
        # Predict cell types using the CNN model
        normal, abnormal, normal_2x, abnormal_2x, bounding_boxes,contours_list = predict_cell(image, mask, cnn)
        
        response_content = {
            "image_id": image_id,
            "cell_counts": {
                "normal": normal,
                "abnormal": abnormal,
                "normal_2x": normal_2x,
                "abnormal_2x": abnormal_2x
            },
            "bounding_boxes": bounding_boxes,
            "contours_list": contours_list
        }
        file_name = os.path.join("saved_json",f"{image_id}.json")
        with open(file_name, 'w') as json_file:
            json.dump(response_content, json_file)
        response_only_bounding_boxes = {
            "image_id": image_id,
            "cell_counts": {
                "normal": normal,
                "abnormal": abnormal,
                "normal_2x": normal_2x,
                "abnormal_2x": abnormal_2x
            },
            "bounding_boxes": bounding_boxes
        }
        return JSONResponse(content = response_only_bounding_boxes)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
