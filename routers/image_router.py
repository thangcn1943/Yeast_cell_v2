from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import base64
import numpy as np
from process_image import cut_unecessary_img, new_resize_image, predict_cell, split_image, merge_predictions
from PIL import Image
import io
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
        image = new_resize_image(image, 1280)
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
        merge_mask = merge_predictions(predictions)
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
        
        # Decode Base64 image
        image_data = base64.b64decode(base64_image)
        
        # Predict image and mask
        image, mask = predict_mask(image_data)
        
        # Predict cell types using the CNN model
        normal, abnormal, normal_2x, abnormal_2x, result, bounding_boxes = predict_cell(image, mask, cnn)
        
        response_content = {
            "cell_counts": {
                "normal": normal,
                "abnormal": abnormal,
                "normal_2x": normal_2x,
                "abnormal_2x": abnormal_2x
            },
            "bounding_boxes": bounding_boxes,
        }
        
        return JSONResponse(content=response_content)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
