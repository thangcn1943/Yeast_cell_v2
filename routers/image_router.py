from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import base64
from process.prediction import predict_mask, predict_cell
import json
import os
from models.cnn_model import cnn

image_router = APIRouter()

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
