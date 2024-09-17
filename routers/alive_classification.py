from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import base64
import cv2
import numpy as np
from io import BytesIO
from process.prediction import dead_or_alive_black_percentage

alive_classification_route = APIRouter()

@alive_classification_route.post("/alive_classification/")
async def alive_classification(request: Request):
    # try:
        # Read JSON data from request body
        body = await request.json()
        
        if "base64_image" not in body:
            raise HTTPException(status_code=400, detail="base64_image field is required")
        
        base64_image = body["base64_image"]
        image_id = body["image_id"]

        # Decode Base64 image
        image = base64.b64decode(base64_image)

        # Process the image using the prediction function
        bounding_boxes, contours_list, processed_image = dead_or_alive_black_percentage(image)

        response_content = {
            "image_id": image_id,
            "bounding_boxes": bounding_boxes,
            "contours_list": contours_list,
        }
        
        return JSONResponse(content=response_content)
        
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Error in image prediction: {e}")
