from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import base64
from process.prediction import dead_or_alive_black_percentage
from concurrent.futures import ThreadPoolExecutor
import os
import json 
upload_ethanol_image_router = APIRouter()

def analyze_ethanol_image(image_data, image_id):
    bounding_boxes, contours_list = dead_or_alive_black_percentage(image_data)
    response_content = {
        "image_id": image_id,
        "bounding_boxes": bounding_boxes,
        "contours_list": contours_list,
    }
    file_name = os.path.join("saved_json",f"{image_id}.json")
    os.makedirs("saved_json", exist_ok=True)
    with open(file_name, 'w') as json_file:
        json.dump(response_content, json_file)

executor = ThreadPoolExecutor(max_workers=4)

@upload_ethanol_image_router.post("/upload_image/ethanol_image/")
async def upload_ethanol_image(request: Request):
    try:
        body = await request.json()
        
        if "base64_image" not in body:
            raise HTTPException(status_code=400, detail="base64_image field is required")
        
        base64_image = body["base64_image"]
        image_id = body["image_id"]

        # Decode Base64 image
        image_data = base64.b64decode(base64_image)

        executor.submit(analyze_ethanol_image, image_data, image_id)
        
        response_content = {
            "message": "Image uploaded successfully"
        }
        
        return JSONResponse(content=response_content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
