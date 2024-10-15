from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import json
import os

alive_classification_router = APIRouter()

@alive_classification_router.post("/request_ethanol_image/")
async def alive_classification(request: Request):
    try:
        # Read JSON data from request body
        body = await request.json()
        
        if "image_id" not in body:
            raise HTTPException(status_code=400, detail="image_id field is required")
        
        image_id = body["image_id"]
        
        file_name = os.path.join("saved_json", f"{image_id}.json")

        if not os.path.exists(file_name):
            response = {
                "message": "Image has not been analyzed yet"
            }
            return JSONResponse(content=response)
        else:
            with open(file_name, "r") as file:
                data = json.load(file)
                contours_list = data["contours_list"]
                bbox = data["bounding_boxes"]
                response = {
                    "bounding_boxes": bbox,
                    "contours_list": contours_list,
                }
                return JSONResponse(content=response)
                
            return JSONResponse(content=data)
        

        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))