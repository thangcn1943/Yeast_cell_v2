from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import json
import os

image_router = APIRouter()

@image_router.post("/request_normal_image/")
async def upload_image(request: Request):
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
                cell_counts = data["cell_counts"]
                bbox = data["bounding_boxes"]
                response = {
                    "cell_counts": cell_counts,
                    "bounding_boxes": bbox,
                }
                return JSONResponse(content=response)
                
            return JSONResponse(content=data)
        

        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
