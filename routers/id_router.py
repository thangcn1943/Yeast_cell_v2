from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import json
import os

id_router = APIRouter()

@id_router.post("/request_id/")
async def request_id(request: Request):
    try:
        body = await request.json()
        
        if "image_id" not in body or "cell_id" not in body:
            raise HTTPException(status_code=400, detail="image_id and cell_id fields are required")
        
        image_id = body["image_id"]
        cell_id = body["cell_id"]

        file_name = os.path.join("saved_json", f"{image_id}.json")

        if not os.path.exists(file_name):
            raise HTTPException(status_code=404, detail=f"File {file_name} not found")

        with open(file_name, "r") as file:
            data = json.load(file)
        
        if "contours_list" not in data:
            raise HTTPException(status_code=500, detail="Invalid JSON format: 'contours_list' field is missing")
        
        contours_list = data["contours_list"]

        cell_info = None
        
        for box in contours_list:
            if box["cell_id"] == cell_id:
                cell_info = box
                break
        
        if cell_info is None:
            raise HTTPException(status_code=404, detail=f"Cell ID {cell_id} not found in bounding boxes")
        
        return JSONResponse(content={"cell_info": cell_info})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in ID prediction: {e}")
