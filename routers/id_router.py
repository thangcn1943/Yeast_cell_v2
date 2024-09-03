from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import base64
import numpy as np
from PIL import Image
import io

id_route = APIRouter()

@id_route.post("/request_id/")

async def request_id(request: Request):
    try:
        body = await request.json()
        if ("image_id" and "cell_id" ) not in body:
            raise HTTPException(status_code=400, detail="image_id field is required")
        
        image_id = body["image_id"]
        cell_id = body["cell_id"]
        
        
    except Exception as e:
        raise RuntimeError(f"Error in ID prediction: {e}")