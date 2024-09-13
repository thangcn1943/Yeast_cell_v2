import cv2
import numpy as np
import json
from process_image import find_space,Finding_ans
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import base64
from PIL import Image
import io

measure_router = APIRouter()

@measure_router.post("/measure/")
async def pixel_to_um_4_times(request: Request):
    try:
        body = await request.json()
        if "base64_image_measure" not in body:
            raise HTTPException(status_code=400, detail="base64_image_measure field is required")
        base64_image_measure = body["base64_image_measure"]
        
        image_measure = base64.b64decode(base64_image_measure)
        image = Image.open(io.BytesIO(image_measure))
        image = np.array(image)
        
        Dark_Distance=[]
        White_Distance=[]
        Is_Dark=125 # Checking whether a pixel is dark or white in grayscale: 0->255
        for turn in range(4):
            cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            height, width, channels = image.shape 
            # Convert image to grayscale
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #cv2.imshow("gray",gray)
                
            # Use canny edge detection
            edges = cv2.Canny(gray,50,150,apertureSize=3)

            # Apply HoughLinesP method to 
            # to directly obtain line end points
            #lines_list =[]
            lines = cv2.HoughLinesP(
                        edges, # Input edge image
                        1, # Distance resolution in pixels 1
                        np.pi/2160, # Angle resolution in radians  np.pi/180              |  2160
                        threshold=100, # Min number of votes for valid line  100          |  100
                        minLineLength=100, # Min allowed length of line 5                 |  100
                        maxLineGap=300 # Max allowed gap between line for joining them 10 | 300
                        )

            # Iterate over points
            for points in lines:
                # Extracted points nested in the list
                x1,y1,x2,y2=points[0]
                # Draw the lines joing the points
                # On the original image
                cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
                White,Dark=find_space(x1,y1,x2,y2,height,width,Is_Dark,gray)
                White_Distance.extend(White)
                Dark_Distance.extend(Dark)
                # Maintain a simples lookup list for points
                #lines_list.append([(x1,y1),(x2,y2)])

        White_ans=Finding_ans(White_Distance,26,32)
        Dark_ans=Finding_ans(Dark_Distance,9,14)

        # Save the result image
        #cv2.imshow("result",image)
        #cv2.waitKey(0)
        #cv2.imwrite('Result.jpg',image) #<= origin image with green lines
        #cv2.imwrite('Result Gray.jpg',gray) #<= gray image

        return White_ans*4+Dark_ans*4
    except Exception as e:
        return {"error": str(e)}



