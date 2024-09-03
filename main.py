from routers.image_router import image_router
from fastapi import FastAPI

app = FastAPI()

app.include_router(image_router)

