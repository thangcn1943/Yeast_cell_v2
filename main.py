from routers.image_router import image_router
from routers.id_router import id_router
from routers.measure import measure_router
from routers.alive_classification import alive_classification_route
from routers.upload_image import upload_image_router
from fastapi import FastAPI

app = FastAPI()

app.include_router(image_router)
app.include_router(id_router)
app.include_router(measure_router)
app.include_router(alive_classification_route)
app.include_router(upload_image_router)