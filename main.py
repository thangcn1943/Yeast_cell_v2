from routers.image_router import image_router
from routers.id_router import id_router
from routers.measure import measure_router
from fastapi import FastAPI

app = FastAPI()

app.include_router(image_router)
app.include_router(id_router)
app.include_router(measure_router)