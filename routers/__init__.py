from .normal_analyze import image_router
from .id_router import id_router
from .measure import measure_router
from .upload_ethanol_image import upload_ethanol_image_router
from .upload_normal_image import upload_normal_image_router
from .alive_classification import alive_classification_router
__all__ = ["normal_analyze", "id_router","measure_router", "upload_ethanol_image_router", "upload_normal_image_router", "alive_classification_router"]