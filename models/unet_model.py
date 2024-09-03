from keras.models import load_model
import os

model_path = os.path.join('models', '256unet-non-aug.keras')

try:
    unet = load_model(model_path)
except Exception as e:
    print(f"Error loading U-Net model from {model_path}: {e}")
    unet = None
