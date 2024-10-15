# Yeast cell detection
Dự án này sử dụng mô hình UNET để phát hiện, phân đoạn và phân loại các tế bào nấm men từ hình ảnh. Ứng dụng của dự án giúp cải thiện độ chất lượng của bia, các sản phẩm lên men liên quan đến nấm men thông qua việc phân tích tự động các tế bào bất thường.
## Dataset

Dataset cho dự án có thể được tải xuống từ liên kết dưới đây:

- [Link tải Dataset](https://drive.google.com/drive/folders/1XESHkHmGj8op8PaZ2ETmSZWLJJdJxhIk?usp=drive_link) 

## Cài đặt Thư viện

Để chạy dự án, bạn cần cài đặt các thư viện cần thiết. Thực hiện các bước dưới đây:

1. **Tạo môi trường ảo** (khuyến khích):

```bash
python -m venv env
source env/bin/activate  # Với Linux/MacOS
env\Scripts\activate  # Với Windows
```
### Cài đặt các thư viện từ tệp requirements.txt:

```bash
pip install -r requirements.txt
```
### Training unet with keras
```bash
unet_keras.ipynb
```
### Transfer learning cnn with pytorch
```bash
cnn_training_pytorch.ipynb
```
### Chạy API với FASTAPI
```bash
uvicorn main:app --reload
```
### Truy cập để đọc hướng dẫn và thử nghiệm
```bash
http://localhost:8000/docs
```
Có thể import file json ở trên trong postman để test các api
