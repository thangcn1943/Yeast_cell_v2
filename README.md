# Yeast cell detection
Dự án này sử dụng mô hình UNET để phát hiện, phân đoạn và phân loại các tế bào nấm men từ hình ảnh. Ứng dụng của dự án giúp cải thiện độ chính xác trong việc chẩn đoán các bệnh liên quan đến nấm men thông qua việc phân tích tự động các tế bào bất thường.
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
Cài đặt các thư viện từ tệp requirements.txt:

   ```bash
   pip install -r requirements.txt

Chạy mô hình phân đoạn và phân loại tế bào nấm men: 
Đảm bảo rằng dataset được tải về và nằm đúng vị trí được chỉ định trong mã nguồn.  

Chạy API với FastAPI:

Nếu dự án có API để trả về thông tin tế bào, bạn có thể chạy bằng lệnh:
   ```bash
   uvicorn app:app --reload


Truy cập vào http://localhost:8000/docs để xem tài liệu và thử nghiệm  
Có thể sử dụng postman và import để test  
