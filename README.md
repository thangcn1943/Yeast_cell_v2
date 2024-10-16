# Yeast-Cell-Detection
## Link Datasets: https://s.net.vn/n2O9

## I. Giới thiệu bài toán:
&ensp;&ensp;&ensp;Bài toán đặt ra là phát hiện và phân loại tế bào nấm men có trong ảnh chụp một mẫu nấm men.

![image](https://github.com/user-attachments/assets/d3f6028f-357d-49a3-aac8-fe4020980e90)

<p>&ensp;&ensp;&ensp;Cần phân loại các tế bào nấm men thành 4 loại:</p>
<p>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;Tế bào đơn, bình thường:  </p> 

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![image](https://github.com/user-attachments/assets/6c86edea-d819-4adf-9003-67501c3c780b)

<p>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;Tế bào đơn, bất thường:  </p> 

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![image](https://github.com/user-attachments/assets/cc669829-738a-44d2-8069-6ddb36e835ec)


<p>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;Tế bào kép, bình thường:  </p> 

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![image](https://github.com/user-attachments/assets/aff5b79e-e764-4d91-9c44-2e68d1bfbd4d)


<p>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;Tế bào kép, bất thường:  </p> 

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![image](https://github.com/user-attachments/assets/9374aacc-6e74-4f2d-b6d9-ffeddba42586)


## II. Tiền xử lý dữ liệu:
&ensp;&ensp;&ensp;Từ ảnh gốc ban đầu được gán nhãn, sử dụng một sô kỹ thuật xử lý ảnh để có thể tách ra được các tế bào đã được gán nhãn,sau đó kiểm tra màu sắc có trong ảnh để phân loại các tế bào này vào các phân loại của nó.

## III. Huấn luyện mô hình CNN:
&ensp;&ensp;&ensp; Xây dựng mô hình CNN để phân loại 4 loại tế bào:

![image](https://github.com/user-attachments/assets/a8f9a018-d128-4118-bffc-84c013eeaa17)

## IV. Huấn luyện UNET để dự đoán mask của ảnh:
&ensp;&ensp;&ensp; Xây dựng và huấn luyện một mô hình UNET để dự đoán mask của ảnh đầu vào:

![image](https://github.com/user-attachments/assets/ea788cd9-be7b-4e7a-b708-e7c80b4c88b1)
<p>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Ảnh gốc</p>

![image](https://github.com/user-attachments/assets/ba09b0eb-3823-4b5c-8454-9ee3f3c51f11)
<p>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Mask</p>
## IV. Dự đoán:

&ensp;&ensp;&ensp;Sử dụng UNET đã huấn luyện để dự đoán ra mask của ảnh, sau đó CNN để phân loại từng tế bào

![image](https://github.com/user-attachments/assets/7f52f81b-4405-4c1b-9767-6e165b5d543d)
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
