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

