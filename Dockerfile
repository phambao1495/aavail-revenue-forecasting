# Sử dụng Python 3.10 bản gọn nhẹ
FROM python:3.10-slim

# Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# Ngăn Python tạo file .pyc và giúp log in ra màn hình ngay lập tức
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Copy file thư viện và cài đặt trước (tận dụng cache của Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn, dữ liệu, models và tests vào container
COPY . .

# Chạy toàn bộ unit tests ngay khi build để đảm bảo container hoạt động đúng
RUN python run_tests.py

# Mở cổng 8080 cho Flask API
EXPOSE 8080

# Lệnh khởi chạy Flask server khi container bắt đầu
CMD ["python", "app.py"]