# Deployment Instructions for Onrender

## Environment Variables

Trong Onrender dashboard, thêm environment variable sau:

```
APP_URL=https://your-app-name.onrender.com
```

(Thay `your-app-name` bằng tên app thực tế của bạn)

## Cách hoạt động

### 1. Data Refresh (Mỗi 30 phút)

- API tự động reload dữ liệu từ database mỗi 30 phút
- Đảm bảo recommendations luôn được cập nhật với dữ liệu mới nhất

### 2. Keep-Alive (Mỗi 14 phút)

- API tự động ping endpoint `/health` mỗi 14 phút
- Ngăn Onrender free tier đưa service vào chế độ sleep (sau 15 phút không hoạt động)
- **Lưu ý**: Free tier của Onrender vẫn có giới hạn 750 giờ/tháng

## Endpoints

### Health Check

```
GET /health
```

Trả về status của service

### Recommendations

```
GET /recommend/{user_id}?top_n=10
```

Trả về top N recommendations cho user

## Start Command

```
uvicorn api:app --host 0.0.0.0 --port $PORT
```

## Build Command (nếu cần)

```
pip install -r requirements.txt
```
