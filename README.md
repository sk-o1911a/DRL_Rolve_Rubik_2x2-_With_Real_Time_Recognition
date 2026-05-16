# Deep Reinforcement Learning for Rubik's 2x2 Cube with Real-Time Recognition

## Thông Tin Đồ Án

**Trường Đại học:** Trường Đại học Tôn Đức Thắng

**Loại đồ án:** Đồ án chuyên ngành nâng cao

**Họ và tên:** Nguyễn Quốc Khánh

**MSSV:** 42200211

---

## Mô Tả Dự Án

Dự án này kết hợp **Deep Reinforcement Learning (DRL)** với **Real-Time Recognition** để tự động giải quyết Rubik's Cube 2x2. Hệ thống sử dụng:

- **Deep Reinforcement Learning**: Huấn luyện agent để tìm ra cách tối ưu để giải Rubik's cube
- **Real-Time Recognition**: Nhận dạng trạng thái hiện tại của Rubik's cube từ hình ảnh
- **Computer Vision**: Xử lý hình ảnh và phát hiện màu sắc

---

## Mục Tiêu

- Xây dựng mô hình DRL để học cách giải Rubik's 2x2 cube một cách tự động
- Phát triển hệ thống nhận dạng thời gian thực trạng thái của cube từ camera
- Tích hợp hai hệ thống để tạo ra một giải pháp hoàn chỉnh

---

## Công Nghệ Sử Dụng

- **Ngôn ngữ lập trình:** Python
- **Deep Learning Framework:** TensorFlow / PyTorch
- **Computer Vision:** OpenCV
- **RL Framework:** Stable Baselines3 / Ray RLLib
- **Xử lý dữ liệu:** NumPy, Pandas
- **Trực quan hóa:** Matplotlib, Plotly

---

## Cấu Trúc Thư Mục

```
DRL_Rolve_Rubik_2x2-_With_Real_Time_Recognition/
├── README.md                          # File mô tả dự án
├── requirements.txt                   # Các thư viện cần thiết
├── data/                              # Dữ liệu huấn luyện
│   ├── training_data/
│   └── test_data/
├── src/                               # Mã nguồn chính
│   ├── __init__.py
│   ├── cube_env.py                   # Môi trường Rubik's cube
│   ├── drl_agent.py                  # Model DRL
│   ├── vision.py                     # Nhận dạng hình ảnh
│   └── main.py                       # Chương trình chính
├── models/                            # Lưu trữ model đã huấn luyện
├── notebooks/                         # Jupyter notebooks
│   ├── exploration.ipynb
│   └── results_analysis.ipynb
├── logs/                              # Logs và kết quả huấn luyện
└── config/                            # File cấu hình

```

---

## Cài Đặt

### Yêu Cầu
- Python 3.8+
- pip hoặc conda

### Bước 1: Clone Repository
```bash
git clone https://github.com/sk-o1911a/DRL_Rolve_Rubik_2x2-_With_Real_Time_Recognition.git
cd DRL_Rolve_Rubik_2x2-_With_Real_Time_Recognition
```

### Bước 2: Tạo Virtual Environment (tùy chọn nhưng khuyên dùng)
```bash
python -m venv venv
source venv/bin/activate  # Trên Linux/Mac
# hoặc
venv\Scripts\activate     # Trên Windows
```

### Bước 3: Cài Đặt Dependencies
```bash
pip install -r requirements.txt
```

---

## Sử Dụng

### Huấn Luyện Mô Hình
```bash
python src/main.py --mode train --episodes 10000
```

### Kiểm Tra Mô Hình
```bash
python src/main.py --mode test --model_path models/best_model.zip
```

### Sử Dụng Real-Time Recognition
```bash
python src/main.py --mode demo --camera 0
```

---

## Kết Quả

- **Success Rate:** over 95%
- **Average Steps:** 16

---

## Tài Liệu Tham Khảo

- [OpenAI Gym Documentation](https://gym.openai.com/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Deep Reinforcement Learning Papers](https://arxiv.org/)
