# Deep Reinforcement Learning for Rubik's 2x2 Cube with Real-Time Recognition

## Thông Tin Đồ Án

**Trường Đại học:** Trường Đại học Tôn Đức Thắng

**Loại đồ án:** Đồ án chuyên ngành nâng cao

**Họ và tên:** Nguyễn Quốc Khánh

**MSSV:** 42200211

---

## Mô Tả Dự Án

Dự án này kết hợp **Deep Reinforcement Learning (DRL)** với **Real-Time Recognition** để tự động giải quyết Rubik's Cube 2x2. Hệ thống sử dụng:

- **Deep Reinforcement Learning**: Sử dụng MCTS (Monte Carlo Tree Search) và PBT (Population Based Training) để huấn luyện agent giải Rubik's cube
- **Real-Time Recognition**: Nhận dạng trạng thái cube từ camera sử dụng ResNet
- **Computer Vision**: Xử lý hình ảnh, calibrate camera và phát hiện màu sắc
- **Solver Algorithms**: Triển khai các thuật toán tìm kiếm (BFS, Dijkstra, A*)

---

## Mục Tiêu

- Xây dựng mô hình DRL sử dụng MCTS để học cách giải Rubik's 2x2 cube một cách tự động
- Phát triển hệ thống nhận dạng thời gian thực trạng thái cube từ camera
- Tích hợp DRL agent, computer vision và các solver algorithms
- Đạt được success rate > 95% và giải cube trong trung bình 16 bước

---

## Công Nghệ Sử Dụng

- **Ngôn ngữ lập trình:** Python
- **Deep Learning Framework:** PyTorch (torch 2.9.1)
- **RL Framework:** Ray RLLib (ray 2.53.0)
- **Computer Vision:** OpenCV (qua camera.py và camera_calibrate.py)
- **Neural Network:** ResNet
- **Model Format:** ONNX (rubik2x2.onnx)
- **Game Engine:** Pygame (pygame 2.6.1)
- **Xử lý dữ liệu:** NumPy 2.4.2, Pandas
- **Tối ưu hóa:** Numba 0.63.1
- **Phân tích đồ thị:** Matplotlib 3.10.8

---

## Cấu Trúc Thư Mục

```
DRL_Rolve_Rubik_2x2-_With_Real_Time_Recognition/
├── README.md                          # File mô tả dự án
├── requirements.txt                   # Các thư viện cần thiết
├── .gitignore                         # File gitignore
│
├── Core Training & Algorithms
│   ├── Rubik2x2Env.py                # Môi trường Rubik's 2x2 cube
│   ├── PBT_main.py                   # Population Based Training - chương trình huấn luyện chính
│   ├── MCTS_Core.py                  # Monte Carlo Tree Search core algorithm
│   ├── Self_Play_mp.py               # Self-play training với multiprocessing
│   ├── Train_Network.py              # Huấn luyện neural network
│   └── Action_MCTS.py                # Action selection using MCTS
│
├── Solver Algorithms
│   ├── AStar_Solver.py               # A* algorithm solver
│   ├── BFS_Solver.py                 # Breadth-First Search solver
│   ├── Dijkstra_Solver.py            # Dijkstra algorithm solver
│   └── benchmark.py                  # Benchmark các thuật toán solver
│
├── Vision & Recognition
│   ├── Resnet.py                     # ResNet model cho vision
│   ├── Normalize.py                  # Normalization utilities
│   ├── camera.py                     # Camera capture và processing
│   ├── camera_calibrate.py           # Camera calibration
│   └── Plot_Scatter.py               # Visualization utilities
│
├── Model & Inference
│   ├── Onnx_Model.py                 # ONNX model loading và inference
│   ├── export_onnx.py                # Export model to ONNX format
│   ├── rubik2x2.onnx                 # Pre-trained ONNX model
│   └── evaluate.py                   # Evaluation script
│
├── Demo & Application
│   ├── PyGame_Onnx.py                # Game demo sử dụng ONNX model
│
├── Logs & Checkpoints
│   ├── pbt_checkpoints/              # Lưu trữ PBT model checkpoints
│   ├── pbt_logs/                     # PBT training logs
│   └── training_logs_2x2/            # Training logs 2x2 cube

```

---

## Cài Đặt

### Yêu Cầu

- Python 3.8 trở lên
- GPU (khuyến nghị): CUDA 11.8+ để sử dụng ONNX Runtime GPU
- pip hoặc conda

### Bước 1: Clone Repository

```bash
git clone https://github.com/sk-o1911a/DRL_Rolve_Rubik_2x2-_With_Real_Time_Recognition.git
cd DRL_Rolve_Rubik_2x2-_With_Real_Time_Recognition
```

### Bước 2: Tạo Virtual Environment (khuyến nghị)

```bash
python -m venv venv

# Trên Linux/Mac:
source venv/bin/activate

# Trên Windows:
venv\Scripts\activate
```

### Bước 3: Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý về ONNX Runtime:**
- Nếu có GPU NVIDIA: `onnxruntime-gpu` sẽ được cài đặt tự động
- Nếu không có GPU hoặc gặp lỗi, cài thủ công: `pip install onnxruntime`

---

## Sử Dụng

### 1. Huấn Luyện Mô Hình (PBT)

```bash
python PBT_main.py
```

**Các tham số tùy chỉnh:**
- Chỉnh sửa trong `PBT_main.py` để thay đổi:
  - Số episodes huấn luyện
  - Số population agents
  - Learning rate
  - Hyperparameters MCTS

### 2. Kiểm Tra & Đánh Giá

```bash
python evaluate.py
```

### 3. Benchmark Các Solver Algorithms

```bash
python benchmark.py
```

So sánh hiệu suất của A*, BFS, Dijkstra và DRL agent

### 4. Demo với Real-Time Recognition

```bash
python PyGame_Onnx.py
```

**Tính năng:**
- Hiển thị Rubik's 2x2 cube 3D
- Sử dụng ONNX model để nhận dạng trạng thái
- Giải cube tự động sử dụng DRL agent

### 5. Camera Calibration (nếu cần)

```bash
python camera_calibrate.py
```

Calibrate camera để cải thiện độ chính xác nhận dạng màu sắc

### 6. Export Model sang ONNX

```bash
python export_onnx.py
```

Xuất PyTorch model sang ONNX format cho inference nhanh hơn

---

## Kết Quả

- **Success Rate:** over 95%
- **Average Steps:** 16 bước để giải cube
- **Training Framework:** Population Based Training (PBT) với 128 agents
- **Model Architecture:** ResNet + MCTS
- **Inference Speed:** Thực thời gian với GPU ONNX Runtime

---

## Tài Liệu Tham Khảo

- [Ray RLLib Documentation](https://docs.ray.io/en/latest/rllib/index.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Gymnasium (Gym) Documentation](https://gymnasium.farama.org/)
- [Monte Carlo Tree Search Paper](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
- [Population Based Training Paper](https://arxiv.org/abs/1711.09846)
