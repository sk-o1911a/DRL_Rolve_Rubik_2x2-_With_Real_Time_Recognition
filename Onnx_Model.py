import onnxruntime as ort
import numpy as np
import torch

class RubikONNXModel:
    def __init__(self, onnx_path="rubik2x2.onnx", device="cpu"):
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3

        # Tối ưu inference
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = (
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if device == "cuda"
            else ['CPUExecutionProvider']
        )

        try:
            self.session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        except Exception as e:
            print(f"Lỗi khởi tạo ONNX (chuyển sang CPU): {e}")
            self.session = ort.InferenceSession(
                onnx_path, sess_options, providers=['CPUExecutionProvider'])

        self.input_name   = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Buffer tái sử dụng — tránh alloc numpy mỗi lần gọi
        self._input_buf = np.zeros((1, 144), dtype=np.float32)

    def predict(self, x):
        """
        Nhận: torch.Tensor shape (24,6) hoặc (1,144) hoặc numpy tương đương
        Trả về: (policy_tensor, value_tensor) — torch.Tensor để MCTS_Core dùng được
        
        Tối ưu: tái sử dụng buffer, bỏ các bước convert trung gian thừa
        """
        # Convert sang numpy một lần duy nhất
        if isinstance(x, torch.Tensor):
            np_x = x.detach().cpu().numpy().reshape(1, -1).astype(np.float32)
        elif isinstance(x, np.ndarray):
            np_x = x.reshape(1, -1)
            if np_x.dtype != np.float32:
                np_x = np_x.astype(np.float32)
        else:
            np_x = np.array(x, dtype=np.float32).reshape(1, -1)

        # Chạy ONNX
        outputs       = self.session.run(self.output_names, {self.input_name: np_x})
        policy_logits = outputs[0]   # shape (1, 9)
        value         = outputs[1]   # shape (1, 1) hoặc (1,)

        # Softmax thuần numpy — không qua torch
        policy_probs = self._softmax(policy_logits)

        # Trả torch.Tensor để MCTS_Core không cần sửa
        policy_tensor = torch.from_numpy(policy_probs[0])       # shape (9,)
        value_scalar  = float(value.flat[0])
        value_tensor  = torch.tensor(value_scalar, dtype=torch.float32)

        return policy_tensor, value_tensor

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        # Numerically stable softmax
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
