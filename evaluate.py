import os
import json
import torch
import matplotlib.pyplot as plt

from Resnet import ResNetPolicyValueNet
from Self_Play_mp import SelfPlayManager  # Đã thay đổi import theo kiến trúc mới
from Rubik2x2Env import NUM_ACTIONS

class ScrambleEvaluator:
    def __init__(
        self,
        checkpoint_path="pbt_checkpoints/best_latest.pt",
        result_json_path=os.path.join("training_logs_2x2", "eval_scramble_result.json"),
        n_runs=100,
        device=None,
        num_workers=7, # Thêm tham số quản lý số worker
    ):
        self.checkpoint_path = checkpoint_path
        self.result_json_path = result_json_path
        self.n_runs = n_runs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = num_workers
        self.model = None
        self.results = []

    def load_model(self):
        model = ResNetPolicyValueNet(
            num_colors=6,
            hidden_dim=192,
            num_res_blocks=8,
            num_actions=NUM_ACTIONS,
            dropout=0.05
        ).to(self.device)

        if os.path.exists(self.checkpoint_path):
            print(f"[evaluate] Loading checkpoint: {self.checkpoint_path}")
            state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            
            # Xử lý trường hợp weight có prefix _orig_mod (do torch.compile tạo ra trước đó)
            if hasattr(model, "_orig_mod"):
                clean_state = {}
                for k, v in state.items():
                    if k.startswith("_orig_mod."):
                        clean_state[k] = v
                    else:
                        clean_state[f"_orig_mod.{k}"] = v
                model.load_state_dict(clean_state)
            else:
                model.load_state_dict(state)
                
            model.eval()
        else:
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")
        self.model = model

    def evaluate_scramble_range(self, min_len=1, max_len=20, num_simulations=500, max_episode_steps=40):
        if self.model is None:
            self.load_model()
        self.results = []
        
        best_hp_path = os.path.join("pbt_logs", "best_hyperparams.json")
        if os.path.exists(best_hp_path):
            with open(best_hp_path, "r") as f:
                best_c_puct = json.load(f)["hyperparams"]["c_puct"]
        else:
            best_c_puct = 3.5
        sp_manager = SelfPlayManager(num_workers=self.num_workers)

        try:
            sp_manager.update_weights(self.model)

            for scramble_len in range(min_len, max_len + 1):
                print(f"\n==> Đang đánh giá scramble_len = {scramble_len} ...")

                sp_manager.update_configs(
                    scramble_len=scramble_len, 
                    max_steps=max_episode_steps, 
                    num_simulations=num_simulations,
                    mcts_hyperparams={"c_puct": best_c_puct},
                    color_augmentation=True
                )
                output = sp_manager.collect_data(
                    num_episodes=self.n_runs,
                    select_mode="greedy", 
                    scramble_len_display=scramble_len,
                    add_noise=False
                )
                
                # output[0] là data, output[1] luôn là solve_rate theo chuẩn code hiện tại
                solve_rate = output[1]

                self.results.append({
                    "scramble_len": scramble_len,
                    "solve_rate": solve_rate
                })
                
        finally:
            # Đảm bảo luôn tắt Ray worker khi test xong hoặc có lỗi
            print("[evaluate] Đang tắt các Ray workers...")
            sp_manager.shutdown()

        # Lưu file JSON
        os.makedirs(os.path.dirname(self.result_json_path), exist_ok=True)
        with open(self.result_json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"[evaluate] Đã lưu kết quả tại {self.result_json_path}")

    def plot_scramble_results(self, json_path=None):
        path = json_path or self.result_json_path
        if not self.results and os.path.exists(path):
            with open(path, "r") as f:
                self.results = json.load(f)
        if not self.results:
            print("[evaluate] Không có dữ liệu để vẽ biểu đồ.")
            return
            
        scramble_lengths = [entry["scramble_len"] for entry in self.results]
        solve_rates = [entry["solve_rate"] * 100 for entry in self.results] 

        plt.figure(figsize=(10, 6))
        bars = plt.bar(scramble_lengths, solve_rates, color='steelblue', width=0.8, edgecolor='navy')
        for bar, rate in zip(bars, solve_rates):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{rate: .0f}%', ha='center', va='bottom', fontsize=9)
                     
        plt.xlabel("Scramble Length", fontsize=13)
        plt.ylabel("Solve Rate (%)", fontsize=13)
        plt.title("Solve rate (%) depend on scramble length", fontsize=16)
        plt.xticks(scramble_lengths)
        plt.ylim(0, 105)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        plot_path = os.path.join("training_logs_2x2", "eval_bar_chart.png")
        plt.savefig(plot_path, dpi=150)
        plt.show()
        print(f"[evaluate] Đã lưu biểu đồ tại {plot_path}")

if __name__ == "__main__":
    # Khởi tạo Evaluator với 7 workers (hoặc bạn có thể đổi thành số core bạn muốn)
    evaluator = ScrambleEvaluator(num_workers=12)
    # Bạn có thể chỉnh lại max_len tùy ý (mặc định đang để chạy tới 20)
    evaluator.evaluate_scramble_range(min_len=1, max_len=30, num_simulations=400, max_episode_steps=30)
    evaluator.plot_scramble_results()
