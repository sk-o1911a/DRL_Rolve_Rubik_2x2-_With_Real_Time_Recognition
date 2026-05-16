import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
from typing import List


class MetricsLogger:
    def __init__(self, log_dir: str = "training_logs_2x2"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.iterations: List[int] = []
        self.losses: List[float] = []
        self. policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.solve_rates: List[float] = []
        self.scramble_lengths: List[int] = []
        self.num_samples: List[int] = []

    def log_iteration(
            self,
            iteration: int,
            loss:  float,
            policy_loss:  float,
            value_loss:  float,
            solve_rate:  float,
            scramble_len: int,
            num_samples: int
    ):
        self.iterations.append(iteration)
        self.losses.append(loss)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.solve_rates.append(solve_rate)
        self.scramble_lengths.append(scramble_len)
        self.num_samples.append(num_samples)

    def save_json(self, filename: str = "metrics.json"):
        data = {
            "iterations": self.iterations,
            "losses": self. losses,
            "policy_losses": self.policy_losses,
            "value_losses": self. value_losses,
            "solve_rates": self.solve_rates,
            "scramble_lengths": self.scramble_lengths,
            "num_samples": self.num_samples
        }
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[logger] Saved metrics to {filepath}")

    def load_json(self, filename: str = "metrics.json"):
        filepath = os.path. join(self.log_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.iterations = data. get("iterations", [])
            self.losses = data.get("losses", [])
            self.policy_losses = data.get("policy_losses", [])
            self.value_losses = data. get("value_losses", [])
            self.solve_rates = data.get("solve_rates", [])
            self.scramble_lengths = data.get("scramble_lengths", [])
            self.num_samples = data. get("num_samples", [])
            print(f"[logger] Loaded {len(self.iterations)} iterations from {filepath}")
            return True
        print(f"[logger] No existing metrics file found at {filepath}")
        return False

    def plot_all(self, filename: str = "training_metrics.png", show: bool = False, start_time: str = None, end_time: str = None, total_time: str = None):
        if len(self.iterations) == 0:
            print("[logger] No data to plot")
            return

        fig, axes = plt. subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Metrics - Rubik\'s Cube 2x2', fontsize=16, fontweight='bold')

        # Total Loss plot
        ax1 = axes[0, 0]
        ax1.plot(self. iterations, self.losses, linewidth=1.5, color='blue', alpha=0.8)
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('Total Loss', fontsize=11)
        ax1.set_title('Total Loss', fontweight='bold', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Solve rate plot
        ax2 = axes[0, 1]
        ax2.plot(self.iterations, [sr * 100 for sr in self.solve_rates],
                 linewidth=1.5, color='green', alpha=0.8)
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('Solve Rate (%)', fontsize=11)
        ax2.set_title('Solve Rate', fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 105])

        # Policy Loss plot
        ax3 = axes[0, 2]
        ax3.plot(self.iterations, self.policy_losses,
                 linewidth=1.5, color='orange', alpha=0.8)
        ax3.set_xlabel('Iteration', fontsize=11)
        ax3.set_ylabel('Policy Loss', fontsize=11)
        ax3.set_title('Policy Loss', fontweight='bold', fontsize=12)
        ax3.grid(True, alpha=0.3)

        # Value Loss plot
        ax4 = axes[1, 0]
        ax4.plot(self.iterations, self.value_losses,
                 linewidth=1.5, color='red', alpha=0.8)
        ax4.set_xlabel('Iteration', fontsize=11)
        ax4.set_ylabel('Value Loss', fontsize=11)
        ax4.set_title('Value Loss', fontweight='bold', fontsize=12)
        ax4.grid(True, alpha=0.3)

        # Scramble Length plot (Curriculum Progress)
        ax5 = axes[1, 1]
        ax5.plot(self.iterations, self.scramble_lengths,
                 linewidth=2, color='purple', alpha=0.8)
        ax5.fill_between(self.iterations, self.scramble_lengths, alpha=0.3, color='purple')
        ax5.set_xlabel('Iteration', fontsize=11)
        ax5.set_ylabel('Scramble Length', fontsize=11)
        ax5.set_title('Curriculum Progress', fontweight='bold', fontsize=12)
        ax5.grid(True, alpha=0.3)

        # Samples per iteration plot
        ax6 = axes[1, 2]
        ax6.plot(self.iterations, self.num_samples,
                 linewidth=1.5, color='teal', alpha=0.8)
        ax6.set_xlabel('Iteration', fontsize=11)
        ax6.set_ylabel('Samples', fontsize=11)
        ax6.set_title('Samples per Iteration', fontweight='bold', fontsize=12)
        ax6.grid(True, alpha=0.3)
        
        # Time
        if start_time or end_time or total_time:
            info_lines = []
            if start_time: info_lines.append(f"Start: {start_time}")
            if end_time:   info_lines.append(f"End:   {end_time}")
            if total_time: info_lines.append(f"Total: {total_time}")
            
            info_text = "\n".join(info_lines)

            plt.figtext(0.015, 0.015, info_text, fontsize=10, ha="left", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))

        plt.tight_layout()
        filepath = os.path.join(self.log_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')

        if show:
            plt. show()
        plt.close()
        print(f"[logger] Saved plot to {filepath}")


if __name__ == "__main__":
    logger = MetricsLogger(log_dir="training_logs_2x2")
    if logger.load_json():
        logger.plot_all(show=True)
    else:
        print("No metrics to plot")
