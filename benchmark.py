import time
import os
import sys
import psutil
import multiprocessing
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich import box
from rich.panel import Panel

from Rubik2x2Env import Rubik2x2Env, scramble, solved_cube, encode_onehot, apply_random_map
from MCTS_Core import MCTS
from Action_MCTS import pick_action_from_mcts
from BFS_Solver import BFSSolver
from Dijkstra_Solver import DijkstraSolver
from AStar_Solver import AStarSolver
import torch.nn.functional as F

try:
    with open("pbt_logs/best_hyperparams.json", "r") as f:
        BEST_C_PUCT = json.load(f)["hyperparams"]["c_puct"]
except:
    pass
    

@contextmanager
def suppress_output():
    try:
        with open(os.devnull, "w") as devnull:
            old_stdout_fd = os.dup(sys.stdout.fileno())
            old_stderr_fd = os.dup(sys.stderr.fileno())

            try:
                sys.stdout.flush()
                sys.stderr.flush()
                
                os.dup2(devnull.fileno(), sys.stdout.fileno())
                os.dup2(devnull.fileno(), sys.stderr.fileno())
                
                yield
            finally:
                os.dup2(old_stdout_fd, sys.stdout.fileno())
                os.dup2(old_stderr_fd, sys.stderr.fileno())
                os.close(old_stdout_fd)
                os.close(old_stderr_fd)
    except Exception:
        yield

# ================= CẤU HÌNH =================
ONNX_PATH = "rubik2x2.onnx"
TIMEOUT = 10
MAX_MEM_GB = 1
SCRAMBLE_RANGE = range(2, 30)
NUM_SAMPLES = 10
MAX_DEPTH = 30
MCTS_SIMS = 400

# ================= WRAPPER & LOGIC =================

class ONNXModelWrapper:
    def __init__(self, onnx_path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        
        with suppress_output():
            self.session = ort.InferenceSession(onnx_path, opts, providers=['CPUExecutionProvider'])
            
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, x):
        if isinstance(x, torch.Tensor): x = x.cpu().numpy()
        if x.ndim > 2: x = x.reshape(x.shape[0], -1)
        elif x.ndim == 2: x = x.reshape(1, -1)
        
        with suppress_output():
            outputs = self.session.run(None, {self.input_name: x})
            
        policy_t = torch.tensor(outputs[0])
        value_t = torch.tensor(outputs[1])
        probs = F.softmax(policy_t, dim=-1)
        return probs.squeeze(0), value_t.squeeze()

def run_algo_logic(algo_name, cube, return_dict):
    with suppress_output():
        try:
            torch.set_num_threads(1)
            os.environ["OMP_NUM_THREADS"] = "1"
            
            start_t = time.time()
            solved = False
            steps = 0
            nodes = 0
            timed_out = False
            total_sims = 0   # chỉ có nghĩa với MCTS

            if algo_name == "BFS":
                solver = BFSSolver(max_depth=MAX_DEPTH)
                actions, _, solved, nodes, _ = solver.solve_with_stats(cube)
                steps = len(actions) if solved else 0
                timed_out = not solved
                
            elif algo_name == "Dijkstra":
                solver = DijkstraSolver(max_depth=MAX_DEPTH)
                actions, _, solved, nodes, _ = solver.solve_with_stats(cube)
                steps = len(actions) if solved else 0
                timed_out = not solved
                
            elif algo_name == "A*":
                solver = AStarSolver(max_depth=MAX_DEPTH)
                actions, _, solved, nodes, _ = solver.solve_with_stats(cube)
                steps = len(actions) if solved else 0
                timed_out = not solved
                
            elif algo_name == "MCTS":
                env = Rubik2x2Env(use_action_mask=True)
                env.cube = cube.copy()
                env.max_steps = 40
                
                if os.path.exists(ONNX_PATH):
                    model = ONNXModelWrapper(ONNX_PATH)
                    mcts = MCTS(model=model, num_simulations=MCTS_SIMS, c_puct=BEST_C_PUCT, device="cpu")
                    obs = encode_onehot(env.cube)
                    info = {"action_mask": env._legal_action_mask()}
                    
                    steps_taken = 0
                    
                    for _ in range(env.max_steps):
                        root_cube = env.cube.copy()
                        visit_counts, nodes_exp = mcts.run(root_cube, obs, info["action_mask"], add_noise=False)
                        total_sims += MCTS_SIMS
                        nodes += nodes_exp
                        
                        action = pick_action_from_mcts(visit_counts, mode="greedy")
                        obs, _, terminated, truncated, info = env.step(action)
                        steps_taken += 1
                        if terminated:
                            solved = True
                            break
                        if truncated:
                            timed_out = True
                            break

                    if not solved and steps_taken >= env.max_steps:
                        timed_out = True

                    steps = steps_taken
                    nodes = total_sims
                else:
                    return_dict["status"] = "Error"
                    return_dict["error"] = "No ONNX"
                    return

            end_t = time.time()
            return_dict["solved"] = solved
            return_dict["time"] = end_t - start_t
            return_dict["steps"] = steps
            return_dict["nodes"] = nodes
            return_dict["total_sims"] = total_sims
            return_dict["timed_out"] = timed_out
            return_dict["status"] = "OK"

        except Exception as e:
            return_dict["status"] = "Error"
            return_dict["error"] = str(e)

def monitor_and_run(algo_name, cube):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    p = multiprocessing.Process(target=run_algo_logic, args=(algo_name, cube, return_dict))
    p.start()
    
    start_time = time.time()
    max_ram_usage = 0.0
    
    while p.is_alive():
        if time.time() - start_time > TIMEOUT:
            p.terminate()
            p.join()
            return {"status": "Timeout", "ram": max_ram_usage, "solved": False,
                    "time": TIMEOUT, "timed_out": True, "total_sims": 0}
        
        try:
            proc = psutil.Process(p.pid)
            mem_gb = proc.memory_info().rss / (1024**3)
            max_ram_usage = max(max_ram_usage, mem_gb)
            
            if mem_gb > MAX_MEM_GB:
                p.terminate()
                p.join()
                return {"status": "OOM", "ram": mem_gb, "solved": False,
                        "time": TIMEOUT, "timed_out": True, "total_sims": 0}
        except:
            break
        time.sleep(0.05)
        
    p.join()
    
    if return_dict.get("status") == "OK":
        res = dict(return_dict)
        res["ram"] = max_ram_usage
        # Nếu process bị Timeout/OOM từ monitor thì timed_out đã được set bên ngoài
        return res
    return {"status": "Fail", "ram": max_ram_usage, "solved": False, "time": 0,
            "timed_out": True, "total_sims": 0}

# ================= MAIN =================

def main():
    multiprocessing.set_start_method('spawn', force=True)
    console = Console()
    console.clear()
    
    console.print(Panel.fit(
        f"[bold yellow]BENCHMARK RUBIK 2x2[/]\n",
        border_style="green"
    ))

    algorithms = ["BFS", "Dijkstra", "A*", "MCTS"]
    
    # Pre-generate
    with console.status("[bold green]Generating Cubes..."):
        test_cubes = {}
        for k in SCRAMBLE_RANGE:
            cubes_for_k = []
            for _ in range(NUM_SAMPLES):
                c = scramble(solved_cube(), k)

                perm = np.random.permutation(6).astype(np.int8)
                c = apply_random_map(c, perm)
                
                cubes_for_k.append(c)
            test_cubes[k] = cubes_for_k
    
    # Results storage
    final_results = {algo: {"rate": [], "time": [], "ram": [], "nodes": [], "step": [],
                            "sims": [], "timeouts": []} for algo in algorithms}

    # Đếm tổng solved cho biểu đồ cột ô thứ 4
    total_solved_count = {algo: 0 for algo in algorithms}
    total_sample_count = {algo: 0 for algo in algorithms}
    total_timeout_count = {algo: 0 for algo in algorithms}  # tổng timeout/OOM toàn bộ scramble

    # Progress Bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        main_task = progress.add_task("[cyan]Total Progress", total=len(SCRAMBLE_RANGE))
        
        for k in SCRAMBLE_RANGE:
            table = Table(title=f"Len {k}", box=box.SIMPLE_HEAD, show_edge=False)
            table.add_column("Algo", style="cyan")
            table.add_column("Solved")
            table.add_column("Time")
            table.add_column("RAM")
            table.add_column("Avg Nodes", justify="right")
            table.add_column("Status")
            
            for algo in algorithms:
                task = progress.add_task(f"Running {algo}", total=NUM_SAMPLES)
                
                stats = {"solved": 0, "time": [], "ram": [], "status": [],
                         "total_sims": 0, "timeouts": 0, "nodes": []}
                
                for cube in test_cubes[k]:
                    res = monitor_and_run(algo, cube)
                    
                    if res.get("solved"): stats["solved"] += 1
                    stats["time"].append(res.get("time", 0))
                    stats["ram"].append(res.get("ram", 0) * 1024)
                    stats["status"].append(res.get("status"))
                    stats["total_sims"] += res.get("total_sims", 0)

                    is_timeout = res.get("timed_out") or res.get("status") in ("Timeout", "OOM")
                    nodes_val = res.get("nodes", 0) or res.get("total_sims", 0)

                    if is_timeout and nodes_val == 0:
                        if algo == "MCTS":
                            nodes_val = MCTS_SIMS * 40
                        else:
                            nodes_val = int(MAX_DEPTH * (9 ** (MAX_DEPTH // 2)))

                    stats["nodes"].append(nodes_val)
                    if is_timeout:
                        stats["timeouts"] += 1
                    
                    progress.advance(task)
                
                progress.remove_task(task)
                
                # Aggregation
                rate = (stats["solved"]/NUM_SAMPLES)*100
                avg_time = np.mean(stats["time"])
                avg_ram = np.mean(stats["ram"])
                
                avg_nodes = np.mean(stats["nodes"]) if stats["nodes"] else 0.0

                final_results[algo]["rate"].append(rate)
                final_results[algo]["time"].append(avg_time)
                final_results[algo]["ram"].append(avg_ram)
                final_results[algo]["sims"].append(stats["total_sims"])
                final_results[algo]["timeouts"].append(stats["timeouts"])
                final_results[algo]["nodes"].append(avg_nodes)

                # Tích lũy tổng solved / tổng sample / tổng timeout
                total_solved_count[algo] += stats["solved"]
                total_sample_count[algo] += NUM_SAMPLES
                total_timeout_count[algo] += stats["timeouts"]
                
                # Status string formatting
                oom_count = stats["status"].count("OOM")
                to_count = stats["status"].count("Timeout")
                err_count = stats["status"].count("Error")

                status_parts = []
                if oom_count:
                    status_parts.append(f"[red]OOM ({oom_count})[/]")
                if to_count:
                    status_parts.append(f"[yellow]Timeout ({to_count})[/]")
                if err_count:
                    status_parts.append(f"[magenta]Error ({err_count})[/]")

                status_str = "[green]OK[/]" if not status_parts else " ".join(status_parts)

                nodes_str = f"{avg_nodes:,.0f}" if avg_nodes > 0 else "—"

                table.add_row(
                    algo, 
                    f"{rate:.0f}%", 
                    f"{avg_time:.2f}s", 
                    f"{avg_ram:.0f}MB",
                    nodes_str,
                    status_str
                )
            
            console.print(table)
            progress.advance(main_task)

    # Plotting
    console.print("\n[bold magenta]Saving plots...[/]")
    fig, axs = plt.subplots(2, 3, figsize=(21, 12))
    fig.suptitle("COMPARISON BETWEEN BFS, DIJKSTRA, A*, AND MCTS", fontsize=16, fontweight="bold")
    x = list(SCRAMBLE_RANGE)
    colors_map = {"BFS": "#3498db", "Dijkstra": "#e67e22", "A*": "#2ecc71", "MCTS": "#e74c3c"}
    markers_map = {"BFS": "o", "Dijkstra": "s", "A*": "^", "MCTS": "D"}

    # 1. Solved Rate
    for algo in algorithms:
        axs[0,0].plot(x, final_results[algo]["rate"], label=algo, marker=markers_map[algo],
                      color=colors_map[algo], linewidth=2)
    axs[0,0].set_title("Solved Rate (%)", fontweight="bold")
    axs[0,0].set_xlabel("Scramble Length")
    axs[0,0].set_ylabel("Solved Rate (%)")
    axs[0,0].legend(); axs[0,0].grid(True, alpha=0.3)
    axs[0,0].set_xticks(x)

    # 2. Avg Time
    for algo in algorithms:
        axs[0,1].plot(x, final_results[algo]["time"], label=algo, marker=markers_map[algo],
                      color=colors_map[algo], linewidth=2)
    axs[0,1].set_title("Avg Time (s)", fontweight="bold")
    axs[0,1].set_xlabel("Scramble Length")
    axs[0,1].set_ylabel("Time (s)")
    axs[0,1].legend(); axs[0,1].grid(True, alpha=0.3)
    axs[0,1].set_xticks(x)

    # 3. RAM
    for algo in algorithms:
        axs[0,2].plot(x, final_results[algo]["ram"], label=algo, marker=markers_map[algo],
                      color=colors_map[algo], linewidth=2)
    axs[0,2].set_title("Avg RAM (MB, max 2048MB)", fontweight="bold")
    axs[0,2].set_xlabel("Scramble Length")
    axs[0,2].set_ylabel("RAM (MB)")
    axs[0,2].legend(); axs[0,2].grid(True, alpha=0.3)
    axs[0,2].set_xticks(x)

    # 4. Biểu đồ cột: Avg Solve Rate tổng hợp
    avg_solve_rates = []
    for algo in algorithms:
        if total_sample_count[algo] > 0:
            avg_solve_rates.append((total_solved_count[algo] / total_sample_count[algo]) * 100)
        else:
            avg_solve_rates.append(0.0)

    bar_colors = [colors_map[a] for a in algorithms]
    bars = axs[1,0].bar(algorithms, avg_solve_rates, color=bar_colors, width=0.5,
                        edgecolor="black", linewidth=0.8)
    for bar, rate in zip(bars, avg_solve_rates):
        height = bar.get_height()
        axs[1,0].text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{rate:.1f}%",
                      ha="center", va="bottom", fontsize=11, fontweight="bold")
    axs[1,0].set_title("Avg Solve Rate (%) — All Scrambles", fontweight="bold")
    axs[1,0].set_ylabel("Solve Rate (%)")
    axs[1,0].set_ylim(0, 115)
    axs[1,0].grid(axis="y", linestyle="--", alpha=0.4)

    # 5. Biểu đồ đường: Avg Nodes Explored per Scramble Length (cả 4 thuật toán)
    for algo in algorithms:
        nodes_data = final_results[algo]["nodes"]
        if any(n > 0 for n in nodes_data):
            axs[1,1].plot(x, nodes_data, label=algo, marker=markers_map[algo],
                          color=colors_map[algo], linewidth=2, linestyle="--")
    axs[1,1].set_title("Avg Nodes Explored per Scramble Length", fontweight="bold")
    axs[1,1].set_xlabel("Scramble Length")
    axs[1,1].set_ylabel("Avg Nodes Explored")
    axs[1,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    axs[1,1].legend(); axs[1,1].grid(True, alpha=0.3)
    axs[1,1].set_xticks(x)

    # 6. Biểu đồ cột: Tổng Timeout/OOM theo thuật toán
    total_timeouts = [total_timeout_count[algo] for algo in algorithms]
    timeout_colors = [
        "#e74c3c" if t > NUM_SAMPLES * len(list(SCRAMBLE_RANGE)) * 0.3
        else "#f39c12" if t > 0
        else "#2ecc71"
        for t in total_timeouts
    ]
    bars_to = axs[1,2].bar(algorithms, total_timeouts, color=timeout_colors, width=0.5,
                           edgecolor="black", linewidth=0.8)
    for bar, t in zip(bars_to, total_timeouts):
        axs[1,2].text(bar.get_x() + bar.get_width() / 2.0,
                      bar.get_height() + 0.3, str(t),
                      ha="center", va="bottom", fontsize=12, fontweight="bold")
    axs[1,2].set_title("Total Timeout / OOM Count — All Scrambles", fontweight="bold")
    axs[1,2].set_ylabel("Count")
    axs[1,2].set_ylim(0, max(total_timeouts + [1]) * 1.3 + 2)
    axs[1,2].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[1,2].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=150, bbox_inches="tight")
    console.print("[green]Done! Saved to benchmark_results.png[/]")

if __name__ == "__main__":
    main()
