"""
Population-Based Training (PBT) for Rubik 2x2 AlphaZero Solver
================================================================
Reference: Jaderberg et al. (2017) - "Population Based Training of Neural Networks"
           https://arxiv.org/abs/1711.09846

Hyperparameters được tối ưu:
  - lr              : learning rate
  - c_puct          : exploration constant trong MCTS UCB
  - value_weight    : hệ số value loss
  - policy_weight   : hệ số policy loss
  - dirichlet_alpha : noise concentration tại root
  - dirichlet_eps   : noise mixing ratio tại root
  - step_penalty    : penalty mỗi bước thêm vào loss
  - num_simulations : số MCTS simulations — PBT tự tìm mức tối thiểu đủ dùng
                      Agent giải tốt với ít simulations → fitness cao hơn (nhanh hơn)
"""

import os
import copy
import json
import time
import random
import torch
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.align import Align

from Resnet import ResNetPolicyValueNet
from Self_Play_mp import SelfPlayManager
from Train_Network import dataset_to_tensors
from Rubik2x2Env import NUM_ACTIONS

# ─────────────────────────────────────────────
# Global Config
# ─────────────────────────────────────────────
DEVICE                = "cuda" if torch.cuda.is_available() else "cpu"
POPULATION_SIZE       = 6
NUM_WORKERS_PER_AGENT = 2
EPISODES_PER_ITER     = 100
BUFFER_MAXLEN         = 20000
MIN_BUFFER_SIZE       = 2000
TRAIN_SAMPLE_SIZE     = 512 * 12
BATCH_SIZE            = 256
TRAIN_EPOCHS          = 2 #4
EXPLOIT_INTERVAL      = 15
EXPLOIT_FRACTION      = 0.2
NUM_ITERS             = 300
FITNESS_WINDOW        = 8
MAX_SCRAMBLE          = 21
LOG_DIR               = "pbt_logs"
CHECKPOINT_DIR        = "pbt_checkpoints"

GAMMA_DISCOUNT   = 0.92
NUM_SIMULATIONS  = 400

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

console = Console()


# ─────────────────────────────────────────────
# Hyperparameter Space
# ─────────────────────────────────────────────
HP_BOUNDS = {
    "lr":              (1e-5, 1e-3),
    "c_puct":          (1.0, 4.0),
    "value_weight":    (1, 5.0),
    "policy_weight":   (0.8, 2.0),
    "dirichlet_alpha": (0.1, 0.5),
    "dirichlet_eps":   (0.1, 0.25),
    "step_penalty":    (0.005, 0.05),#(0.0, 0.02),
    "entropy_coeff":   (0.0, 0.003)#(0.001, 0.01),
}

# HP nào dùng log-uniform sampling
LOG_UNIFORM_HPS = {"lr"}

# HP nào là integer
INT_HPS = set()


def sample_hyperparams() -> Dict:
    hp = {}
    for name, (lo, hi) in HP_BOUNDS.items():
        if name in LOG_UNIFORM_HPS:
            hp[name] = float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))
        else:
            hp[name] = float(np.random.uniform(lo, hi))
    return hp


def perturb_hyperparams(hp: Dict, perturb_factor: float = 0.2) -> Dict:
    """
    EXPLORE: perturb ±20% hoặc resample 10%.

    """
    new_hp = {}
    for name, val in hp.items():
        lo, hi = HP_BOUNDS[name]
        if random.random() < 0.1:
            if name in LOG_UNIFORM_HPS:
                new_hp[name] = float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))
            else:
                new_hp[name] = float(np.random.uniform(lo, hi))
        else:
            factor      = 1.0 + np.random.choice([-1, 1]) * perturb_factor
            new_hp[name] = float(np.clip(val * factor, lo, hi))
    return new_hp




# ─────────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────────
@dataclass
class AgentState:
    agent_id:   int
    hyperparams: Dict
    model:      object   
    optimizer:  object 

    replay_buffer:      deque = field(default_factory=lambda: deque(maxlen=BUFFER_MAXLEN))
    scramble_len:       int   = 3
    recent_solve_rates: List[float] = field(default_factory=list)

    fitness:            float = 0.0
    fitness_history:    List[float] = field(default_factory=list)
    avg_steps_history:  List[float] = field(default_factory=list)
    loss_history:       List[float] = field(default_factory=list)
    pol_loss_history:   List[float] = field(default_factory=list)
    val_loss_history:   List[float] = field(default_factory=list)
    hp_history:         List[Dict]  = field(default_factory=list)
    exploit_count:      int   = 0
    iter_count:         int   = 0


def make_agent(agent_id: int, hp: Optional[Dict] = None) -> AgentState:
    if hp is None:
        hp = sample_hyperparams()

    model = ResNetPolicyValueNet(
        num_colors=6, hidden_dim=128, num_res_blocks=4,
        num_actions=NUM_ACTIONS, dropout=0.05
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hp["lr"], weight_decay=1e-4
    )

    return AgentState(agent_id=agent_id, hyperparams=hp, model=model, optimizer=optimizer)


# ─────────────────────────────────────────────
# Curriculum helpers
# ─────────────────────────────────────────────
def get_max_steps(scramble_len: int) -> int:
    return min(20, max(5, int(scramble_len * 2) + 1))


def get_threshold_config(scramble_len: int) -> Tuple[int, float, float]:
    if scramble_len <= 4:
        return 10, 0.97, 0.94
    elif scramble_len <= 7:
        return 12, 0.95, 0.90
    elif scramble_len <= 15:
        return 15, 0.90, 0.85 
    else:
        return 20, 0.92, 0.85 


def update_curriculum(agent: AgentState, solve_rate: float):
    agent.recent_solve_rates.append(solve_rate)
    window_size, avg_thr, min_thr = get_threshold_config(agent.scramble_len)

    if len(agent.recent_solve_rates) > window_size:
        agent.recent_solve_rates.pop(0)

    if len(agent.recent_solve_rates) == window_size:
        avg_s = float(np.mean(agent.recent_solve_rates))
        min_s = min(agent.recent_solve_rates)

        if avg_s >= avg_thr and min_s >= min_thr:
            # LEVEL UP
            agent.scramble_len += 1
            agent.recent_solve_rates.clear()
            console.print(
                f"[bold green]Agent {agent.agent_id}: "
                f"LEVEL UP → scramble {agent.scramble_len}[/]"
            )
        elif avg_s < 0.70 and agent.scramble_len > 3:
            agent.scramble_len -= 1
            agent.recent_solve_rates.clear()
            console.print(
                f"[bold red]Agent {agent.agent_id}: "
                f"LEVEL DOWN → scramble {agent.scramble_len} (avg={avg_s:.0%})[/]"
            )


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train_agent(agent: AgentState, dataset: list) -> Tuple[float, float, float]:
    if len(dataset) < 2:
        return 0.0, 0.0, 0.0

    hp        = agent.hyperparams
    model     = agent.model
    optimizer = agent.optimizer

    for pg in optimizer.param_groups:
        pg["lr"] = hp["lr"]

    model.to(DEVICE)
    model.train()

    obs, pi, v = dataset_to_tensors(dataset, DEVICE)
    N = obs.size(0)

    total_loss = total_pol = total_val = 0.0
    nb = 0

    import torch.nn.functional as F

    for _ in range(TRAIN_EPOCHS):
        perm = torch.randperm(N, device=DEVICE)
        obs_s, pi_s, v_s = obs[perm], pi[perm], v[perm]

        for start in range(0, N, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N)
            if end - start < 2:
                continue

            bo, bp, bv = obs_s[start:end], pi_s[start:end], v_s[start:end]
            optimizer.zero_grad()

            policy_logits, value_pred = model(bo)
            value_pred = value_pred.squeeze(-1)

            val_loss = F.smooth_l1_loss(value_pred, bv)
            log_prob = F.log_softmax(policy_logits, dim=-1)
            pol_loss = -(bp * log_prob).sum(dim=-1).mean()

            # entropy bonus: -(p log p) — thưởng policy đa dạng, phạt khi collapse
            # Quan trọng khi buffer đầy data cũ: tránh model "chắc chắn" sai
            prob     = log_prob.exp()
            entropy  = -(prob * log_prob).sum(dim=-1).mean()

            # step_penalty: phạt bước dài
            step_pen = hp.get("step_penalty", 0.0) * (1.0 - bv.clamp(0, 1)).mean()

            loss = (hp["value_weight"] * val_loss
                  + hp["policy_weight"] * pol_loss
                  - hp.get("entropy_coeff", 0.0) * entropy   # trừ vì maximize entropy
                  + step_pen)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_pol  += pol_loss.item()
            total_val  += val_loss.item()
            nb         += 1

    model.eval()
    if nb == 0:
        return 0.0, 0.0, 0.0
    return total_loss / nb, total_pol / nb, total_val / nb


# ─────────────────────────────────────────────
# Fitness
# ─────────────────────────────────────────────
def update_fitness(agent: AgentState, solve_rate: float, avg_steps: float):
    """
    Fitness = 3 thành phần:

        fitness = 0.50 * solve_component
                + 0.30 * level_component
                + 0.20 * step_efficiency

    solve_component  : trung bình solve_rate gần nhất → đo chất lượng giải
    level_component  : scramble_len / MAX_SCRAMBLE   → đo tiến độ curriculum
    step_efficiency  : expected_steps / avg_steps    → đo giải nhanh cỡ nào
    """
    # solve component
    agent.fitness_history.append(solve_rate)
    if len(agent.fitness_history) > FITNESS_WINDOW:
        agent.fitness_history.pop(0)
    solve_component = float(np.mean(agent.fitness_history))

    # level component
    level_component = agent.scramble_len / MAX_SCRAMBLE

    # step efficiency
    if avg_steps > 0:
        expected_steps  = agent.scramble_len * 1.5
        step_efficiency = min(1.0, expected_steps / avg_steps)
    else:
        step_efficiency = 0.0

    # Lưu history
    agent.avg_steps_history.append(avg_steps)
    if len(agent.avg_steps_history) > FITNESS_WINDOW:
        agent.avg_steps_history.pop(0)

    agent.fitness = (
        0.50 * solve_component +
        0.30 * level_component +
        0.20 * step_efficiency
    )


# ─────────────────────────────────────────────
# Exploit + Explore
# ─────────────────────────────────────────────
def exploit_and_explore(agents: List[AgentState]) -> List[str]:
    """
    Bottom EXPLOIT_FRACTION copy toàn bộ state từ top EXPLOIT_FRACTION,
    sau đó perturb hyperparams.
    """
    n         = len(agents)
    n_exploit = max(1, int(n * EXPLOIT_FRACTION))
    ranked    = sorted(agents, key=lambda a: a.fitness, reverse=True)
    top_agents    = ranked[:n_exploit]
    bottom_agents = ranked[n - n_exploit:]

    logs = []
    for loser in bottom_agents:
        winner = random.choice(top_agents)
        if winner.agent_id == loser.agent_id:
            continue

        old_hp       = copy.deepcopy(loser.hyperparams)
        old_scramble = loser.scramble_len

        # EXPLOIT: copy toàn bộ state của winner
        loser.model.load_state_dict(copy.deepcopy(winner.model.state_dict()))
        loser.hyperparams       = copy.deepcopy(winner.hyperparams)
        loser.scramble_len      = winner.scramble_len
        loser.fitness_history   = list(winner.fitness_history)
        loser.avg_steps_history = list(winner.avg_steps_history)
        loser.recent_solve_rates.clear()
        loser.replay_buffer.clear()

        # EXPLORE: perturb hyperparams
        loser.hyperparams = perturb_hyperparams(loser.hyperparams)

        # Warmup lr × 0.3 sau exploit — tránh loss spike (agent 1,4,5 bị spike sau copy weights)
        loser.optimizer = torch.optim.AdamW(
            loser.model.parameters(),
            lr=loser.hyperparams["lr"] * 0.3,
            weight_decay=1e-4
        )
        
        loser.optimizer.state.clear()
        loser.exploit_count += 1
        logs.append(
            f"Agent {loser.agent_id} [red]exploited[/] Agent {winner.agent_id} "
            f"(fit: {loser.fitness:.3f}←{winner.fitness:.3f}) | "
            f"scram: {old_scramble}→{loser.scramble_len} | "
            f"lr: {old_hp['lr']:.1e}→{loser.hyperparams['lr']:.1e}"
        )

    return logs


# ─────────────────────────────────────────────
# SelfPlay Pool
# ─────────────────────────────────────────────
class PBTSelfPlayPool:
    def __init__(self, total_workers: int):
        self.sp_manager  = SelfPlayManager(num_workers=total_workers)
        self.total_workers = total_workers

    def collect_for_agent(
        self,
        agent: AgentState,
        num_episodes: int,
    ) -> Tuple[list, float, float]:
        """
        Trả về (dataset, solve_rate, avg_steps).
        num_simulations cố định = NUM_SIMULATIONS.
        """
        hp           = agent.hyperparams
        scramble_len = agent.scramble_len
        max_steps    = get_max_steps(scramble_len)
        simulations  = NUM_SIMULATIONS   # cố định

        mcts_hp = {
            "c_puct":          hp["c_puct"],
            "dirichlet_alpha": hp["dirichlet_alpha"],
            "dirichlet_eps":   hp["dirichlet_eps"],
        }

        self.sp_manager.update_configs(
            scramble_len, max_steps, simulations,
            mcts_hyperparams=mcts_hp,
        )
        self.sp_manager.update_weights(agent.model)

        dataset, solve_rate, avg_steps = self.sp_manager.collect_data(
            num_episodes=num_episodes,
            select_mode="greedy",
            scramble_len_display=scramble_len,
            gamma_discount=GAMMA_DISCOUNT,
        )
        return dataset, solve_rate, avg_steps

    def shutdown(self):
        self.sp_manager.shutdown()


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
def save_pbt_log(agents: List[AgentState], iteration: int):
    snapshot = {
        "iteration": iteration,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "agents": []
    }
    for a in agents:
        snapshot["agents"].append({
            "agent_id":        a.agent_id,
            "fitness":         round(a.fitness, 4),
            "scramble_len":    a.scramble_len,
            "exploit_count":   a.exploit_count,
            "iter_count":      a.iter_count,
            "avg_steps_recent": round(float(np.mean(a.avg_steps_history[-FITNESS_WINDOW:])), 2)
                                 if a.avg_steps_history else 0.0,
            "hyperparams":     {k: round(v, 6) for k, v in a.hyperparams.items()},
            "fitness_history": [round(f, 4) for f in a.fitness_history[-20:]],
            "loss_history":    [round(l, 4) for l in a.loss_history[-20:]],
        })
    with open(os.path.join(LOG_DIR, "pbt_log.json"), "w") as f:
        json.dump(snapshot, f, indent=2)


def print_population_table(agents: List[AgentState], iteration: int):
    ranked = sorted(agents, key=lambda a: a.fitness, reverse=True)

    table = Table(
        title=f"[bold cyan]PBT Population — Iteration {iteration}[/bold cyan]",
        box=box.ROUNDED, show_header=True, header_style="bold magenta"
    )
    table.add_column("Rank",     justify="center", style="cyan")
    table.add_column("Agent",    justify="center")
    table.add_column("Fitness",  justify="right",  style="green")
    table.add_column("Scram",    justify="center", style="yellow")
    table.add_column("AvgSteps", justify="right",  style="cyan")
    table.add_column("lr",       justify="right",  style="blue")
    table.add_column("c_puct",   justify="right",  style="blue")
    table.add_column("v_w",      justify="right",  style="magenta")
    table.add_column("p_w",      justify="right",  style="magenta")
    table.add_column("dir_alpha",justify="right",  style="dim")
    table.add_column("s_pen",    justify="right",  style="dim")
    table.add_column("Exploit",  justify="right",  style="red")

    for rank, a in enumerate(ranked, 1):
        hp = a.hyperparams

        fit_str = f"{a.fitness*100:.1f}%"
        if a.fitness >= 0.9:
            fit_str = f"[bold green]{fit_str}[/]"
        elif a.fitness >= 0.6:
            fit_str = f"[yellow]{fit_str}[/]"
        else:
            fit_str = f"[red]{fit_str}[/]"

        avg_s_str = (
            f"{float(np.mean(a.avg_steps_history[-FITNESS_WINDOW:])):.1f}"
            if a.avg_steps_history else "—"
        )

        table.add_row(
            f"#{rank}",
            f"A{a.agent_id}",
            fit_str,
            str(a.scramble_len),
            avg_s_str,
            f"{hp['lr']:.1e}",
            f"{hp['c_puct']:.2f}",
            f"{hp['value_weight']:.2f}",
            f"{hp['policy_weight']:.2f}",
            f"{hp['dirichlet_alpha']:.2f}",
            f"{hp['step_penalty']:.3f}",
            str(a.exploit_count),
        )

    console.print(table)


def save_best_checkpoint(agents: List[AgentState], iteration: int) -> AgentState:
    best = max(agents, key=lambda a: a.fitness)
    torch.save(best.model.state_dict(),
               os.path.join(CHECKPOINT_DIR, f"best_agent_iter{iteration}.pt"))
    torch.save(best.model.state_dict(),
               os.path.join(CHECKPOINT_DIR, "best_latest.pt"))
    return best


# ─────────────────────────────────────────────
# Main PBT Loop
# ─────────────────────────────────────────────
def main():
    console.clear()
    info_text = (
        "[bold blue]Ton Duc Thang University[/bold blue]\n"
        "[white]Author    : [/white][bold green]Nguyen Quoc Khanh[/bold green]\n"
        "[white]Student ID: [/white][bold yellow]42200211[/bold yellow]\n"
        "[white]Major     : [/white][bold magenta]Electronics and Telecommunications[/bold magenta]"
    )
    console.print(Panel(Align.center(info_text), title="Student Info", border_style="blue", expand=False))
    console.print(Panel.fit("[bold cyan]RESNET-MCTS RUBIK SOLVER[/bold cyan]", border_style="cyan"))
    console.print("-" * 60)

    # ── Khởi tạo population ──
    console.print("[bold]Initializing population...[/]")
    agents = [make_agent(i) for i in range(POPULATION_SIZE)]

    # Bảng initial hyperparams
    init_table = Table(title="Initial Hyperparameters", box=box.SIMPLE)
    init_table.add_column("Agent")
    for k in HP_BOUNDS:
        init_table.add_column(k, justify="right")
    for a in agents:
        row = []
        for k in HP_BOUNDS:
            v = a.hyperparams[k]
            row.append(f"{int(round(v))}" if k in INT_HPS else f"{v:.3f}")
        init_table.add_row(f"A{a.agent_id}", *row)
    console.print(init_table)

    # ── Ray pool ──
    total_workers = POPULATION_SIZE * NUM_WORKERS_PER_AGENT
    pool = PBTSelfPlayPool(total_workers=total_workers)

    start_time = time.time()

    try:
        for iteration in range(1, NUM_ITERS + 1):
            console.rule(f"[bold cyan]Iteration {iteration}/{NUM_ITERS}[/]", style="cyan")
            iter_start = time.time()

            for agent in agents:
                agent.iter_count += 1

                # Step 1: Self-play
                dataset, solve_rate, avg_steps = pool.collect_for_agent(
                    agent, num_episodes=EPISODES_PER_ITER
                )

                # Step 2: Replay buffer
                agent.replay_buffer.extend(dataset)

                # Step 3: Train
                avg_loss = avg_pol_loss = avg_val_loss = 0.0
                if len(agent.replay_buffer) >= MIN_BUFFER_SIZE:
                    if len(agent.replay_buffer) > TRAIN_SAMPLE_SIZE:
                        # 50% data mới nhất + 50% random cũ
                        # tránh buffer stale (loss agent 2 tăng dần 0.439→0.459)
                        n_recent   = TRAIN_SAMPLE_SIZE // 2
                        buf_list   = list(agent.replay_buffer)
                        recent     = buf_list[-n_recent:]
                        old_sample = random.sample(buf_list[:-n_recent], TRAIN_SAMPLE_SIZE - n_recent)
                        train_data = recent + old_sample
                    else:
                        train_data = list(agent.replay_buffer)
                    avg_loss, avg_pol_loss, avg_val_loss = train_agent(agent, train_data)
                    agent.loss_history.append(avg_loss)
                    agent.pol_loss_history.append(avg_pol_loss)
                    agent.val_loss_history.append(avg_val_loss)

                    ckpt_path = os.path.join(CHECKPOINT_DIR, f"agent_{agent.agent_id}.pt")
                    torch.save(agent.model.state_dict(), ckpt_path)

                # Step 4: Fitness + Curriculum
                update_fitness(agent, solve_rate, avg_steps)
                update_curriculum(agent, solve_rate)
                agent.hp_history.append(copy.deepcopy(agent.hyperparams))

                # Log từng agent
                steps_str = f"{avg_steps:.1f}" if avg_steps > 0 else "—"
                solve_color = (
                    "bold green" if solve_rate >= 0.85
                    else "yellow" if solve_rate >= 0.5
                    else "red"
                )
                pol_str = f"{agent.pol_loss_history[-1]:.4f}" if agent.pol_loss_history else "—"
                val_str = f"{agent.val_loss_history[-1]:.4f}" if agent.val_loss_history else "—"
                console.print(
                    f"  [cyan]A{agent.agent_id}[/] | "
                    f"scram=[magenta]{agent.scramble_len}[/] | "
                    f"solve=[{solve_color}]{solve_rate*100:.1f}%[/] | "
                    f"fit=[bold]{agent.fitness*100:.1f}%[/] | "
                    f"steps=[cyan]{steps_str}[/] | "
                    f"loss=[yellow]{avg_loss:.4f}[/] pol=[blue]{pol_str}[/] val=[red]{val_str}[/] | "
                    f"lr={agent.hyperparams['lr']:.1e}"
                )

            # ── Exploit + Explore ──
            if iteration % EXPLOIT_INTERVAL == 0:
                console.rule("[bold red]EXPLOIT + EXPLORE[/bold red]", style="red")
                exploit_logs = exploit_and_explore(agents)
                for log in exploit_logs:
                    console.print(f"  {log}")
                if not exploit_logs:
                    console.print("  [dim]No exploit needed[/]")

            # ── Population table ──
            print_population_table(agents, iteration)

            # ── Best checkpoint ──
            best = save_best_checkpoint(agents, iteration)
            console.print(
                f"[bold green]Best: Agent {best.agent_id} "
                f"(fit={best.fitness*100:.1f}%, scram={best.scramble_len})[/]"
            )

            # ── Log ──
            save_pbt_log(agents, iteration)

            elapsed       = time.time() - iter_start
            total_elapsed = time.time() - start_time
            console.print(
                f"[dim]Iter time: {elapsed:.1f}s | Total: {total_elapsed/60:.1f}min[/]"
            )

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Training interrupted by user.[/]")

    finally:
        console.print("[yellow]Shutting down Ray workers...[/]")
        pool.shutdown()

    # ── Summary ──
    console.rule("[bold green]PBT TRAINING FINISHED[/]")
    best = max(agents, key=lambda a: a.fitness)
    console.print(Panel(
        f"[bold]Best Agent       :[/] A{best.agent_id}\n"
        f"[bold]Fitness          :[/] {best.fitness*100:.1f}%\n"
        f"[bold]Scramble Len     :[/] {best.scramble_len}\n"
        f"[bold]Num Simulations  :[/] {NUM_SIMULATIONS}\n"
        f"[bold]Exploit Count    :[/] {best.exploit_count}\n\n"
        f"[bold]Best Hyperparameters:[/]\n"
        + "\n".join(f"  {k}: {v:.5f}" for k, v in best.hyperparams.items()),
        title="[bold cyan]Best Agent Summary[/]",
        border_style="green"
    ))

    total_time = time.time() - start_time
    console.print(f"Total training time: [bold yellow]{total_time/60:.1f} minutes[/]")

    best_hp_path = os.path.join(LOG_DIR, "best_hyperparams.json")
    with open(best_hp_path, "w") as f:
        json.dump({
            "agent_id":        best.agent_id,
            "fitness":         round(best.fitness, 4),
            "scramble_len":    best.scramble_len,
            "num_simulations": NUM_SIMULATIONS,
            "gamma_discount":  GAMMA_DISCOUNT,
            "hyperparams":     {k: round(v, 6) for k, v in best.hyperparams.items()},
        }, f, indent=2)
    console.print(f"[green]Best hyperparams saved")


if __name__ == "__main__":
    main()
