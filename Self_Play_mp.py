import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

import ray
import torch
import numpy as np
import time
from tqdm import tqdm
from Rubik2x2Env import Rubik2x2Env, NUM_ACTIONS, apply_move_idx
from MCTS_Core import MCTS
from Action_MCTS import pick_action_from_mcts
from Resnet import ResNetPolicyValueNet

@ray.remote(num_cpus=1)
class SelfPlayWorker:
    def __init__(self, device="cpu"):
        torch.set_num_threads(1) 
        torch.set_num_interop_threads(1)
        self.device = device
        
        self.model = ResNetPolicyValueNet(
            num_colors=6, hidden_dim=192, num_res_blocks=8, 
            num_actions=NUM_ACTIONS, dropout=0.05
        ).to(self.device)
        self.model.eval()
        
        self.env = None
        self.mcts = None

    def update_config(self, scramble_len, max_steps, num_simulations,
                      mcts_hyperparams=None, color_augmentation=True):
                      
        self.env = Rubik2x2Env(scramble_len=scramble_len, max_steps=max_steps, use_action_mask=True, color_augmentation=color_augmentation)
        self.max_steps = max_steps

        hp = mcts_hyperparams or {}
        c_puct          = float(hp.get("c_puct",          3.0))
        dirichlet_alpha = float(hp.get("dirichlet_alpha", 0.3))
        dirichlet_eps   = float(hp.get("dirichlet_eps",   0.25))

        if self.mcts is None:
            # Khởi tạo lần đầu
            self.mcts = MCTS(
                self.model, num_actions=NUM_ACTIONS,
                num_simulations=num_simulations, device=self.device,
                c_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_eps=dirichlet_eps,
            )
        else:
            # Cập nhật in-place — tránh tốn kém tái tạo object trong Ray
            self.mcts.num_simulations = num_simulations
            self.mcts.update_mcts_hyperparams(c_puct, dirichlet_alpha, dirichlet_eps)

    def update_weights(self, state_dict):
        if hasattr(self.model, "_orig_mod"):
            self.model._orig_mod.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def run_episodes(self, num_episodes, select_mode, gamma_discount=0.96, add_noise=True):
        worker_dataset = []
        solved_count   = 0
        solved_steps   = []  
        np.random.seed(int(time.time() * 1000) % 2**32)

        for _ in range(num_episodes):
            obs, info = self.env.reset()
            cube = self.env.cube
            action_mask = info.get("action_mask", None)
            episode_data = []
            solved = False
            
            visited_states = set()
            visited_states.add(cube.tobytes())

            for step in range(self.max_steps):
                visit_counts, _ = self.mcts.run(cube, obs, action_mask, add_noise=add_noise)
                vc_sum = visit_counts.sum()
                pi = visit_counts / vc_sum if vc_sum > 0 else np.ones_like(visit_counts) / len(visit_counts)

                episode_data.append((obs.copy(), pi.copy()))
                
                if select_mode == "greedy":
                    sorted_actions = np.argsort(visit_counts)[::-1]
                    action = sorted_actions[0] 
                    
                    for a in sorted_actions:
                        if visit_counts[a] == 0:
                            break
                        next_cube = apply_move_idx(cube, a)
                        if next_cube.tobytes() not in visited_states:
                            action = a
                            break
                else:
                    action = pick_action_from_mcts(visit_counts, mode=select_mode)

                obs, reward, terminated, truncated, info = self.env.step(action)
                cube = self.env.cube
                action_mask = info.get("action_mask", None)
                visited_states.add(cube.tobytes())

                if terminated:
                    solved = True
                    break

            for i, (o, p) in enumerate(episode_data):
                if solved:
                    steps_to_goal = len(episode_data) - i - 1
                    step_z = 1.0 * (gamma_discount ** steps_to_goal)
                else:
                    #progress = i / max(len(episode_data) - 1, 1)
                    #step_z = -0.5 - 0.5 * progress
                    step_z = -0.2 
                worker_dataset.append((o, p, step_z))

            if solved:
                solved_count += 1
                solved_steps.append(len(episode_data))   # ← ghi lại số bước

        # avg_steps = trung bình số bước khi solved, hoặc max_steps nếu không solved
        avg_steps = float(np.mean(solved_steps)) if solved_steps else float(self.max_steps)
        return worker_dataset, solved_count, avg_steps


class SelfPlayManager:
    def __init__(self, num_workers=11):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False, logging_level="error", include_dashboard=False)
        self.workers = [SelfPlayWorker.remote(device="cpu") for _ in range(num_workers)]
        self.num_workers = num_workers

    def update_configs(self, scramble_len, max_steps, num_simulations,
                       mcts_hyperparams=None, color_augmentation=True):
        futures = [
            w.update_config.remote(scramble_len, max_steps, num_simulations, mcts_hyperparams, color_augmentation)
            for w in self.workers
        ]
        ray.get(futures)

    def update_weights(self, model):
        state_dict = {k: v.cpu() for k, v in (model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()).items()}
        weights_ref = ray.put(state_dict) 
        futures = [w.update_weights.remote(weights_ref) for w in self.workers]
        ray.get(futures)

    def collect_data(self, num_episodes, select_mode, scramble_len_display,
                     gamma_discount=0.96, add_noise=True):
        """
        gamma_discount: truyền xuống worker để dùng trong reward shaping.
        add_noise: True khi training, False khi eval/inference.
        Mặc định 0.96 — backward-compatible với main.py cũ.

        Trả về thêm avg_steps: trung bình số bước giải của toàn bộ episodes solved.
        """
        eps_per_worker = num_episodes // self.num_workers
        remainder      = num_episodes % self.num_workers
        futures = []
        for i, worker in enumerate(self.workers):
            count = eps_per_worker + (1 if i < remainder else 0)
            if count > 0:
                futures.append(worker.run_episodes.remote(count, select_mode, gamma_discount, add_noise))

        results = []
        with tqdm(total=len(futures), desc=f"Scramble {scramble_len_display}", leave=False, colour="green") as pbar:
            unfinished = futures
            while unfinished:
                done, unfinished = ray.wait(unfinished, num_returns=1)
                results.append(ray.get(done[0]))
                pbar.update(1)

        total_data    = []
        total_solved  = 0
        steps_list    = []   # ← thu thập avg_steps từng worker

        for data, solved_n, avg_steps_w in results:
            total_data.extend(data)
            total_solved += solved_n
            if solved_n > 0:
                steps_list.append(avg_steps_w)

        solve_rate = total_solved / num_episodes if num_episodes > 0 else 0
        avg_steps  = float(np.mean(steps_list)) if steps_list else 0.0
        return total_data, solve_rate, avg_steps

    def shutdown(self):
        for w in self.workers: ray.kill(w)
        ray.shutdown()
