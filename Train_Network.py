import numpy as np
import torch
import torch.nn. functional as F
import torch.optim as optim

def cal_loss(policy_logits, value_pred, target_pi, target_v):
    # value loss
    value_pred = value_pred.squeeze(-1)
    value_loss = F.smooth_l1_loss(value_pred, target_v)

    # policy loss
    log_prob = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -(target_pi * log_prob).sum(dim=-1).mean()

    # entropy loss
    entropy = -(F.softmax(policy_logits, dim=-1) * log_prob).sum(dim=-1).mean()

    loss = 1.5 * value_loss + 1 * policy_loss #- 0.005 * entropy
    return loss, policy_loss, value_loss

def dataset_to_tensors(dataset, device):
    obs_list, pi_list, v_list = [], [], []
    for s, pi, z in dataset:
        obs_list.append(s)
        pi_list.append(pi)
        v_list.append(z)

    obs = torch.from_numpy(np.stack(obs_list)).float().to(device)
    pi  = torch.from_numpy(np.stack(pi_list)).float().to(device)
    v   = torch.from_numpy(np.array(v_list, dtype=np.float32)).to(device)
    return obs, pi, v

def train_on_selfplay_data(
    model,
    optimizer,
    dataset,
    batch_size: int = 128,
    epochs: int = 10,
    lr: float = 1e-4,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.train()

    obs, pi, v = dataset_to_tensors(dataset, device)
    N = obs.size(0)
    
    final_loss = 0.0
    final_pol_loss = 0.0
    final_val_loss = 0.0

    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        obs_shuf = obs[perm]
        pi_shuf  = pi[perm]
        v_shuf   = v[perm]

        total_loss = 0.0
        total_pol  = 0.0
        total_val  = 0.0
        nb = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size,N)
            
            if end - start < 2:
                continue
                
            batch_obs = obs_shuf[start:end]
            batch_pi  = pi_shuf[start:end]
            batch_v   = v_shuf[start:end]

            optimizer.zero_grad()

            policy_logits, value_pred = model(batch_obs)
            loss, pol_loss, val_loss = cal_loss(policy_logits, value_pred, batch_pi, batch_v)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_pol  += pol_loss.item()
            total_val  += val_loss.item()
            nb += 1

        final_loss = total_loss / nb
        final_pol_loss = total_pol / nb
        final_val_loss = total_val / nb


    return model, final_loss, final_pol_loss, final_val_loss
