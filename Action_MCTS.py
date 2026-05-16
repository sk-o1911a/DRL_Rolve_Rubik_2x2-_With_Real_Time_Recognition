import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SELECT_MODE = "greedy"

def pick_action_from_mcts(visit_counts: np.ndarray,mode: str = "greedy", temperature: float = 1.0) -> int:

    if mode == "greedy":
        return int(np.argmax(visit_counts))

    vc = visit_counts.astype(np.float32)
    
    if vc.sum() == 0:
        vc = np.ones_like(vc, dtype=np.float32)
    probs = vc / vc.sum()
    action = np.random.choice(len(probs), p=probs)
    return int(action)

