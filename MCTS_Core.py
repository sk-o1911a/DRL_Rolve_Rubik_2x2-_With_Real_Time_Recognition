import numpy as np
import torch
from Rubik2x2Env import (
    Rubik2x2Env, apply_move_idx, encode_onehot,
    is_solved, NUM_ACTIONS, fast_legal_action_mask
)


class MCTSNode:
    __slots__ = [
        'cube', 'obs', 'action_mask', 'parent', 'prior', 'priors',
        'children', 'num_children', 'expanded',
        'N', 'W', 'Q', 'last_face', 'repeat_count', 'total_children_N'
    ]

    def __init__(self, cube, obs, action_mask=None, parent=None, prior=1.0,
                 last_face=None, repeat_count=0, num_actions=NUM_ACTIONS):
        self.cube = cube
        self.obs = obs
        self.action_mask = action_mask
        self.parent = parent
        self.prior = float(prior)
        self.priors = None

        self.children = [None] * num_actions
        self.num_children = 0
        self.expanded = False

        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.last_face = last_face
        self.repeat_count = repeat_count
        self.total_children_N = 0

    def is_leaf(self):
        return not self.expanded


class MCTS:
    def __init__(self, model, num_actions=NUM_ACTIONS, c_puct=1.5,
                 num_simulations=400, device="cpu",
                 dirichlet_alpha=0.3, dirichlet_eps=0.25):
        self.model = model
        self.num_actions = num_actions
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = device
        # Dirichlet noise hyperparams — được PBT tối ưu
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps   = dirichlet_eps
        self._visit_counts_buffer = np.zeros(num_actions, dtype=np.float32)
        if hasattr(model, "parameters"):
            self._model_dtype = next(model.parameters()).dtype
        else:
            self._model_dtype = torch.float32

    def update_mcts_hyperparams(self, c_puct: float, dirichlet_alpha: float, dirichlet_eps: float):
        """Cập nhật hyperparams MCTS khi PBT exploit/explore mà không tái tạo object."""
        self.c_puct          = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps   = dirichlet_eps

    def run(self, root_cube, root_obs, root_action_mask=None, add_noise=False):
        root = MCTSNode(
            cube=root_cube, obs=root_obs, action_mask=root_action_mask,
            parent=None, prior=1.0, last_face=None, repeat_count=0,
            num_actions=self.num_actions,
        )

        # Evaluate root lần đầu để có priors
        self._evaluate(root)

        # Nếu root đã solved thì không cần search
        if is_solved(root.cube):
            vc = np.zeros(self.num_actions, dtype=np.float32)
            return vc,0

        # Thêm Dirichlet noise vào priors của root để khuyến khích exploration
        # Dùng self.dirichlet_alpha / self.dirichlet_eps thay vì hardcode
        if add_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * self.num_actions)
            eps   = self.dirichlet_eps
            for a in range(self.num_actions):
                if root.action_mask is None or root.action_mask[a]:
                    root.priors[a] = (
                        root.priors[a] * (1.0 - eps) + noise[a] * eps
                    )

        self._nodes_expanded = 0   # reset mỗi lần run

        for _ in range(self.num_simulations):
            leaf, path = self._select(root)
            value = self._evaluate(leaf)
            self._backup(path, value)

            if not path:
                root.N += 1
                root.W += value
                root.Q = root.W / root.N

        vc = self._visit_counts_buffer
        vc.fill(0)
        for action in range(self.num_actions):
            child = root.children[action]
            if child is not None:
                vc[action] = child.N
        return vc.copy(), self._nodes_expanded

    def _select(self, node):
        path = []
        current = node
        while current.expanded and not is_solved(current.cube):
            action = self._select_child(current)
            child = self._get_or_create_child(current, action)
            path.append((current, action))
            current = child
        return current, path

    def _select_child(self, node: MCTSNode):
        # FPU fix: parent_N tối thiểu 1 để U > 0 ngay từ lần duyệt đầu
        # → prior của policy head quyết định node đầu tiên được chọn
        parent_N   = max(1, node.total_children_N)
        sqrt_total = np.sqrt(parent_N)
        c_puct     = self.c_puct
        best_score = -1e9
        best_action = -1

        priors     = node.priors
        children   = node.children
        action_mask = node.action_mask

        for action in range(self.num_actions):
            if action_mask is not None and not action_mask[action]:
                continue

            child = children[action]
            if child is not None:
                Q      = child.Q
                N_child = child.N
                prior  = child.prior
            else:
                # FPU = Parent's Q — tránh BFS trap với value targets âm
                # U > 0 (vì parent_N >= 1) nên prior vẫn phân biệt được các actions
                Q      = node.Q
                N_child = 0
                prior  = priors[action]

            U     = c_puct * prior * sqrt_total / (1 + N_child)
            score = Q + U

            if score > best_score:
                best_score  = score
                best_action = action

        return best_action

    def _get_or_create_child(self, node: MCTSNode, action: int) -> MCTSNode:
        child = node.children[action]
        if child is not None:
            return child

        # Node mới được tạo → tăng counter
        self._nodes_expanded += 1

        new_cube = apply_move_idx(node.cube, action)
        new_obs = encode_onehot(new_cube)

        curr_face = action // 3
        if node.last_face is not None and node.last_face == curr_face:
            new_repeat = node.repeat_count + 1
        else:
            new_repeat = 1

        mask = fast_legal_action_mask(curr_face, self.num_actions)

        child = MCTSNode(
            cube=new_cube, obs=new_obs, action_mask=mask,
            parent=node, prior=node.priors[action],
            last_face=curr_face, repeat_count=new_repeat,
            num_actions=self.num_actions,
        )
        node.children[action] = child
        node.num_children += 1
        return child

    def _evaluate(self, node: MCTSNode):
        if is_solved(node.cube):
            node.expanded = True
            if node.priors is None:
                node.priors = np.ones(self.num_actions, dtype=np.float32) / self.num_actions
            return 1.0

        obs_t = torch.as_tensor(
            node.obs, device=self.device
        ).to(dtype=self._model_dtype)

        with torch.inference_mode():
            policy, value = self.model.predict(obs_t)

        node.priors = policy.cpu().numpy()
        node.expanded = True
        return float(value.item())

    def _backup(self, path, value):
        for node, action in reversed(path):
            child = node.children[action]
            child.N += 1
            child.W += value
            child.Q = child.W / child.N
            node.total_children_N += 1
