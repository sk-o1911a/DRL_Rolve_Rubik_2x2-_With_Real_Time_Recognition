import numpy as np
import gymnasium as gym
import random

from gymnasium import spaces
from numba import njit, objmode

FACE_NAMES = ["U", "R", "F", "D", "L", "B"]
U, R, F, D, L, B = 0, 1, 2, 3, 4, 5

@njit(cache=True)
def solved_cube():
    cube = np.zeros((6, 2, 2), dtype=np.int8)
    for f in range(6):
        cube[f, :, :] = f
    return cube

@njit(cache=True, inline='always')
def rotate_face_clockwise(face):
    return np.rot90(face, -1)

@njit(cache=True, inline='always')
def rotate_face_counter_clockwise(face):
    return np.rot90(face, 1)

@njit(cache=True, inline='always')
def rotate_face_180(face):
    return np.rot90(face, 2)

@njit(cache=True, inline='always')
def deep_copy_cube(cube):
    return np.copy(cube)

@njit(cache=True, inline='always')
def copy_row(src_face, src_row):
    return src_face[src_row, :].copy()

@njit(cache=True, inline='always')
def copy_col(src_face, src_col):
    return src_face[:, src_col].copy()


####U####
@njit(cache=True)
def move_u_cw(cube):
    cube = deep_copy_cube(cube)
    cube[U] = rotate_face_clockwise(cube[U])
    temp_r_row0 = copy_row(cube[R], 0)
    temp_b_row0 = copy_row(cube[B], 0)
    temp_l_row0 = copy_row(cube[L], 0)
    temp_f_row0 = copy_row(cube[F], 0)

    cube[F][0, :] = temp_r_row0
    cube[R][0, :] = temp_b_row0
    cube[B][0, :] = temp_l_row0
    cube[L][0, :] = temp_f_row0
    return cube


####U'####
@njit(cache=True)
def move_u_ccw(cube):
    cube = deep_copy_cube(cube)
    cube[U] = rotate_face_counter_clockwise(cube[U])
    temp_l_row0 = copy_row(cube[L], 0)
    temp_b_row0 = copy_row(cube[B], 0)
    temp_r_row0 = copy_row(cube[R], 0)
    temp_f_row0 = copy_row(cube[F], 0)

    cube[F][0, :] = temp_l_row0
    cube[L][0, :] = temp_b_row0
    cube[B][0, :] = temp_r_row0
    cube[R][0, :] = temp_f_row0
    return cube


####U2####
@njit(cache=True)
def move_u2(cube):
    cube = deep_copy_cube(cube)
    cube[U] = rotate_face_180(cube[U])
    temp_f_row0 = copy_row(cube[F], 0)
    temp_b_row0 = copy_row(cube[B], 0)
    temp_l_row0 = copy_row(cube[L], 0)
    temp_r_row0 = copy_row(cube[R], 0)

    cube[F][0, :] = temp_b_row0
    cube[B][0, :] = temp_f_row0
    cube[L][0, :] = temp_r_row0
    cube[R][0, :] = temp_l_row0
    return cube


###R###
@njit(cache=True)
def move_r_cw(cube):
    cube = deep_copy_cube(cube)
    cube[R] = rotate_face_clockwise(cube[R])
    temp_u_col2 = copy_col(cube[U], 1)
    temp_f_col2 = copy_col(cube[F], 1)
    temp_d_col2 = copy_col(cube[D], 1)
    temp_b_col0 = copy_col(cube[B], 0)

    cube[U][:, 1] = temp_f_col2
    cube[F][:, 1] = temp_d_col2
    cube[D][:, 1] = temp_b_col0[::-1]
    cube[B][:, 0] = temp_u_col2[::-1]
    return cube


###R'###
@njit(cache=True)
def move_r_ccw(cube):
    cube = deep_copy_cube(cube)
    cube[R] = rotate_face_counter_clockwise(cube[R])
    temp_u_col2 = copy_col(cube[U], 1)
    temp_b_col0 = copy_col(cube[B], 0)
    temp_d_col2 = copy_col(cube[D], 1)
    temp_f_col2 = copy_col(cube[F], 1)

    cube[F][:, 1] = temp_u_col2
    cube[U][:, 1] = temp_b_col0[::-1]
    cube[B][:, 0] = temp_d_col2[::-1]
    cube[D][:, 1] = temp_f_col2
    return cube


###R2###
@njit(cache=True)
def move_r2(cube):
    cube = deep_copy_cube(cube)
    cube[R] = rotate_face_180(cube[R])
    temp_u_col2 = copy_col(cube[U], 1)
    temp_d_col2 = copy_col(cube[D], 1)
    temp_f_col2 = copy_col(cube[F], 1)
    temp_b_col0 = copy_col(cube[B], 0)

    cube[U][:, 1] = temp_d_col2
    cube[D][:, 1] = temp_u_col2
    cube[F][:, 1] = temp_b_col0[::-1]
    cube[B][:, 0] = temp_f_col2[::-1]
    return cube


###F###
@njit(cache=True)
def move_f_cw(cube):
    cube = deep_copy_cube(cube)
    cube[F] = rotate_face_clockwise(cube[F])
    temp_l_col2 = copy_col(cube[L], 1)
    temp_r_col0 = copy_col(cube[R], 0)
    temp_d_row0 = copy_row(cube[D], 0)
    temp_u_row2 = copy_row(cube[U], 1)

    cube[U][1, :] = temp_l_col2[::-1]
    cube[D][0, :] = temp_r_col0[::-1]
    cube[L][:, 1] = temp_d_row0
    cube[R][:, 0] = temp_u_row2
    return cube


###F'###
@njit(cache=True)
def move_f_ccw(cube):
    cube = deep_copy_cube(cube)
    cube[F] = rotate_face_counter_clockwise(cube[F])
    temp_r_col0 = copy_col(cube[R], 0)
    temp_l_col2 = copy_col(cube[L], 1)
    temp_u_row2 = copy_row(cube[U], 1)
    temp_d_row0 = copy_row(cube[D], 0)

    cube[U][1, :] = temp_r_col0
    cube[D][0, :] = temp_l_col2
    cube[L][:, 1] = temp_u_row2[::-1]
    cube[R][:, 0] = temp_d_row0[::-1]
    return cube


###F2###
@njit(cache=True)
def move_f2(cube):
    cube = deep_copy_cube(cube)
    cube[F] = rotate_face_180(cube[F])
    temp_u_row2 = copy_row(cube[U], 1)
    temp_d_row0 = copy_row(cube[D], 0)
    temp_l_col2 = copy_col(cube[L], 1)
    temp_r_col0 = copy_col(cube[R], 0)

    cube[U][1, :] = temp_d_row0[::-1]
    cube[D][0, :] = temp_u_row2[::-1]
    cube[L][:, 1] = temp_r_col0[::-1]
    cube[R][:, 0] = temp_l_col2[::-1]
    return cube

MOVE_FUNCS = {
    0: ("U", move_u_cw),
    1: ("U'", move_u_ccw),
    2: ("U2", move_u2),
    3: ("R", move_r_cw),
    4: ("R'", move_r_ccw),
    5: ("R2", move_r2),
    6: ("F", move_f_cw),
    7: ("F'", move_f_ccw),
    8: ("F2", move_f2),
}

NUM_ACTIONS = len(MOVE_FUNCS)

@njit(cache=True)
def apply_move_idx(cube, move_idx):
    if move_idx == 0:
        return move_u_cw(cube)
    elif move_idx == 1:
        return move_u_ccw(cube)
    elif move_idx == 2:
        return move_u2(cube)
    elif move_idx == 3:
        return move_r_cw(cube)
    elif move_idx == 4:
        return move_r_ccw(cube)
    elif move_idx == 5:
        return move_r2(cube)
    elif move_idx == 6:
        return move_f_cw(cube)
    elif move_idx == 7:
        return move_f_ccw(cube)
    elif move_idx == 8:
        return move_f2(cube)
    else:
        raise ValueError("Invalid move_idx")


@njit(cache=True)
def scramble(cube, k=8, seed=None):
    if seed is not None:
        np.random.seed(seed)
    last = -1
    applied = 0
    while applied < k:
        move = np.random.randint(NUM_ACTIONS)
        if last != -1 and move // 3 == last // 3:
            continue
        cube = apply_move_idx(cube, move)
        last = move
        applied += 1
    return cube

@njit(cache=True)
def is_solved(cube):
    for f in range(6):
        c = cube[f, 0, 0]
        if cube[f, 0, 1] != c: return False
        if cube[f, 1, 0] != c: return False
        if cube[f, 1, 1] != c: return False
    return True

@njit(cache=True)
def encode_onehot(cube):
    arr = cube.reshape(-1)
    result = np.zeros((len(arr), 6), dtype=np.float32)
    for i in range(len(arr)):
        result[i, arr[i]] = 1.0
    return result

@njit(cache=True)
def fast_legal_action_mask(last_face, num_actions=9):
    mask = np.ones(num_actions, dtype=np.bool_)
    
    if last_face == -1:
        return mask

    base = last_face * 3
    mask[base:base+3] = False
    
    return mask
    

@njit(cache=True)
def apply_random_map(cube, map_arr):
    new_cube = np.zeros_like(cube)
    for r in range(6):
        for i in range(2):
            for j in range(2):
                old_color = cube[r, i, j]
                new_cube[r, i, j] = map_arr[old_color]
    return new_cube
    
class Rubik2x2Env(gym.Env):
    metadata = {"render_modes": ["ansi", "none"]}
    def __init__(
        self,
        scramble_len: int = 4,
        max_steps: int = 100,
        use_action_mask: bool = True,
        render_mode: str = "none",
        seed: int | None = None,
        color_augmentation: bool = False,
    ):
        super().__init__()
        self.scramble_len   = int(scramble_len)
        self.max_steps      = int(max_steps)
        self.use_action_mask = bool(use_action_mask)
        self.render_mode    = render_mode
        self.color_augmentation = color_augmentation
        
        # Spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(24,6),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Episode state
        self.cube = solved_cube()
        self.steps = 0
        self._last_face = -1 


    # Create mask
    def _legal_action_mask(self) -> np.ndarray:
        return fast_legal_action_mask(self._last_face, self.action_space.n)


    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.cube = solved_cube()
        if self.color_augmentation:
            perm = np.random.permutation(6).astype(np.int8)
            self.cube = apply_random_map(self.cube, perm)
        
        prev_len = max(1, self.scramble_len - 1)
        actual_k = int(np.random.choice(a=[prev_len, self.scramble_len], p=[0.2, 0.8]))
        
        self.cube = scramble(self.cube, k=actual_k)
        self.steps = 0
        self._last_face = -1
        
        obs = encode_onehot(self.cube)
        
        info: dict = {}
        if self.use_action_mask:
            info["action_mask"] = self._legal_action_mask()
        return obs, info


    def step(self, action: int):
        assert self.action_space.contains(action), f"invalid action: {action}"
        # move the cube
        self.cube = apply_move_idx(self.cube, action)
        self.steps += 1
 
        self._last_face = action // 3

        # check if solved
        solved = is_solved(self.cube)
        terminated = solved
        truncated = self.steps >= self.max_steps

        # obs & info
        obs = encode_onehot(self.cube)
        info = {"solved": solved}

        #update last mask
        if self.use_action_mask:
            info["action_mask"] = self._legal_action_mask()

        reward = 0.0
        return obs, reward, terminated, truncated, info

    def as_ascii(self) -> str:
        faces = self.cube

        def row_to_str(row):
            return " ".join(str(int(x)) for x in row)

        lines = []
        # U
        lines.append("     " + row_to_str(faces[U][0]))
        lines.append("     " + row_to_str(faces[U][1]))
        # L F R B
        for r in range(2):
            lines.append(
                row_to_str(faces[L][r]) + "  " +
                row_to_str(faces[F][r]) + "  " +
                row_to_str(faces[R][r]) + "  " +
                row_to_str(faces[B][r])
            )
        # D
        lines.append("     " + row_to_str(faces[D][0]))
        lines.append("     " + row_to_str(faces[D][1]))
        return "\n".join(lines)

    def get_action_meanings(self):
        return [MOVE_FUNCS[i][0] for i in range(self.action_space.n)]

    def close(self):
        pass
