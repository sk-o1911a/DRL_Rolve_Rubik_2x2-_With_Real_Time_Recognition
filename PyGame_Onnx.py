import os
import faulthandler
import serial

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

faulthandler.enable()

import sys
import json
import time
import numpy as np
import pygame
import gc

try:
    ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
    print("[UART] Connected to ESP32")
except Exception as e:
    ser = None
    print(f"[UART] Error: {e}")
    
try:
    from RPLCD.i2c import CharLCD
    lcd = CharLCD('PCF8574', 0x27)
    lcd.clear()
except Exception as e:
    lcd = None
    print(f"[LCD] Warning: {e}")

from Rubik2x2Env import (
    Rubik2x2Env,
    solved_cube,
    encode_onehot,
    is_solved,
    apply_move_idx,
)

from Onnx_Model import RubikONNXModel
from MCTS_Core import MCTS
from Action_MCTS import pick_action_from_mcts

# Camera 
CAMERA_AVAILABLE = False
RubikCamera = None
process_region = None
cv2 = None


def try_init_camera_modules():
    global CAMERA_AVAILABLE, RubikCamera, process_region, cv2
    if CAMERA_AVAILABLE:
        return True
    try:
        from camera import RubikCamera as _RubikCamera, process_region as _process_region
        import cv2 as _cv2
        RubikCamera = _RubikCamera
        process_region = _process_region
        cv2 = _cv2
        CAMERA_AVAILABLE = True
        return True
    except Exception as e:
        CAMERA_AVAILABLE = False
        print(f"[PyGame] Camera module not available — PLAY mode disabled ({e})")
        return False


# ---------- Load BEST_C_PUCT ----------
try:
    with open("pbt_logs/best_hyperparams.json", "r") as f:
        BEST_C_PUCT = json.load(f)["hyperparams"]["c_puct"]
except Exception:
    BEST_C_PUCT = 3.5


def validate_scanned_cube(cube):
    corners = [
        ("URF", [cube[0,1,1], cube[1,0,0], cube[2,0,1]]),
        ("UFL", [cube[0,1,0], cube[2,0,0], cube[4,0,1]]),
        ("ULB", [cube[0,0,0], cube[4,0,0], cube[5,0,1]]),
        ("UBR", [cube[0,0,1], cube[5,0,0], cube[1,0,1]]),
        ("DFR", [cube[3,0,1], cube[2,1,1], cube[1,1,0]]),
        ("DLF", [cube[3,0,0], cube[4,1,1], cube[2,1,0]]),
        ("DBL", [cube[3,1,0], cube[5,1,1], cube[4,1,0]]),
        ("DRB", [cube[3,1,1], cube[1,1,1], cube[5,1,0]]),
    ]
    for name, corner in corners:
        if len(set(int(x) for x in corner)) < 3:
            return False, f"Góc {name} bị xoắn/trùng màu."
    return True, "Hợp lệ"


U_face, R_face, F_face, D_face, L_face, B_face = 0, 1, 2, 3, 4, 5
MOVE_NAMES_18 = ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2", "L", "L'", "L2", "B", "B'", "B2", "D", "D'", "D2"]
MOVE_NAMES_9 = ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2"]


def apply_manual_move(cube, move_idx):
    if move_idx < 9:
        return apply_move_idx(cube, move_idx)
    c = np.copy(cube)
    if move_idx == 9:
        c[L_face] = np.rot90(c[L_face], -1)
        tu, td, tb, tf = c[U_face][:, 0].copy(), c[D_face][:, 0].copy(), c[B_face][:, 1].copy(), c[F_face][:, 0].copy()
        c[F_face][:, 0] = tu; c[B_face][:, 1] = td[::-1]; c[U_face][:, 0] = tb[::-1]; c[D_face][:, 0] = tf
    elif move_idx == 10:
        c[L_face] = np.rot90(c[L_face], 1)
        tu, td, tb, tf = c[U_face][:, 0].copy(), c[D_face][:, 0].copy(), c[B_face][:, 1].copy(), c[F_face][:, 0].copy()
        c[F_face][:, 0] = td; c[B_face][:, 1] = tu[::-1]; c[U_face][:, 0] = tf; c[D_face][:, 0] = tb[::-1]
    elif move_idx == 11:
        c[L_face] = np.rot90(c[L_face], 2)
        tu, td, tb, tf = c[U_face][:, 0].copy(), c[D_face][:, 0].copy(), c[B_face][:, 1].copy(), c[F_face][:, 0].copy()
        c[U_face][:, 0] = td; c[D_face][:, 0] = tu; c[F_face][:, 0] = tb[::-1]; c[B_face][:, 1] = tf[::-1]
    elif move_idx == 12:
        c[B_face] = np.rot90(c[B_face], -1)
        tr, tl, tu, td = c[R_face][:, 1].copy(), c[L_face][:, 0].copy(), c[U_face][0, :].copy(), c[D_face][1, :].copy()
        c[U_face][0, :] = tr; c[D_face][1, :] = tl; c[L_face][:, 0] = tu[::-1]; c[R_face][:, 1] = td[::-1]
    elif move_idx == 13:
        c[B_face] = np.rot90(c[B_face], 1)
        tr, tl, tu, td = c[R_face][:, 1].copy(), c[L_face][:, 0].copy(), c[U_face][0, :].copy(), c[D_face][1, :].copy()
        c[U_face][0, :] = tl[::-1]; c[D_face][1, :] = tr[::-1]; c[L_face][:, 0] = td; c[R_face][:, 1] = tu
    elif move_idx == 14:
        c[B_face] = np.rot90(c[B_face], 2)
        tr, tl, tu, td = c[R_face][:, 1].copy(), c[L_face][:, 0].copy(), c[U_face][0, :].copy(), c[D_face][1, :].copy()
        c[U_face][0, :] = td[::-1]; c[D_face][1, :] = tu[::-1]; c[L_face][:, 0] = tr[::-1]; c[R_face][:, 1] = tl[::-1]
    elif move_idx == 15:
        c[D_face] = np.rot90(c[D_face], -1)
        tf, tb, tl, tr = c[F_face][1, :].copy(), c[B_face][1, :].copy(), c[L_face][1, :].copy(), c[R_face][1, :].copy()
        c[F_face][1, :] = tl; c[L_face][1, :] = tb; c[B_face][1, :] = tr; c[R_face][1, :] = tf
    elif move_idx == 16:
        c[D_face] = np.rot90(c[D_face], 1)
        tf, tb, tl, tr = c[F_face][1, :].copy(), c[B_face][1, :].copy(), c[L_face][1, :].copy(), c[R_face][1, :].copy()
        c[F_face][1, :] = tr; c[R_face][1, :] = tb; c[B_face][1, :] = tl; c[L_face][1, :] = tf
    elif move_idx == 17:
        c[D_face] = np.rot90(c[D_face], 2)
        tf, tb, tl, tr = c[F_face][1, :].copy(), c[B_face][1, :].copy(), c[L_face][1, :].copy(), c[R_face][1, :].copy()
        c[F_face][1, :] = tb; c[B_face][1, :] = tf; c[L_face][1, :] = tr; c[R_face][1, :] = tl
    return c


# UI config
WIDTH, HEIGHT = 1024, 600

BG_COLOR = (45, 52, 54)
PANEL_COLOR = (30, 39, 46)
PANEL_BORDER = (99, 110, 114)
TEXT_COLOR = (255, 255, 255)
TEXT_DARK = (45, 52, 54)
GRAY_COLOR = (128, 128, 128)

CUBE_COLORS = {
    0: (255, 255, 255), 1: (220, 20, 20), 2: (0, 180, 0),
    3: (255, 220, 0),   4: (255, 140, 0), 5: (0, 100, 255),
}
DRAW_COLORS_CAMERA = {
    "W": (255, 255, 255), "R": (220, 20, 20), "G": (0, 180, 0),
    "Y": (255, 220, 0), "O": (255, 140, 0), "B": (0, 100, 255)
}
BTN_COLORS = {
    "normal": (99, 110, 114), "hover": (130, 140, 145), "active": (0, 184, 148),
    "danger": (214, 48, 49), "warning": (253, 203, 110), "success": (0, 184, 148), "disabled": (60, 60, 60),
}

# rubik faces smaller
TILE = 40
MARGIN = 5
CUBE_OFFSET_X = 40
CUBE_OFFSET_Y = 55
FACE_GAP_VERTICAL = 0

FACE_POS = {
    "U": (WIDTH // 2 - int(1.5 * TILE) + CUBE_OFFSET_X, CUBE_OFFSET_Y),
    "L": (WIDTH // 2 - int(4.5 * TILE) - 12 + CUBE_OFFSET_X, CUBE_OFFSET_Y + 3 * TILE),
    "F": (WIDTH // 2 - int(1.5 * TILE) + CUBE_OFFSET_X, CUBE_OFFSET_Y + 3 * TILE),
    "R": (WIDTH // 2 + int(1.5 * TILE) + 12 + CUBE_OFFSET_X, CUBE_OFFSET_Y + 3 * TILE),
    "B": (WIDTH // 2 + int(4.5 * TILE) + 24 + CUBE_OFFSET_X, CUBE_OFFSET_Y + 3 * TILE),
    "D": (WIDTH // 2 - int(1.5 * TILE) + CUBE_OFFSET_X, CUBE_OFFSET_Y + 6 * TILE),
}

CAMERA_W, CAMERA_H = 320, 230
CAMERA_X, CAMERA_Y = WIDTH - CAMERA_W - 16, 280

SCAN_ORDER = ["U", "F", "D", "B", "L", "R"]
SCAN_DELAYS_MS = [2000, 2000, 2800, 4000, 3100, 2200]
FACE_INDEX_MAP = {"U": 0, "R": 1, "F": 2, "D": 3, "L": 4, "B": 5}

CLICK_GUARD_MS = 500
KEY_GUARD_MS = 350

global_camera_instance = None


class Button:
    def __init__(self, x, y, w, h, text, callback=None, color="normal", font=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback
        self.color_key = color
        self.font = font
        self.hover = False
        self.enabled = True
        self.visible = True

    def draw(self, surface):
        if not self.visible:
            return
        col = BTN_COLORS["disabled"] if not self.enabled else (BTN_COLORS["hover"] if self.hover else BTN_COLORS.get(self.color_key, BTN_COLORS["normal"]))
        tc = (100, 100, 100) if not self.enabled else (TEXT_DARK if self.color_key == "warning" and not self.hover else TEXT_COLOR)
        pygame.draw.rect(surface, col, self.rect, border_radius=10)
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 2, border_radius=10)
        t = self.font.render(self.text, True, tc)
        surface.blit(t, t.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if not self.visible or not self.enabled:
            return False
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos) and self.callback:
            self.callback()
            return True
        return False


class InputBox:
    def __init__(self, x, y, w, h, text="", label=""):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.label = label
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.unicode.isdigit() and len(self.text) < 3:
                self.text += event.unicode

    def draw(self, surface, panel_x=0, panel_w=0):
        if self.label:
            lbl = FONT_SMALL.render(self.label, True, TEXT_COLOR)
            surface.blit(lbl, (panel_x + (panel_w - lbl.get_width()) // 2 if panel_w > 0 else self.rect.x, self.rect.y - 20))
        pygame.draw.rect(surface, (255, 255, 200) if self.active else (255, 255, 255), self.rect, border_radius=6)
        pygame.draw.rect(surface, (0, 184, 148) if self.active else (100, 100, 100), self.rect, 2, border_radius=6)
        t = FONT.render(self.text, True, TEXT_DARK)
        surface.blit(t, t.get_rect(center=self.rect.center))

    def get_value(self, default=5):
        try:
            return max(1, min(int(self.text), 50))
        except:
            return default


class Slider:
    def __init__(self, x, y, w, min_val, max_val, initial, label=""):
        self.rect = pygame.Rect(x, y, w, 20)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.label = label
        self.dragging = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.dragging = True
            self._upd(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._upd(event.pos[0])

    def _upd(self, mx):
        self.value = int(self.min_val + max(0, min(1, (mx - self.rect.x) / self.rect.width)) * (self.max_val - self.min_val))

    def draw(self, surface, panel_x=0, panel_w=0):
        if self.label:
            lbl = FONT_SMALL.render(f"{self.label}: {self.value}", True, TEXT_COLOR)
            surface.blit(lbl, (panel_x + (panel_w - lbl.get_width()) // 2 if panel_w > 0 else self.rect.x, self.rect.y - 20))
        pygame.draw.rect(surface, (80, 80, 80), self.rect, border_radius=10)
        fw = int(self.rect.width * (self.value - self.min_val) / (self.max_val - self.min_val))
        pygame.draw.rect(surface, (0, 184, 148), pygame.Rect(self.rect.x, self.rect.y, fw, self.rect.height), border_radius=10)
        pygame.draw.circle(surface, (255, 255, 255), (self.rect.x + fw, self.rect.centery), 10)
        pygame.draw.circle(surface, (0, 184, 148), (self.rect.x + fw, self.rect.centery), 6)


def draw_face(surface, cube, face_idx, face_name, mode="SIM", raw_scans=None):
    ox, oy = FACE_POS[face_name]
    if mode == "PLAY":
        if raw_scans and face_name in raw_scans:
            face_strs = raw_scans[face_name]
            for r in range(2):
                for c in range(2):
                    color_str = face_strs[r * 2 + c]
                    color = DRAW_COLORS_CAMERA.get(color_str, (180, 50, 180))
                    x = ox + c * (TILE + MARGIN)
                    y = oy + r * (TILE + MARGIN)
                    pygame.draw.rect(surface, (20, 20, 20), (x + 2, y + 2, TILE, TILE), border_radius=5)
                    pygame.draw.rect(surface, color, (x, y, TILE, TILE), border_radius=5)
                    pygame.draw.rect(surface, (30, 30, 30), (x, y, TILE, TILE), 2, border_radius=5)
        else:
            for r in range(2):
                for c in range(2):
                    x = ox + c * (TILE + MARGIN)
                    y = oy + r * (TILE + MARGIN)
                    pygame.draw.rect(surface, (20, 20, 20), (x + 2, y + 2, TILE, TILE), border_radius=5)
                    pygame.draw.rect(surface, GRAY_COLOR, (x, y, TILE, TILE), border_radius=5)
                    pygame.draw.rect(surface, (30, 30, 30), (x, y, TILE, TILE), 2, border_radius=5)
    else:
        face = cube[face_idx]
        for r in range(2):
            for c in range(2):
                color = CUBE_COLORS.get(int(face[r, c]), (128, 128, 128))
                x = ox + c * (TILE + MARGIN)
                y = oy + r * (TILE + MARGIN)
                pygame.draw.rect(surface, (20, 20, 20), (x + 2, y + 2, TILE, TILE), border_radius=5)
                pygame.draw.rect(surface, color, (x, y, TILE, TILE), border_radius=5)
                pygame.draw.rect(surface, (30, 30, 30), (x, y, TILE, TILE), 2, border_radius=5)

    lbl = FONT.render(face_name, True, TEXT_COLOR)
    surface.blit(lbl, (ox + (2 * TILE + MARGIN) // 2 - lbl.get_width() // 2, oy - 24))


def draw_cube(surface, cube, mode="SIM", raw_scans=None):
    for name, idx in [("U", 0), ("L", 4), ("F", 2), ("R", 1), ("B", 5), ("D", 3)]:
        draw_face(surface, cube, idx, name, mode, raw_scans)


def draw_panel(surface, x, y, w, h, title=""):
    r = pygame.Rect(x, y, w, h)
    pygame.draw.rect(surface, PANEL_COLOR, r, border_radius=12)
    pygame.draw.rect(surface, PANEL_BORDER, r, 2, border_radius=12)
    if title:
        ts = FONT.render(title, True, TEXT_COLOR)
        surface.blit(ts, (x + (w - ts.get_width()) // 2, y + 6))
        pygame.draw.line(surface, PANEL_BORDER, (x + 10, y + 34), (x + w - 10, y + 34), 1)
    return y + 42 if title else y + 10


def draw_camera_feed(surface, camera_frame):
    if camera_frame is None or not CAMERA_AVAILABLE:
        return
    try:
        frame_display = camera_frame.copy()
        frame_lab = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2LAB)
        points = [(130, 170), (130, 100), (180, 170), (180, 100)]
        for i, (x, y) in enumerate(points):
            cn, dist = process_region(frame_lab, x, y, region_size=30)
            bc = (0, 255, 0) if dist < 40 else (255, 0, 0)
            cv2.rectangle(frame_display, (x - 15, y - 15), (x + 15, y + 15), bc, 2)
            cv2.putText(frame_display, f"{i+1}:{cn}", (x + 30, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        fs = pygame.surfarray.make_surface(frame_display.swapaxes(0, 1))
        cr = pygame.Rect(CAMERA_X - 6, CAMERA_Y - 6, CAMERA_W + 12, CAMERA_H + 12)
        pygame.draw.rect(surface, PANEL_COLOR, cr, border_radius=10)
        pygame.draw.rect(surface, PANEL_BORDER, cr, 2, border_radius=10)
        tl = FONT_SMALL.render("Live Camera", True, TEXT_COLOR)
        surface.blit(tl, (CAMERA_X + (CAMERA_W - tl.get_width()) // 2, CAMERA_Y - 20))
        surface.blit(fs, (CAMERA_X, CAMERA_Y))
    except Exception as e:
        print(f"[Camera Feed] Error: {e}")


def draw_status_panel(surface, cube, manual_steps, play_solve, is_thinking,formula, move_count, moves_remaining, scramble_text,mode="SIM", current_scan_face=None, scan_progress="",scan_auto_running=False, show_play_scan_info=True):
    px, py = 274, 500
    pw = 736
    ph = 96
    cy = draw_panel(surface, px, py, pw, ph, "Status")
    cx = px + pw // 2

    mode_label = "SIMULATION" if mode == "SIM" else "PLAY (Camera)"
    mode_color = (116, 185, 255) if mode == "SIM" else (0, 184, 148)
    surface.blit(FONT.render(mode_label, True, mode_color), (px + 12, cy - 2))

    if mode == "PLAY":
        if is_thinking:
            st, sc = "AI is thinking...", (253, 203, 110)
        elif formula.startswith("Lỗi"):
            st, sc = "Invalid Cube Scanned", (214, 48, 49)
        elif formula and not formula.startswith("Lỗi"):
            st, sc = "Formula Ready", (0, 230, 118)
        elif scan_auto_running:
            st, sc = "Auto Scanning...", (253, 203, 110)
        else:
            st, sc = "Waiting Scan/Solve", (200, 200, 200) if show_play_scan_info else ("", (200, 200, 200))
    else:
        if is_thinking:
            st, sc = "AI is thinking...", (253, 203, 110)
        elif is_solved(cube):
            st, sc = "SOLVED!", (0, 230, 118)
        else:
            st, sc = "Not Solved", (255, 118, 117)

    ss = FONT_LARGE.render(st, True, sc)
    surface.blit(ss, ss.get_rect(center=(cx, cy + 12)))
    iy = cy + 30

    if mode == "SIM":
        surface.blit(FONT.render(f"Manual Steps: {manual_steps}", True, (200, 200, 200)), (px + 12, iy - 2))
    else:
        if show_play_scan_info:
            if scan_auto_running and current_scan_face:
                info = f"Auto scanning: {current_scan_face} | {scan_progress}"
            elif current_scan_face:
                info = f"Ready to scan: {current_scan_face} | {scan_progress}"
            else:
                info = f"{scan_progress} | Press Scan/SPACE"
            surf = FONT.render(info, True, (200, 200, 200))
            surface.blit(surf, surf.get_rect(center=(cx, iy)))

    fy = iy + (2 if mode == "PLAY" else 18)
    if formula and not is_thinking:
        lbl = "Solution:" if not formula.startswith("Lỗi") else "Error:"
        c_text = (150, 150, 150) if not formula.startswith("Lỗi") else (214, 48, 49)
        surface.blit(FONT_SMALL.render(lbl, True, c_text), (px + 12, fy))

        mw = pw - 140
        words = formula.split()
        lines, cur = [], ""
        for w in words:
            test = cur + " " + w if cur else w
            if FONT_SMALL.size(test)[0] < mw:
                cur = test
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)

        for i, line in enumerate(lines[:1]):
            color = (129, 236, 236) if not formula.startswith("Lỗi") else (255, 200, 200)
            surface.blit(FONT_SMALL.render(line, True, color), (px + 90, fy + i * 18))


def solve_with_mcts(cube, model, device, num_simulations=400, max_steps=30):
    temp_env = Rubik2x2Env(scramble_len=1, max_steps=max_steps, use_action_mask=True)
    temp_env.cube = cube.copy()
    temp_env.steps = 0
    temp_env._last_face = -1

    obs = encode_onehot(temp_env.cube)
    action_mask = temp_env._legal_action_mask()
    mcts = MCTS(model=model, num_actions=9, num_simulations=num_simulations, c_puct=BEST_C_PUCT)

    actions, names = [], []
    for _ in range(max_steps):
        if is_solved(temp_env.cube):
            break
        vc, _ = mcts.run(temp_env.cube, obs, action_mask)
        act = pick_action_from_mcts(vc, mode="greedy")
        obs, _, terminated, _, info = temp_env.step(act)
        action_mask = info.get("action_mask", None)
        actions.append(act)
        names.append(MOVE_NAMES_9[act])
        if terminated:
            break
    return actions, " ".join(names)


def scan_face_from_camera(camera_frame):
    if camera_frame is None or not CAMERA_AVAILABLE:
        return None
    try:
        frame_lab = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2LAB)
        points = [(130, 170), (130, 100), (180, 170), (180, 100)]
        names = []
        for (x, y) in points:
            cn, _ = process_region(frame_lab, x, y, region_size=30)
            names.append(cn)
        return names
    except Exception as e:
        print(f"[Scan] Error: {e}")
        return None

def is_cube_present(camera_frame):
    if camera_frame is None or not CAMERA_AVAILABLE:
        return True 

    x1, y1 = 80, 50
    x2, y2 = 210, 200

    h, w = camera_frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    roi = camera_frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 35, 100)

    edge_count = cv2.countNonZero(edges)

    return edge_count < 1800

def init_camera():
    global global_camera_instance
    if not try_init_camera_modules():
        return False
    if global_camera_instance is None:
        global_camera_instance = RubikCamera(resolution=(320, 240))
    return global_camera_instance.start()


def cleanup_camera():
    if global_camera_instance:
        global_camera_instance.stop()


def get_camera_frame():
    if global_camera_instance is None:
        return None
    try:
        frame = global_camera_instance.get_frame()
        if frame is None or frame.size == 0 or len(frame.shape) != 3:
            return None
        return frame.copy()
    except Exception:
        return None


def main():
    global FONT, FONT_SMALL, FONT_LARGE, FONT_TITLE, CLOCK, SCREEN

    pygame.init()
    pygame.display.set_caption("Rubik 2x2 Solver")
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))  # no fullscreen

    FONT = pygame.font.SysFont("consolas", 19)
    FONT_SMALL = pygame.font.SysFont("consolas", 14)
    FONT_LARGE = pygame.font.SysFont("consolas", 24, bold=True)
    FONT_TITLE = pygame.font.SysFont("consolas", 28, bold=True)
    CLOCK = pygame.time.Clock()

    device = "cpu"
    print(f"[PyGame] Device: {device}")

    model_loaded = False
    model = None
    try:
        model = RubikONNXModel("rubik2x2.onnx", device=device)
        model_loaded = True
        print("[PyGame] ONNX loaded!")
    except Exception as e:
        print(f"[PyGame] ONNX error: {e}")

    cube = solved_cube()
    raw_scans = {}
    manual_steps = 0
    solve_formula = ""
    current_scramble = ""
    solve_moves = []
    play_solve = False
    next_move_time = 0
    move_count = 0
    is_thinking = False
    mode = "SIM"
    current_scan_idx = 0
    scanned_faces = set()
    camera_frame = None
    camera_stopped_for_solve = False

    scan_auto_running = False
    next_auto_scan_time = 0
    show_play_scan_info = True
    
    serial_moves_queue = []
    waiting_for_esp = False
    wait_esp_start_time = 0
    
    solve_start_time = 0
    solve_timer_running = False
    final_solve_time = 0.0
    last_lcd_update_time = 0
        
    last_click_ms = {}
    last_key_ms = {}

    def click_allowed(action_name: str) -> bool:
        now = pygame.time.get_ticks()
        prev = last_click_ms.get(action_name, -10000)
        if now - prev < CLICK_GUARD_MS:
            return False
        last_click_ms[action_name] = now
        return True

    def key_allowed(key_name: str) -> bool:
        now = pygame.time.get_ticks()
        prev = last_key_ms.get(key_name, -10000)
        if now - prev < KEY_GUARD_MS:
            return False
        last_key_ms[key_name] = now
        return True

    def hard_reset_state():
        nonlocal show_play_scan_info
        nonlocal cube, manual_steps, solve_formula, solve_moves, play_solve
        nonlocal move_count, current_scramble, raw_scans
        nonlocal current_scan_idx, scanned_faces
        nonlocal scan_auto_running, next_auto_scan_time
        nonlocal serial_moves_queue, waiting_for_esp
        nonlocal solve_start_time, solve_timer_running, final_solve_time
        serial_moves_queue.clear()
        solve_start_time = 0
        solve_timer_running = False
        final_solve_time = 0.0
        cube = solved_cube()
        manual_steps = 0
        solve_formula = ""
        solve_moves = []
        play_solve = False
        move_count = 0
        current_scramble = ""
        raw_scans.clear()
        current_scan_idx = 0
        scanned_faces.clear()
        scan_auto_running = False
        next_auto_scan_time = 0
        show_play_scan_info = True
        waiting_for_esp = False
        
        if lcd:
            try:
                lcd.cursor_pos = (0, 0)
                lcd.write_string("System Ready    ")
                lcd.cursor_pos = (1, 0)
                lcd.write_string("Waiting scan... ")
            except:
                pass

    def apply_user_move(move_idx):
        nonlocal cube, manual_steps
        cube = apply_manual_move(cube, move_idx)
        manual_steps += 1

    def run_solver_logic():
        nonlocal solve_moves, solve_formula, play_solve, move_count, is_thinking, next_move_time, serial_moves_queue, waiting_for_esp
        try:
            acts, formula = solve_with_mcts(cube, model, device, num_simulations=400, max_steps=50)
            solve_moves = acts
            solve_formula = formula
            move_count = len(acts)

            if move_count > 0:
                if mode == "SIM":
                    play_solve = True
                    next_move_time = pygame.time.get_ticks() + 500
                else:
                    play_solve = False 
                    
                if mode == "PLAY" and ser and ser.is_open:
                    ser.reset_input_buffer()
                    move_list = formula.split()
                    move_list.append("finish")
                    serial_moves_queue = move_list
                    waiting_for_esp = False
                    
                    #for i, move in enumerate(moves):
                        #msg = f"play {move}\n"
                        #ser.write(msg.encode("utf-8"))
                        #print(f"[UART] Sent: {msg.strip()} ({i+1}/{len(moves)})")
                    
                    #msg = f"Play {formula}\n"
                    #ser.write(msg.encode("utf-8"))
                    #print(f"[UART] Sent: {msg.strip()}")
            else:
                solve_formula = "Thất bại / Đã giải"
                if mode == "PLAY" and ser and ser.is_open:
                    ser.write("Not solve\n".encode("utf-8"))
                    print("[UART] Sent: Not solve")
        except Exception as e:
            solve_formula = f"Lỗi thuật toán: {e}"
        finally:
            is_thinking = False

    def do_solve():
        nonlocal solve_formula, is_thinking, camera_stopped_for_solve, cube, camera_frame, show_play_scan_info
        if is_thinking or not click_allowed("solve"):
            return
        if not model_loaded or model is None:
            solve_formula = "Chưa load mô hình AI!"
            return

        if mode == "PLAY":
            if len(raw_scans) < 6:
                solve_formula = "Lỗi: Bạn chưa quét đủ 6 mặt!"
                return

            all_colors = []
            for f in ["U", "F", "D", "B", "L", "R"]:
                all_colors.extend(raw_scans[f])

            unique_colors = list(set(all_colors))
            if len(unique_colors) != 6:
                solve_formula = f"Lỗi: Quét ra {len(unique_colors)} màu khác nhau (Yêu cầu đúng 6 màu)."
                return

            for c in unique_colors:
                if all_colors.count(c) != 4:
                    solve_formula = f"Lỗi: Màu '{c}' xuất hiện {all_colors.count(c)} lần (Phải đúng 4 lần)."
                    return

            dynamic_map = {c: i for i, c in enumerate(unique_colors)}
            for f_name, f_idx in FACE_INDEX_MAP.items():
                mapped_ints = [dynamic_map[c] for c in raw_scans[f_name]]
                cube[f_idx] = np.array(mapped_ints, dtype=np.int8).reshape(2, 2)

            is_valid, err_msg = validate_scanned_cube(cube)
            if not is_valid:
                solve_formula = f"Lỗi khối Rubik Fake: {err_msg}"
                return

        if is_solved(cube) and mode == "SIM":
            solve_formula = "Already solved!"
            return

        if mode == "PLAY" and global_camera_instance:
            #cleanup_camera()
            camera_stopped_for_solve = True
            #time.sleep(0.5)
            #camera_frame = None
            gc.collect()

        solve_formula = "AI is thinking..."
        is_thinking = True
        if mode == "PLAY":
            show_play_scan_info = False
        run_solver_logic()

    def do_scramble():
        nonlocal cube, manual_steps, solve_formula, solve_moves, play_solve, move_count, current_scramble
        if is_thinking or not click_allowed("scramble"):
            return
        cube = solved_cube()
        manual_steps = 0
        k = input_scramble.get_value(5)
        names = []
        last_face = -1
        for _ in range(k):
            while True:
                idx = np.random.randint(0, 18)
                if idx // 3 != last_face:
                    break
            cube = apply_manual_move(cube, idx)
            last_face = idx // 3
            names.append(MOVE_NAMES_18[idx])
        current_scramble = " ".join(names)
        solve_formula, solve_moves, play_solve, move_count = "", [], False, 0

    def do_reset():
        if is_thinking or not click_allowed("reset"):
            return
        hard_reset_state()

    def do_reset_play():
        if is_thinking or not click_allowed("reset_play"):
            return
        hard_reset_state()
        if ser and ser.is_open:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            ser.write("reset\n".encode("utf-8"))
            print("[UART] Sent: reset")

    def do_stop():
        nonlocal play_solve, solve_moves
        if is_thinking or not click_allowed("stop"):
            return
        play_solve = False
        solve_moves = []

    def toggle_mode():
        nonlocal mode
        if is_thinking or not click_allowed("mode"):
            return
        if not CAMERA_AVAILABLE and mode == "SIM":
            if not try_init_camera_modules():
                return
        if mode == "SIM":
            mode = "PLAY"
            init_camera()
        else:
            mode = "SIM"
            cleanup_camera()
        hard_reset_state()

    def scan_current_face():
        nonlocal current_scan_idx, scan_auto_running
        if mode != "PLAY" or current_scan_idx >= 6:
            return
        face_name = SCAN_ORDER[current_scan_idx]
        color_strings = scan_face_from_camera(camera_frame)
        if color_strings is not None:
            raw_scans[face_name] = color_strings
            scanned_faces.add(face_name)
            if ser and ser.is_open:
                ser.write(b"scan\n")
                print("[UART] Sent: scan")
            current_scan_idx += 1
            print(f"[Scan] {face_name}: {color_strings}")
            if current_scan_idx >= 6:
                scan_auto_running = False

    def start_auto_scan():
        nonlocal scan_auto_running, next_auto_scan_time
        if mode != "PLAY" or is_thinking or not click_allowed("scan_start"):
            return
        if current_scan_idx >= 6 or scan_auto_running:
            return
        scan_auto_running = True
        next_auto_scan_time = pygame.time.get_ticks()

    # Panels & buttons
    move_panel_x, move_panel_y = 10, 50
    move_panel_w, move_panel_h = 250, 300

    ctrl_panel_x, ctrl_panel_y = 10, 355
    ctrl_panel_w, ctrl_panel_h = 250, 235
    
    ctrl_panel_x_c, ctrl_panel_y_c = 10, 50
    ctrl_panel_w_c, ctrl_panel_h_c = 250, 540

    buttons = []

    move_groups = [
        [("U", 0), ("U'", 1), ("U2", 2)],
        [("R", 3), ("R'", 4), ("R2", 5)],
        [("F", 6), ("F'", 7), ("F2", 8)],
        [("L", 9), ("L'", 10), ("L2", 11)],
        [("B", 12), ("B'", 13), ("B2", 14)],
        [("D", 15), ("D'", 16), ("D2", 17)],
    ]

    sx, sy = move_panel_x + 13, move_panel_y + 46
    gx, gy = 75, 40
    btn_w, btn_h = 68, 34

    for ri, group in enumerate(move_groups):
        for ci, (label, act) in enumerate(group):
            def make_cb(a=act):
                def cb():
                    if not is_thinking and mode == "SIM":
                        apply_user_move(a)
                return cb
            buttons.append(Button(sx + ci * gx, sy + ri * gy, btn_w, btn_h, label, make_cb(), "normal", FONT))

    input_scramble = InputBox(ctrl_panel_x + (ctrl_panel_w - 90) // 2, ctrl_panel_y + 18, 90, 32, "5", "Scramble Length")
    speed_slider = Slider(ctrl_panel_x + 14, ctrl_panel_y + 66, ctrl_panel_w - 28, 50, 1000, 400, "Speed (ms)")

    by1 = ctrl_panel_y + 130
    bw = (ctrl_panel_w - 30) // 2
    row_h = 38
    row_gap = 8

    # SIM controls
    btn_scramble = Button(ctrl_panel_x + 10, by1, bw, row_h, "Scramble", do_scramble, "warning", FONT)
    btn_reset = Button(ctrl_panel_x + 20 + bw, by1, bw, row_h, "Reset", do_reset, "normal", FONT)
    btn_solve = Button(ctrl_panel_x + 10, by1 + row_h + row_gap, bw, row_h, "Solve", do_solve, "success", FONT)
    btn_stop = Button(ctrl_panel_x + 20 + bw, by1 + row_h + row_gap, bw, row_h, "Stop", do_stop, "danger", FONT)

    # PLAY controls: Reset -> Scan -> Solve
    play_btn_h = 150
    play_gap = 12
    play_y0 = ctrl_panel_y_c + 50
    btn_reset_play = Button(ctrl_panel_x + 10, play_y0, ctrl_panel_w - 20, play_btn_h, "Reset", do_reset_play, "normal", FONT)
    btn_scan = Button(ctrl_panel_x + 10, play_y0 + play_btn_h + play_gap, ctrl_panel_w - 20, play_btn_h, "Scan", start_auto_scan, "warning", FONT)
    btn_solve_play = Button(ctrl_panel_x + 10, play_y0 + 2 * (play_btn_h + play_gap), ctrl_panel_w - 20, play_btn_h, "Solve", do_solve, "success", FONT)

    btn_toggle = Button(WIDTH - 168, 8, 150, 40, "SIM", toggle_mode, "active", FONT)

    buttons += [btn_scramble, btn_reset, btn_solve, btn_stop, btn_toggle, btn_scan, btn_reset_play, btn_solve_play]

    running = True
    try:
        while running:
            CLOCK.tick(60)
            now = pygame.time.get_ticks()

            if camera_stopped_for_solve and not is_thinking:
                #init_camera()
                camera_stopped_for_solve = False

            if mode == "PLAY" and not camera_stopped_for_solve:
                camera_frame = get_camera_frame()
                if camera_frame is not None and not is_thinking:
                    if not is_cube_present(camera_frame):
                        do_reset_play()
                
            elif camera_stopped_for_solve:
                camera_frame = None

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                btn_toggle.text = mode
                for i, btn in enumerate(buttons):
                    if btn == btn_toggle:
                        btn.enabled = not is_thinking

                    elif 0 <= i < 18:
                        btn.visible = (mode == "SIM")
                        btn.enabled = not is_thinking

                    elif btn in [btn_scramble, btn_reset, btn_solve, btn_stop]:
                        btn.visible = (mode == "SIM")
                        btn.enabled = not is_thinking

                    elif btn in [btn_reset_play, btn_scan, btn_solve_play]:
                        btn.visible = (mode == "PLAY")
                        if btn == btn_scan:
                            btn.enabled = (not is_thinking) and (current_scan_idx < 6) and (not scan_auto_running)
                            btn.text = "Scanning..." if scan_auto_running else "Scan"
                        else:
                            btn.enabled = not is_thinking

                    btn.handle_event(event)

                if event.type == pygame.KEYDOWN and not is_thinking:
                    if mode == "SIM":
                        mods = pygame.key.get_mods()
                        shift, ctrl = mods & pygame.KMOD_SHIFT, mods & pygame.KMOD_CTRL
                        k2b = {pygame.K_u: 0, pygame.K_r: 3, pygame.K_f: 6, pygame.K_l: 9, pygame.K_b: 12, pygame.K_d: 15}
                        if event.key in k2b:
                            base = k2b[event.key]
                            if ctrl:
                                apply_user_move(base + 2)
                            elif shift:
                                apply_user_move(base + 1)
                            else:
                                apply_user_move(base)
                        elif event.key == pygame.K_SPACE and key_allowed("space_sim"):
                            do_scramble()
                        elif event.key == pygame.K_ESCAPE and key_allowed("esc_sim"):
                            do_reset()
                        elif event.key == pygame.K_s and key_allowed("s_sim"):
                            do_solve()
                        elif event.key == pygame.K_x and key_allowed("x_sim"):
                            do_stop()
                    else:
                        if event.key == pygame.K_SPACE and key_allowed("space_play"):
                            start_auto_scan()
                        elif event.key == pygame.K_s and key_allowed("s_play"):
                            do_solve()
                        elif event.key == pygame.K_ESCAPE and key_allowed("esc_play"):
                            do_reset_play()

                    if event.key == pygame.K_m and key_allowed("m_mode"):
                        toggle_mode()
                    elif event.key == pygame.K_q and key_allowed("q_quit"):
                        running = False

                if mode == "SIM":
                    input_scramble.handle_event(event)
                    speed_slider.handle_event(event)

            # Auto scan scheduler
            if mode == "PLAY" and scan_auto_running and current_scan_idx < 6 and not is_thinking:
                if now >= next_auto_scan_time:
                    prev_idx = current_scan_idx
                    scan_current_face()
                    if current_scan_idx > prev_idx and current_scan_idx < 6:
                        next_auto_scan_time = now + SCAN_DELAYS_MS[current_scan_idx - 1]

            # SIM animation only
            if mode == "SIM" and play_solve and solve_moves and now >= next_move_time:
                act = solve_moves.pop(0)
                cube = apply_move_idx(cube, act)
                manual_steps += 1
                next_move_time = now + speed_slider.value
            if mode == "SIM" and play_solve and not solve_moves:
                play_solve = False

            if mode == "PLAY" and ser and ser.is_open and not is_thinking:
                if serial_moves_queue and not waiting_for_esp:
                    move = serial_moves_queue[0]
                    if not solve_timer_running and move != "finish":
                        solve_start_time = now
                        solve_timer_running = True
                        final_solve_time = 0.0
                    msg = f"play {move}\n"
                    ser.write(msg.encode("utf-8"))
                    print(f"[UART] Sent: {msg.strip()} (Remaining: {len(serial_moves_queue)})")
                    waiting_for_esp = True
                    wait_esp_start_time = now
                elif waiting_for_esp:
                    if now - wait_esp_start_time > 10000:
                        print("[UART] Execution interrupted (timeout)")
                        ser.write("reset\n".encode("utf-8"))
                        serial_moves_queue.clear()
                        waiting_for_esp = False

                    elif ser.in_waiting > 0:
                        try:
                            response = ser.readline().decode('utf-8').strip()
                            if response == "done":
                                print("[UART] Received: done")
                                serial_moves_queue.pop(0) 
                                waiting_for_esp = False  
                                if solve_timer_running and (not serial_moves_queue or serial_moves_queue[0] == "finish"):
                                    solve_timer_running = False
                                    final_solve_time = (now - solve_start_time) / 1000.0
                            if not serial_moves_queue:
                                hard_reset_state()
                                if ser and ser.is_open:
                                    ser.reset_input_buffer()
                                    ser.write("reset\n".encode("utf-8"))
                        except Exception as e:
                            print(f"[UART] Read error: {e}")
                            
            if lcd and mode == "PLAY" and (now - last_lcd_update_time > 150):
                last_lcd_update_time = now
                remaining = 0
                if serial_moves_queue:
                    if "finish" in serial_moves_queue:
                        remaining = max(0, len(serial_moves_queue) - 1)
                    else:
                        remaining = len(serial_moves_queue)

                if solve_timer_running:
                    current_time = (now - solve_start_time) / 1000.0
                else:
                    current_time = final_solve_time

                if solve_timer_running or remaining > 0 or final_solve_time > 0:
                    try:
                        line1 = f"Remaining: {remaining}".ljust(16)
                        line2 = f"Time: {current_time:.1f}s".ljust(16)

                        lcd.cursor_pos = (0, 0)
                        lcd.write_string(line1[:16])
                        lcd.cursor_pos = (1, 0)
                        lcd.write_string(line2[:16])
                    except Exception as e:
                        print(f"[LCD] Write error: {e}")
                        
            SCREEN.fill(BG_COLOR)

            title = FONT_TITLE.render("Rubik 2x2 Solver", True, TEXT_COLOR)
            SCREEN.blit(title, (WIDTH // 2 - title.get_width() // 2 + 20, 8))
            model_text = "Model Loaded" if model_loaded else "No Model"
            model_color = (0, 230, 118) if model_loaded else (255, 118, 117)
            SCREEN.blit(FONT_SMALL.render(model_text, True, model_color), (14, 14))

            if mode == "SIM":
                draw_panel(SCREEN, move_panel_x, move_panel_y, move_panel_w, move_panel_h, "Moves (18)")
                draw_panel(SCREEN, ctrl_panel_x, ctrl_panel_y, ctrl_panel_w, ctrl_panel_h, "Controls")
                
            draw_panel(SCREEN, ctrl_panel_x_c, ctrl_panel_y_c, ctrl_panel_w_c, ctrl_panel_h_c, "Controls")

            if mode == "SIM":
                input_scramble.draw(SCREEN, panel_x=ctrl_panel_x, panel_w=ctrl_panel_w)
                speed_slider.draw(SCREEN, panel_x=ctrl_panel_x, panel_w=ctrl_panel_w)
            else:
                draw_camera_feed(SCREEN, camera_frame)

            for btn in buttons:
                btn.draw(SCREEN)

            scan_face = SCAN_ORDER[current_scan_idx] if current_scan_idx < 6 else None
            draw_cube(SCREEN, cube, mode, raw_scans if mode == "PLAY" else None)

            scan_prog = f"Scanned: {len(raw_scans)}/6" if mode == "PLAY" else ""
            draw_status_panel(SCREEN, cube, manual_steps, play_solve, is_thinking,solve_formula, move_count, len(solve_moves),current_scramble, mode, scan_face, scan_prog, scan_auto_running, show_play_scan_info)

            pygame.display.flip()

    except Exception as e:
        print(f"[PyGame] Error: {e}")
    finally:
        if global_camera_instance:
            try:
                global_camera_instance.close()
            except Exception:
                pass
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    main()
