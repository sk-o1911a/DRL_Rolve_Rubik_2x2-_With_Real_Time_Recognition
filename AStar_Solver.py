import numpy as np
import heapq
from Rubik2x2Env import (
    apply_move_idx,
    is_solved,
    NUM_ACTIONS,
    MOVE_FUNCS,
    solved_cube
)

class AStarSolver:
    def __init__(self, max_depth=10, heuristic_type="misplaced"):
        self.max_depth = max_depth
        self.heuristic_type = heuristic_type
        self.goal_state = solved_cube()
        
    def _cube_to_bytes(self, cube):
        return cube.tobytes()
    
    def _heuristic(self, cube):
        misplaced = 0
        for f in range(6):
            face_colors = cube[f].flatten()
            counts = np.bincount(face_colors)
            max_count = np.max(counts)
            misplaced += (4 - max_count)
            
        if self.heuristic_type == "manhattan":
            return misplaced / 4.0
        
        return misplaced
    
    def solve(self, initial_cube):
        if is_solved(initial_cube):
            return [], [], True
            
        h_initial = self._heuristic(initial_cube)
        initial_bytes = self._cube_to_bytes(initial_cube)
        
        pq = [(h_initial, 0, initial_bytes, initial_cube, [])]
        visited = {}
        visited[initial_bytes] = 0
        
        while pq:
            f_score, g_score, _, current_cube, path = heapq.heappop(pq)
            
            if len(path) >= self.max_depth:
                continue
            
            cube_bytes = self._cube_to_bytes(current_cube)
            
            if cube_bytes in visited and visited[cube_bytes] < g_score:
                continue
                
            for action in range(NUM_ACTIONS):
                new_cube = apply_move_idx(current_cube, action)
                new_g_score = g_score + 1
                
                if is_solved(new_cube):
                    solution_actions = path + [action]
                    solution_names = [MOVE_FUNCS[a][0] for a in solution_actions]
                    return solution_actions, solution_names, True
                
                new_bytes = self._cube_to_bytes(new_cube)
                
                if new_bytes not in visited or visited[new_bytes] > new_g_score:
                    visited[new_bytes] = new_g_score
                    h_score = self._heuristic(new_cube)
                    new_f_score = new_g_score + h_score
                    heapq.heappush(pq, (new_f_score, new_g_score, new_bytes, new_cube, path + [action]))
        
        return [], [], False
    
    def solve_with_stats(self, initial_cube):
        if is_solved(initial_cube):
            return [], [], True, 0, 1
            
        h_initial = self._heuristic(initial_cube)
        initial_bytes = self._cube_to_bytes(initial_cube)
        
        pq = [(h_initial, 0, initial_bytes, initial_cube, [])]
        visited = {}
        visited[initial_bytes] = 0
        nodes_explored = 0
        
        while pq:
            f_score, g_score, _, current_cube, path = heapq.heappop(pq)
            nodes_explored += 1
            
            if len(path) >= self.max_depth:
                continue
            
            cube_bytes = self._cube_to_bytes(current_cube)
            
            if cube_bytes in visited and visited[cube_bytes] < g_score:
                continue
                
            for action in range(NUM_ACTIONS):
                new_cube = apply_move_idx(current_cube, action)
                new_g_score = g_score + 1
                
                if is_solved(new_cube):
                    solution_actions = path + [action]
                    solution_names = [MOVE_FUNCS[a][0] for a in solution_actions]
                    return solution_actions, solution_names, True, nodes_explored, len(visited)
                
                new_bytes = self._cube_to_bytes(new_cube)
                
                if new_bytes not in visited or visited[new_bytes] > new_g_score:
                    visited[new_bytes] = new_g_score
                    h_score = self._heuristic(new_cube)
                    new_f_score = new_g_score + h_score
                    heapq.heappush(pq, (new_f_score, new_g_score, new_bytes, new_cube, path + [action]))
        
        return [], [], False, nodes_explored, len(visited)
