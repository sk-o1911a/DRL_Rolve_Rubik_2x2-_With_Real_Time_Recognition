import numpy as np
from collections import deque
from Rubik2x2Env import (
    Rubik2x2Env,
    apply_move_idx,
    is_solved,
    NUM_ACTIONS,
    MOVE_FUNCS
)

class BFSSolver:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        
    def _cube_to_bytes(self, cube):
        return cube.tobytes()
    
    def solve(self, initial_cube):
        if is_solved(initial_cube):
            return [], [], True
            
        queue = deque([(initial_cube, [])])
        visited = {self._cube_to_bytes(initial_cube)}
        
        while queue:
            current_cube, path = queue.popleft()
            
            if len(path) >= self.max_depth:
                continue
                
            for action in range(NUM_ACTIONS):
                new_cube = apply_move_idx(current_cube, action)
                
                if is_solved(new_cube):
                    solution_actions = path + [action]
                    solution_names = [MOVE_FUNCS[a][0] for a in solution_actions]
                    return solution_actions, solution_names, True
                
                cube_bytes = self._cube_to_bytes(new_cube)
                if cube_bytes not in visited:
                    visited.add(cube_bytes)
                    queue.append((new_cube, path + [action]))
        
        return [], [], False
    
    def solve_with_stats(self, initial_cube):
        if is_solved(initial_cube):
            return [], [], True, 0, 1
            
        queue = deque([(initial_cube, [])])
        visited = {self._cube_to_bytes(initial_cube)}
        nodes_explored = 0
        
        while queue:
            current_cube, path = queue.popleft()
            nodes_explored += 1
            
            if len(path) >= self.max_depth:
                continue
                
            for action in range(NUM_ACTIONS):
                new_cube = apply_move_idx(current_cube, action)
                
                if is_solved(new_cube):
                    solution_actions = path + [action]
                    solution_names = [MOVE_FUNCS[a][0] for a in solution_actions]
                    return solution_actions, solution_names, True, nodes_explored, len(visited)
                
                cube_bytes = self._cube_to_bytes(new_cube)
                if cube_bytes not in visited:
                    visited.add(cube_bytes)
                    queue.append((new_cube, path + [action]))
        
        return [], [], False, nodes_explored, len(visited)
