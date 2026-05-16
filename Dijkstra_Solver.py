import numpy as np
import heapq
from Rubik2x2Env import (
    apply_move_idx,
    is_solved,
    NUM_ACTIONS,
    MOVE_FUNCS
)

class DijkstraSolver:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        
    def _cube_to_bytes(self, cube):
        return cube.tobytes()
    
    def _move_cost(self, action):
        return 1
    
    def solve(self, initial_cube):
        if is_solved(initial_cube):
            return [], [], True
            
        pq = [(0, self._cube_to_bytes(initial_cube), initial_cube, [])]
        visited = {}
        visited[self._cube_to_bytes(initial_cube)] = 0
        
        while pq:
            cost, _, current_cube, path = heapq.heappop(pq)
            
            if len(path) >= self.max_depth:
                continue
            
            cube_bytes = self._cube_to_bytes(current_cube)
            if cube_bytes in visited and visited[cube_bytes] < cost:
                continue
                
            for action in range(NUM_ACTIONS):
                new_cube = apply_move_idx(current_cube, action)
                new_cost = cost + self._move_cost(action)
                
                if is_solved(new_cube):
                    solution_actions = path + [action]
                    solution_names = [MOVE_FUNCS[a][0] for a in solution_actions]
                    return solution_actions, solution_names, True
                
                new_bytes = self._cube_to_bytes(new_cube)
                
                if new_bytes not in visited or visited[new_bytes] > new_cost:
                    visited[new_bytes] = new_cost
                    heapq.heappush(pq, (new_cost, new_bytes, new_cube, path + [action]))
        
        return [], [], False
    
    def solve_with_stats(self, initial_cube):
        if is_solved(initial_cube):
            return [], [], True, 0, 1
            
        pq = [(0, self._cube_to_bytes(initial_cube), initial_cube, [])]
        visited = {}
        visited[self._cube_to_bytes(initial_cube)] = 0
        nodes_explored = 0
        
        while pq:
            cost, _, current_cube, path = heapq.heappop(pq)
            nodes_explored += 1
            
            if len(path) >= self.max_depth:
                continue
            
            cube_bytes = self._cube_to_bytes(current_cube)
            if cube_bytes in visited and visited[cube_bytes] < cost:
                continue
                
            for action in range(NUM_ACTIONS):
                new_cube = apply_move_idx(current_cube, action)
                new_cost = cost + self._move_cost(action)
                
                if is_solved(new_cube):
                    solution_actions = path + [action]
                    solution_names = [MOVE_FUNCS[a][0] for a in solution_actions]
                    return solution_actions, solution_names, True, nodes_explored, len(visited)
                
                new_bytes = self._cube_to_bytes(new_cube)
                
                if new_bytes not in visited or visited[new_bytes] > new_cost:
                    visited[new_bytes] = new_cost
                    heapq.heappush(pq, (new_cost, new_bytes, new_cube, path + [action]))
        
        return [], [], False, nodes_explored, len(visited)
