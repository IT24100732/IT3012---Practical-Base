# agent.py
import random
import math
from collections import deque
import heapq
from typing import Tuple, List, Set, Dict


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ["Up", "Down", "Left", "Right"]

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept["agent_pos"]
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


class SearchAgent:
    """
    SearchAgent that plans to the nearest food pellet using BFS, DFS, UCS, or A*.
    - self.plan: list of action strings (e.g. ['Up','Left',...'])
    - self.active_algo: one of 'BFS', 'DFS', 'UCS', 'AStar'
    """

    # NOTE: These deltas are written to match how visual_grid_game.py's
    # execute_action() interprets each action (Up -> y+1, Down -> y-1).
    # The original template had Up/Down swapped relative to the environment,
    # which would make any computed plan move the agent in the wrong
    # direction once executed. Fixed here so BFS/DFS/UCS/A* all agree with
    # the environment.
    DELTAS = {
        "Up": (0, 1),
        "Down": (0, -1),
        "Left": (-1, 0),
        "Right": (1, 0),
    }

    def __init__(self):
        self.plan: List[str] = []
        self.active_algo: str = "AStar"  # 'BFS', 'DFS', 'UCS', or 'AStar'
        self.heuristic_type: str = "manhattan"  # 'manhattan' or 'euclidean'

    def sense_and_act(self, percept: dict) -> str:
        # If we already have a plan, execute the next action
        if self.plan:
            return self.plan.pop(0)

        # Extract the necessary global state from the percept dictionary
        start = tuple(percept.get("agent_pos"))
        all_food = [tuple(f) for f in percept.get("all_food", [])]
        remaining_food = percept.get("remaining_food", len(all_food))
        if not all_food or remaining_food == 0:
            return "Stop"  # nothing to do

        # choose the closest food by Manhattan distance as the goal
        def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        goal = min(all_food, key=lambda f: manhattan(start, f))

        walls: Set[Tuple[int, int]] = set(tuple(w) for w in percept.get("walls", []))
        grid_size = percept.get("grid_size")  # optional (width, height) tuple

        if self.active_algo == "BFS":
            plan = self._bfs(start, goal, walls, grid_size)
        elif self.active_algo == "DFS":
            plan = self._dfs(start, goal, walls, grid_size)
        elif self.active_algo == "UCS":
            plan = self._ucs(start, goal, walls, grid_size)
        elif self.active_algo == "AStar":
            plan = self.astar_search(
                start, goal, walls, grid_size, heuristic_type=self.heuristic_type
            )
        else:
            plan = []

        self.plan = plan
        return self.plan.pop(0) if self.plan else "Stop"

    # ------------------------------------------------------------------
    # Step 1.1: Heuristic functions
    # ------------------------------------------------------------------
    def manhattan_distance(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """h(n) = |x1 - x2| + |y1 - y2|"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """h(n) = sqrt((x1 - x2)^2 + (y1 - y2)^2)"""
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    def _in_bounds(self, pos: Tuple[int, int], grid_size) -> bool:
        if grid_size is None:
            return True
        w, h = grid_size
        x, y = pos
        return 0 <= x < w and 0 <= y < h

    def _neighbors(self, pos: Tuple[int, int], walls: Set[Tuple[int, int]], grid_size):
        for action, delta in self.DELTAS.items():
            nx, ny = pos[0] + delta[0], pos[1] + delta[1]
            npos = (nx, ny)
            if npos in walls:
                continue
            if not self._in_bounds(npos, grid_size):
                continue
            yield npos, action

    def _reconstruct(
        self,
        parent: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[str]:
        actions = []
        cur = goal
        while cur != start:
            if cur not in parent:
                return []  # no path
            prev, action = parent[cur]
            actions.append(action)
            cur = prev
        actions.reverse()
        return actions

    def _bfs(self, start, goal, walls, grid_size) -> List[str]:
        q = deque([start])
        visited = {start}
        parent: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]] = {}
        while q:
            cur = q.popleft()
            if cur == goal:
                return self._reconstruct(parent, start, goal)
            for npos, action in self._neighbors(cur, walls, grid_size):
                if npos not in visited:
                    visited.add(npos)
                    parent[npos] = (cur, action)
                    q.append(npos)
        return []

    def _dfs(self, start, goal, walls, grid_size) -> List[str]:
        stack = [start]
        visited = set()
        parent: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]] = {}
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            if cur == goal:
                return self._reconstruct(parent, start, goal)
            # push neighbors in a deterministic order (so behavior is reproducible)
            for npos, action in reversed(list(self._neighbors(cur, walls, grid_size))):
                if npos not in visited:
                    parent[npos] = (cur, action)
                    stack.append(npos)
        return []

    def _ucs(self, start, goal, walls, grid_size) -> List[str]:
        # Uniform-cost search with unit step costs (behaves like BFS on unweighted grid).
        pq = []
        heapq.heappush(pq, (0, start))
        cost_so_far = {start: 0}
        parent: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]] = {}
        while pq:
            cost, cur = heapq.heappop(pq)
            if cur == goal:
                return self._reconstruct(parent, start, goal)
            if cost > cost_so_far.get(cur, float("inf")):
                continue
            for npos, action in self._neighbors(cur, walls, grid_size):
                new_cost = (
                    cost + 1
                )  # all step costs = 1; replace if you have weighted moves
                if new_cost < cost_so_far.get(npos, float("inf")):
                    cost_so_far[npos] = new_cost
                    parent[npos] = (cur, action)
                    heapq.heappush(pq, (new_cost, npos))
        return []

    # ------------------------------------------------------------------
    # Step 1.2: A* Search
    # ------------------------------------------------------------------
    def astar_search(
        self,
        start_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        walls: Set[Tuple[int, int]],
        grid_size,
        heuristic_type: str = "manhattan",
    ) -> List[str]:
        """
        A* search. Evaluates nodes using f(n) = g(n) + h(n), where g(n) is the
        path cost so far and h(n) is the estimated cost to the goal
        (Manhattan or Euclidean distance).
        """
        # Pick the heuristic function to use for this search
        if heuristic_type == "euclidean":
            heuristic_fn = self.euclidean_distance
        else:
            heuristic_fn = self.manhattan_distance

        # Priority queue entries: (f_cost, g_cost, current_pos, path_taken)
        frontier: List[Tuple[float, int, Tuple[int, int], List[str]]] = []
        reached_states: Set[Tuple[int, int]] = set()

        start_h = heuristic_fn(start_pos, goal_pos)
        heapq.heappush(frontier, (start_h, 0, start_pos, []))

        while frontier:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(frontier)

            if current_pos == goal_pos:
                return path_taken

            if current_pos in reached_states:
                continue
            reached_states.add(current_pos)

            for npos, action in self._neighbors(current_pos, walls, grid_size):
                if npos in reached_states:
                    continue

                g_new = g_cost + 1
                h_new = heuristic_fn(npos, goal_pos)
                f_new = g_new + h_new

                heapq.heappush(frontier, (f_new, g_new, npos, path_taken + [action]))

        return []  # no path found


# Observation:
# Change SearchAgent().active_algo between 'BFS', 'DFS', 'UCS', and 'AStar' and run the simulation.
# DFS will often produce winding, erratic paths; BFS, UCS, and A* give direct/optimal paths
# on an unweighted grid, with A* typically expanding far fewer nodes than BFS/UCS
# because the heuristic guides the search towards the goal.


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Step 1.1 Testing Checkpoint
    # ------------------------------------------------------------------
    agent = SearchAgent()
    mock_start = (0, 0)
    mock_goal = (3, 4)

    manhattan_result = agent.manhattan_distance(mock_start, mock_goal)
    euclidean_result = agent.euclidean_distance(mock_start, mock_goal)

    print(f"Manhattan distance from {mock_start} to {mock_goal}: {manhattan_result}")
    print(f"Euclidean distance from {mock_start} to {mock_goal}: {euclidean_result}")
    # Expected: Manhattan -> 7, Euclidean -> 5.0
