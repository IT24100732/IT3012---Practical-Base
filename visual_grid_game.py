# visual_grid_game.py
import random
import tkinter as tk
from collections import deque

from agent import SearchAgent


class VisualGridHuntGame:
    """A flexible Pacman-style grid environment with support for configurable opponents and larger scales."""

    def __init__(
        self,
        width=10,
        height=10,
        num_food=10,
        num_opponents=2,
        custom_walls=None,
        num_traps=5,
    ):
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]  # Starting position (x, y)

        if custom_walls is not None:
            self.walls = set(custom_walls)
        else:
            # Generate some default scattered walls for a larger grid
            self.walls = {(2, 2), (2, 3), (5, 5), (6, 5), (3, 7)}

        # Dynamically generate random food positions avoiding walls and agent start
        self.food_positions = set()
        while len(self.food_positions) < num_food:
            fx = random.randint(0, self.width - 1)
            fy = random.randint(0, self.height - 1)
            pos_tuple = (fx, fy)
            if pos_tuple != (0, 0) and pos_tuple not in self.walls:
                self.food_positions.add(pos_tuple)

        # Dynamically generate toxic traps avoiding start, walls, and food
        self.toxic_traps = set()
        while len(self.toxic_traps) < num_traps:
            tx = random.randint(0, self.width - 1)
            ty = random.randint(0, self.height - 1)
            pos_tuple = (tx, ty)
            if (
                pos_tuple != (0, 0)
                and pos_tuple not in self.walls
                and pos_tuple not in self.food_positions
            ):
                self.toxic_traps.add(pos_tuple)

        # Generate adversarial opponents avoiding start, walls, and food
        self.opponents = []
        while len(self.opponents) < num_opponents:
            ox = random.randint(0, self.width - 1)
            oy = random.randint(0, self.height - 1)
            op_pos = [ox, oy]
            if (
                tuple(op_pos) != (0, 0)
                and tuple(op_pos) not in self.walls
                and tuple(op_pos) not in self.food_positions
            ):
                self.opponents.append(op_pos)

        self.score = 0
        self.steps = 0
        self.collision = False

    def get_percept(self) -> dict:
        return {
            "agent_pos": list(self.agent_pos),
            "opponent_positions": [list(op) for op in self.opponents],
            "smells_food": tuple(self.agent_pos) in self.food_positions,
            "hit_wall": tuple(self.agent_pos) in self.walls,
            "smells_toxin": tuple(self.agent_pos) in self.toxic_traps,
            "collision": self.collision,
            "score": self.score,
            "remaining_food": len(self.food_positions),
            "grid_size": (self.width, self.height),
            "walls": list(self.walls),
            "all_food": list(self.food_positions),
        }

    def bfs_pathfind(self, start, goal):
        """BFS to find shortest path avoiding walls."""
        queue = deque([(start, [])])
        visited = {tuple(start)}

        while queue:
            current, path = queue.popleft()

            if current == goal:
                return path

            # Check all 4 directions
            for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
                new_x = current[0] + dx
                new_y = current[1] + dy

                # Check bounds and walls
                if (
                    0 <= new_x < self.width
                    and 0 <= new_y < self.height
                    and (new_x, new_y) not in self.walls
                    and (new_x, new_y) not in visited
                ):

                    visited.add((new_x, new_y))
                    new_path = path + [(new_x, new_y)]
                    queue.append(((new_x, new_y), new_path))

        return []  # No path found

    def find_next_action(self):
        """Find next action towards nearest reachable food."""
        if not self.food_positions:
            return random.choice(["Up", "Down", "Left", "Right"])

        current_pos = tuple(self.agent_pos)
        best_path = None
        best_food = None

        # Find nearest food with valid path
        for food in self.food_positions:
            path = self.bfs_pathfind(current_pos, food)
            if path and (best_path is None or len(path) < len(best_path)):
                best_path = path
                best_food = food

        if not best_path:
            return random.choice(["Up", "Down", "Left", "Right"])

        # Get next position in path
        next_pos = best_path[0]
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]

        if dx > 0:
            return "Right"
        elif dx < 0:
            return "Left"
        elif dy > 0:
            return "Up"
        else:
            return "Down"

    def execute_action(self, action: str):
        self.steps += 1
        new_pos = list(self.agent_pos)

        if action == "Up":
            new_pos[1] = min(self.height - 1, new_pos[1] + 1)
        elif action == "Down":
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == "Left":
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == "Right":
            new_pos[0] = min(self.width - 1, new_pos[0] + 1)

        # Check collision with walls
        if tuple(new_pos) in self.walls:
            self.score -= 5
        else:
            self.agent_pos = new_pos

        tuple_pos = tuple(self.agent_pos)

        # Check if eating food
        if tuple_pos in self.food_positions:
            self.food_positions.remove(tuple_pos)
            self.score += 20

        # Check if stepped on toxic trap
        if tuple_pos in self.toxic_traps:
            self.score -= 15

        # Move opponents randomly
        for op in self.opponents:
            move = random.choice(["Up", "Down", "Left", "Right", "Stay"])
            if move == "Up" and op[1] < self.height - 1:
                op[1] += 1
            elif move == "Down" and op[1] > 0:
                op[1] -= 1
            elif move == "Left" and op[0] > 0:
                op[0] -= 1
            elif move == "Right" and op[0] < self.width - 1:
                op[0] += 1

            if op == self.agent_pos:
                self.score -= 50
                self.collision = True

    def is_done(self) -> bool:
        return len(self.food_positions) == 0 or self.steps >= 300 or self.collision


class GridGameGUI:
    """Modern neon-themed Tkinter UI for the grid environment."""

    def __init__(
        self, root, width=10, height=10, num_food=12, num_opponents=2, walls=None
    ):
        self.root = root
        self.root.title("IT3012 - Scalable Multi-Agent Grid Hunt")
        self.root.configure(bg="#0b1020")

        self.env = VisualGridHuntGame(
            width=width,
            height=height,
            num_food=num_food,
            num_opponents=num_opponents,
            custom_walls=walls,
        )

        # Step 1.3: Inject the SearchAgent (running A*) instead of a
        # ModelBasedAgent / the environment's built-in BFS pathfinder.
        self.agent = SearchAgent()
        self.agent.active_algo = "AStar"  # 'BFS', 'DFS', 'UCS', or 'AStar'
        self.agent.heuristic_type = "manhattan"  # 'manhattan' or 'euclidean'

        # Dynamically calculate cell sizes
        max_canvas_dim = 600
        self.cell_size = max(
            20, min(max_canvas_dim // self.env.width, max_canvas_dim // self.env.height)
        )

        canvas_w = self.env.width * self.cell_size
        canvas_h = self.env.height * self.cell_size

        # ---------------------------------------------------------
        # TOP TITLE
        # ---------------------------------------------------------
        self.title_label = tk.Label(
            root,
            text="\u2726  GRID HUNT  \u2726",
            font=("Arial", 20, "bold"),
            fg="#67e8f9",
            bg="#0b1020",
        )
        self.title_label.pack(pady=(12, 2))

        self.subtitle_label = tk.Label(
            root,
            text="SCALABLE MULTI-AGENT SIMULATION",
            font=("Arial", 9, "bold"),
            fg="#64748b",
            bg="#0b1020",
        )
        self.subtitle_label.pack(pady=(0, 10))

        # ---------------------------------------------------------
        # GAME CANVAS
        # ---------------------------------------------------------
        self.canvas = tk.Canvas(
            root,
            width=canvas_w,
            height=canvas_h,
            bg="#111827",
            highlightthickness=2,
            highlightbackground="#1e3a5f",
        )
        self.canvas.pack(padx=15)

        # ---------------------------------------------------------
        # STATUS PANEL
        # ---------------------------------------------------------
        self.status_frame = tk.Frame(
            root, bg="#111827", highlightbackground="#1e3a5f", highlightthickness=1
        )
        self.status_frame.pack(fill="x", padx=15, pady=(10, 5))

        self.label = tk.Label(
            self.status_frame,
            text="SCORE  0     \u2022     STEPS  0     \u2022     FOOD  15",
            font=("Arial", 11, "bold"),
            fg="#e2e8f0",
            bg="#111827",
            pady=8,
        )
        self.label.pack()

        # ---------------------------------------------------------
        # START BUTTON
        # ---------------------------------------------------------
        self.btn = tk.Button(
            root,
            text="\u25b6  START SIMULATION",
            command=self.run_loop,
            font=("Arial", 11, "bold"),
            bg="#312e81",
            fg="#ffffff",
            activebackground="#4338ca",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=25,
            pady=9,
            cursor="hand2",
        )
        self.btn.pack(pady=(7, 14))

        # Hover effect
        self.btn.bind("<Enter>", self.button_hover)
        self.btn.bind("<Leave>", self.button_leave)

        self.draw_grid()

    def button_hover(self, event):
        self.btn.config(bg="#4f46e5")

    def button_leave(self, event):
        self.btn.config(bg="#312e81")

    def draw_grid(self):
        self.canvas.delete("all")

        # ---------------------------------------------------------
        # DRAW GRID
        # ---------------------------------------------------------
        for x in range(self.env.width):
            for y in range(self.env.height):

                x1 = x * self.cell_size
                y1 = (self.env.height - 1 - y) * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                # Modern dark grid
                if (x, y) in self.env.walls:
                    color = "#334155"
                    outline = "#475569"
                else:
                    # Alternating subtle grid background
                    if (x + y) % 2 == 0:
                        color = "#111827"
                    else:
                        color = "#0f172a"

                    outline = "#1e293b"

                self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=color, outline=outline
                )

                # Wall design
                if (x, y) in self.env.walls:

                    # Inner metallic panel
                    margin = self.cell_size * 0.10

                    self.canvas.create_rectangle(
                        x1 + margin,
                        y1 + margin,
                        x2 - margin,
                        y2 - margin,
                        fill="#475569",
                        outline="#64748b",
                        width=1,
                    )

                    # Small futuristic highlight
                    self.canvas.create_line(
                        x1 + margin,
                        y1 + margin,
                        x2 - margin,
                        y1 + margin,
                        fill="#94a3b8",
                        width=2,
                    )

                    if self.cell_size >= 40:
                        self.canvas.create_text(
                            x1 + self.cell_size / 2,
                            y1 + self.cell_size / 2,
                            text="\u25c6",
                            fill="#cbd5e1",
                            font=("Arial", 10, "bold"),
                        )

        # ---------------------------------------------------------
        # DRAW FOOD
        # ---------------------------------------------------------
        for fx, fy in self.env.food_positions:

            cx = fx * self.cell_size + self.cell_size / 2
            cy = (self.env.height - 1 - fy) * self.cell_size + self.cell_size / 2

            radius = self.cell_size * 0.19

            # Outer glow ring
            self.canvas.create_oval(
                cx - radius * 1.55,
                cy - radius * 1.55,
                cx + radius * 1.55,
                cy + radius * 1.55,
                fill="#3f2a00",
                outline="",
            )

            # Main energy orb
            self.canvas.create_oval(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                fill="#f59e0b",
                outline="#fbbf24",
                width=2,
            )

            # Bright center
            self.canvas.create_oval(
                cx - radius * 0.35,
                cy - radius * 0.35,
                cx + radius * 0.35,
                cy + radius * 0.35,
                fill="#fde68a",
                outline="",
            )

        # ---------------------------------------------------------
        # DRAW TOXIC TRAPS
        # ---------------------------------------------------------
        for tx, ty in self.env.toxic_traps:

            cx = tx * self.cell_size + self.cell_size / 2
            cy = (self.env.height - 1 - ty) * self.cell_size + self.cell_size / 2

            size = self.cell_size * 0.27

            # Outer warning diamond
            self.canvas.create_polygon(
                cx,
                cy - size * 1.2,
                cx + size * 1.2,
                cy,
                cx,
                cy + size * 1.2,
                cx - size * 1.2,
                cy,
                fill="#32105c",
                outline="",
            )

            # Inner diamond
            self.canvas.create_polygon(
                cx,
                cy - size,
                cx + size,
                cy,
                cx,
                cy + size,
                cx - size,
                cy,
                fill="#a855f7",
                outline="#c084fc",
                width=2,
            )

            # Toxic symbol
            if self.cell_size >= 40:
                self.canvas.create_text(
                    cx, cy, text="!", fill="#ffffff", font=("Arial", 11, "bold")
                )

        # ---------------------------------------------------------
        # DRAW OPPONENTS
        # ---------------------------------------------------------
        for ox, oy in self.env.opponents:

            cx = ox * self.cell_size + self.cell_size / 2
            cy = (self.env.height - 1 - oy) * self.cell_size + self.cell_size / 2

            size = self.cell_size * 0.25

            # Enemy outer ring
            self.canvas.create_oval(
                cx - size * 1.45,
                cy - size * 1.45,
                cx + size * 1.45,
                cy + size * 1.45,
                fill="#450a0a",
                outline="",
            )

            # Enemy body
            self.canvas.create_polygon(
                cx,
                cy - size,
                cx + size,
                cy,
                cx,
                cy + size,
                cx - size,
                cy,
                fill="#dc2626",
                outline="#f87171",
                width=2,
            )

            # Enemy core
            self.canvas.create_oval(
                cx - size * 0.28,
                cy - size * 0.28,
                cx + size * 0.28,
                cy + size * 0.28,
                fill="#fecaca",
                outline="",
            )

        # ---------------------------------------------------------
        # DRAW AGENT
        # ---------------------------------------------------------
        ax, ay = self.env.agent_pos

        cx = ax * self.cell_size + self.cell_size / 2
        cy = (self.env.height - 1 - ay) * self.cell_size + self.cell_size / 2

        radius = self.cell_size * 0.30

        # Large outer glow
        self.canvas.create_oval(
            cx - radius * 1.45,
            cy - radius * 1.45,
            cx + radius * 1.45,
            cy + radius * 1.45,
            fill="#172554",
            outline="",
        )

        # Energy ring
        self.canvas.create_oval(
            cx - radius * 1.15,
            cy - radius * 1.15,
            cx + radius * 1.15,
            cy + radius * 1.15,
            fill="#1d4ed8",
            outline="#60a5fa",
            width=2,
        )

        # Main agent
        self.canvas.create_oval(
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
            fill="#2563eb",
            outline="#93c5fd",
            width=2,
        )

        # Agent core
        self.canvas.create_oval(
            cx - radius * 0.42,
            cy - radius * 0.42,
            cx + radius * 0.42,
            cy + radius * 0.42,
            fill="#dbeafe",
            outline="",
        )

        # Small highlight
        self.canvas.create_oval(
            cx - radius * 0.22,
            cy - radius * 0.30,
            cx - radius * 0.02,
            cy - radius * 0.10,
            fill="#ffffff",
            outline="",
        )

    def run_loop(self):
        self.btn.config(state="disabled")

        def step():
            if not self.env.is_done():

                percept = self.env.get_percept()
                action = self.agent.sense_and_act(percept)
                self.env.execute_action(action)

                self.draw_grid()

                self.label.config(
                    text=(
                        f"SCORE  {self.env.score}     \u2022     "
                        f"STEPS  {self.env.steps}     \u2022     "
                        f"FOOD  {len(self.env.food_positions)}"
                    )
                )

                self.root.after(250, step)

            else:

                if self.env.collision:
                    end_text = (
                        f"\u26a0 COLLISION! GAME OVER     \u2022     "
                        f"FINAL SCORE  {self.env.score}"
                    )
                else:
                    end_text = (
                        f"\u2713 MISSION COMPLETE     \u2022     "
                        f"FINAL SCORE  {self.env.score}"
                    )

                self.label.config(text=end_text)
                self.btn.config(state="normal")

        step()


if __name__ == "__main__":
    root = tk.Tk()

    app = GridGameGUI(root, width=12, height=12, num_food=15, num_opponents=0)

    root.mainloop()
