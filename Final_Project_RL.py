import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import osmnx as ox
import threading
import time
import subprocess
import collections # For BFS to check path existence
import sys # For sys.exit to properly close matplotlib pop-ups if needed

# --- Ensure required libraries are installed for OSMnx ---
def check_and_install_osmnx():
    required_packages = ['osmnx', 'matplotlib', 'networkx', 'geopandas', 'descartes']
    missing_packages = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing_packages.append(pkg)

    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}. Attempting to install...")
        try:
            # Use subprocess to run pip in a new process
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("Required packages installed successfully.")
            messagebox.showinfo("Installation Complete", "Required dependencies have been installed. You might need to restart the application for changes to full effect.")
        except Exception as e:
            print(f"Error installing packages: {e}")
            messagebox.showerror("Installation Error", f"Failed to install required packages. Please try running 'pip install {' '.join(required_packages)}' manually in your terminal.\nError: {e}")
    else:
        print("All required packages are already installed.")

# Run the check/install when the script starts
check_and_install_osmnx()


# --- Utility Functions (from original code, kept as-is) ---
NO_OF_CHARS = 256

def str_to_list(string):
    temp = []
    for x in string:
        temp.append(x)
    return temp

def lst_to_string(List):
    return ''.join(List)

def get_char_count_array(string):
    count = [0] * NO_OF_CHARS
    for i in string:
        count[ord(i)] += 1
    return count

def remove_dirty_chars(string, second_string):
    count = get_char_count_array(second_string)
    ip_ind = 0
    res_ind = 0
    str_list = str_to_list(string)
    while ip_ind != len(str_list):
        temp = str_list[ip_ind]
        if count[ord(temp)] == 0:
            str_list[res_ind] = str_list[ip_ind]
            res_ind += 1
        ip_ind+=1
    return lst_to_string(str_list[0:res_ind])

# Functionality to get current working directory
import os
def get_working_directory():
    return os.getcwd()

# Functionality to simulate file upload (for utilities tab)
def simulate_file_upload():
    messagebox.showinfo("File Upload", "Simulating file upload...\nIn a real Tkinter app, this would open a file dialog to select a file.")
    return {'dummy_file.txt': b'This is dummy file content.'}

# --- Q-Learning Core Functions (adapted for dynamic grid) ---
# Actions (8 directions)
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
NUM_ACTIONS = len(ACTIONS)

# Rewards (can be made configurable later if desired)
GOAL_REWARD = 100
OBSTACLE_REWARD = -10
DEFAULT_REWARD = -1

def is_valid_state(state, grid_size, obstacles):
    row, col = state
    return 0 <= row < grid_size and 0 <= col < grid_size and state not in obstacles

def get_next_state(current_state, action_index, grid_size, obstacles):
    row, col = current_state
    dr, dc = ACTIONS[action_index]
    next_row, next_col = row + dr, col + dc
    next_state = (next_row, next_col)

    if is_valid_state(next_state, grid_size, obstacles):
        return next_state
    else:
        return current_state # Stay in the current state if the move is invalid (hit boundary or obstacle)

def get_reward(state, end_state, obstacles):
    if state == end_state:
        return GOAL_REWARD
    elif state in obstacles:
        return OBSTACLE_REWARD
    else:
        return DEFAULT_REWARD

def train_q_learning(grid_size, start_state, end_state, obstacles, total_episodes, learning_rate, discount_factor, epsilon_decay_rate, min_epsilon, progress_callback=None):
    """Runs the Q-Learning training process."""
    if not grid_size or start_state is None or end_state is None or obstacles is None:
        print("Cannot train: Invalid grid parameters.")
        return None, []

    q_table = np.zeros((grid_size, grid_size, NUM_ACTIONS))
    epsilon = 1.0 # Epsilon starts high for exploration
    episode_rewards = []

    print(f"Starting Q-Learning training on {grid_size}x{grid_size} grid...")

    # Check if start or end state is an obstacle
    if start_state in obstacles or end_state in obstacles:
        print("Start or End state is an obstacle. Training aborted.")
        messagebox.showerror("Training Error", "Start or End position cannot be an obstacle. Please regenerate grid or select different map start/end points.")
        return None, []

    # Check if a path exists using BFS before training
    if not has_path(grid_size, start_state, end_state, obstacles):
        print("No path exists between start and end with given obstacles. Training aborted.")
        messagebox.showerror("Training Error", "No possible path exists between start and end with current obstacles. Please regenerate grid or reduce obstacles for Original Grid, or try a different location/grid size for Real-World Map.")
        return None, []

    for episode in range(total_episodes):
        current_state = start_state
        done = False
        total_episode_reward = 0
        step_count = 0
        max_steps_per_episode = grid_size * grid_size * 4 # Prevent infinite loops in difficult grids

        while not done and step_count < max_steps_per_episode:
            row, col = current_state
            if random.uniform(0, 1) < epsilon:
                action_index = random.randint(0, NUM_ACTIONS - 1)
            else:
                q_values = q_table[row, col]
                # Handle cases where all Q-values are the same (e.g., all zeros initially)
                max_q = np.max(q_values)
                best_actions = np.where(q_values == max_q)[0]
                action_index = random.choice(best_actions)

            next_state = get_next_state(current_state, action_index, grid_size, obstacles)
            reward = get_reward(next_state, end_state, obstacles)
            total_episode_reward += reward

            next_row, next_col = next_state
            old_value = q_table[row, col, action_index]
            next_max_value = np.max(q_table[next_row, next_col])
            new_value = old_value + learning_rate * (reward + discount_factor * next_max_value - old_value)
            q_table[row, col, action_index] = new_value

            current_state = next_state

            if current_state == end_state:
                done = True
            step_count += 1

        epsilon = max(min_epsilon, epsilon * epsilon_decay_rate)
        episode_rewards.append(total_episode_reward)

        if progress_callback:
            progress_callback(episode + 1, total_episodes)

    print("Training finished.")
    return q_table, episode_rewards

def find_optimal_path(q_table, grid_size, start_state, end_state, obstacles):
    """Finds the optimal path using the trained Q-table."""
    if q_table is None or grid_size is None or start_state is None or end_state is None or obstacles is None:
        print("Cannot find path: Invalid input.")
        return []

    print("\nFinding optimal path using the trained Q-table:")
    current_state = start_state
    optimal_path = []
    step_count = 0
    max_path_length = grid_size * grid_size * 2 # Prevent infinite loops in pathfinding

    # Add start state to path
    optimal_path.append(current_state)

    while current_state != end_state and step_count < max_path_length:
        row, col = current_state
        if not (0 <= row < grid_size and 0 <= col < grid_size):
            print(f"Agent went out of bounds at {current_state}. Pathfinding stopped.")
            break # Safety break if state becomes invalid

        # If current_state somehow became an obstacle, break
        if (row, col) in obstacles:
            print(f"Agent stepped into an obstacle at {current_state}. Pathfinding stopped.")
            break

        # Check if all Q-values are identical (e.g., all zeros or very small)
        q_values_for_state = q_table[row, col]
        if np.all(q_values_for_state == q_values_for_state[0]):
            # If all Q-values are the same, pick a random valid action
            valid_actions_indices = []
            for i, action in enumerate(ACTIONS):
                dr, dc = action
                next_row, next_col = row + dr, col + dc
                if 0 <= next_row < grid_size and 0 <= next_col < grid_size and (next_row, next_col) not in obstacles:
                    valid_actions_indices.append(i)
            if valid_actions_indices:
                action_index = random.choice(valid_actions_indices)
            else:
                # Agent is completely blocked from moving anywhere from this state
                print(f"Agent is trapped at {current_state}. Cannot reach goal.")
                break
        else:
            action_index = np.argmax(q_values_for_state)

        next_state = get_next_state(current_state, action_index, grid_size, obstacles)

        # If agent cannot move to a new state and is not at the end state, it's stuck.
        if next_state == current_state and current_state != end_state:
            print(f"Agent stuck at {current_state}. Cannot reach goal.")
            break

        current_state = next_state
        optimal_path.append(current_state)
        step_count += 1

    if current_state == end_state:
        print("Goal reached.")
    else:
        print("Pathfinding terminated without reaching goal (might be stuck or path too long).")

    return optimal_path

def has_path(grid_size, start, end, obstacles):
    """Checks if a path exists from start to end using BFS."""
    queue = collections.deque([start])
    visited = {start}

    while queue:
        r, c = queue.popleft()

        if (r, c) == end:
            return True

        for dr, dc in ACTIONS:
            nr, nc = r + dr, c + dc
            next_state = (nr, nc)

            if 0 <= nr < grid_size and 0 <= nc < grid_size and \
               next_state not in obstacles and next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)
    return False


# --- Real-World Map Integration Functions (adapted) ---

def get_real_world_graph(place_name="Random Location"):
    """Downloads a street network graph."""
    try:
        print(f"Attempting to download graph for: {place_name}")
        if place_name.lower() == "random location":
            # Example coordinates for a small area if random is too broad
            # These are for a small part of San Francisco for reproducibility/smaller download
            lat, lon = 37.7849, -122.4172 # Near Civic Center, San Francisco
            G = ox.graph_from_point((lat, lon), dist=500, network_type='drive')
            print(f"Downloaded graph for a random location near ({lat:.4f}, {lon:.4f})")
        else:
            G = ox.graph_from_place(place_name, network_type='drive')
            print(f"Downloaded graph for {place_name}")
        return G
    except Exception as e:
        print(f"Error downloading graph: {e}")
        messagebox.showerror("Graph Download Error", f"Could not download graph for '{place_name}'. Common issues: incorrect place name, no internet, or OSMnx server issues.\nError: {e}")
        return None

def get_start_end_nodes(graph):
    """Selects random start and end nodes from the graph."""
    if not graph or not graph.nodes():
        print("Graph is empty or invalid.")
        return None, None
    nodes = list(graph.nodes())
    if len(nodes) < 2:
        print("Graph has too few nodes to select start/end.")
        return None, None
    start_node = random.choice(nodes)
    end_node = random.choice(nodes)
    while start_node == end_node:
        end_node = random.choice(nodes)
    print(f"Selected graph start node: {start_node}, End node: {end_node}")
    return start_node, end_node

def graph_to_simplified_grid(graph, start_node_graph, end_node_graph, grid_size=30):
    """
    Converts a portion of the graph into a simplified grid representation for Q-learning.
    Returns: grid_size, start_grid_state, end_grid_state, grid_obstacles, grid_to_node_map,
             node_to_grid_map, min_coords, max_coords
    """
    if not graph or start_node_graph is None or end_node_graph is None:
        print("Graph or start/end nodes invalid for grid conversion.")
        return None, None, None, None, None, None, None, None

    nodes_data = {node: data for node, data in graph.nodes(data=True)}

    if start_node_graph not in nodes_data or end_node_graph not in nodes_data:
        print("Start or end node not found in graph data. This shouldn't happen if selected from graph.")
        messagebox.showerror("Graph Data Error", "Internal error: Start or end node not found in graph data.")
        return None, None, None, None, None, None, None, None

    coords = np.array([(data['x'], data['y']) for node, data in nodes_data.items()])

    if len(coords) == 0:
        print("No node coordinates found in the graph.")
        messagebox.showerror("Graph Error", "No node coordinates found in the graph. Cannot create a grid.")
        return None, None, None, None, None, None, None, None

    min_x, min_y = np.min(coords, axis=0)
    max_x, max_y = np.max(coords, axis=0)

    # Handle cases where graph has no spatial extent (e.g., all nodes at same point)
    # Add a small buffer to ranges to prevent division by zero or very small ranges
    range_x = max_x - min_x
    range_y = max_y - min_y
    if range_x < 1e-6: range_x = 1e-6 # Minimum range to prevent division by zero
    if range_y < 1e-6: range_y = 1e-6

    node_to_grid_map = {}
    grid_to_node_map = {} # Map grid cells back to graph nodes for path reconstruction

    start_grid_state = None
    end_grid_state = None

    def map_coord_to_grid(x, y, min_x_val, min_y_val, max_x_val, max_y_val, grid_dim, x_range, y_range):
        # Normalize coordinates to [0, 1) and then scale to grid_dim
        # Invert y-axis mapping for grid to match visual (row 0 at top, higher latitude)
        row = int(((max_y_val - y) / y_range) * grid_dim)
        col = int(((x - min_x_val) / x_range) * grid_dim)

        # Ensure coordinates stay within grid bounds [0, grid_dim-1]
        row = max(0, min(grid_dim - 1, row))
        col = max(0, min(grid_dim - 1, col))
        return row, col

    for node, data in nodes_data.items():
        row, col = map_coord_to_grid(data['x'], data['y'], min_x, min_y, max_x, max_y, grid_size, range_x, range_y)
        grid_state = (row, col)

        # Associate this grid state with the graph node
        node_to_grid_map[node] = grid_state
        # Map grid cell back to a graph node (use the last one for simplicity if multiple nodes map to same cell)
        # This will be useful for converting grid path back to graph path
        grid_to_node_map.setdefault(grid_state, node) # Only set if not already set, keeping the first node mapped to this cell

    # Identify grid cells that are not mapped to any node as potential obstacles
    all_grid_cells = {(r, c) for r in range(grid_size) for c in range(grid_size)}
    mapped_grid_cells = set(grid_to_node_map.keys())
    grid_obstacles = all_grid_cells - mapped_grid_cells

    # Convert start and end graph nodes to grid states
    start_grid_state = node_to_grid_map.get(start_node_graph)
    end_grid_state = node_to_grid_map.get(end_node_graph)

    # It's possible that the initial mapping of start/end graph nodes landed them on an obstacle
    # or outside the valid grid (due to rounding/edges). Find nearest valid grid cell if so.
    if start_grid_state in grid_obstacles:
        temp_start_grid_state = find_nearest_mapped_cell(start_grid_state, mapped_grid_cells)
        if temp_start_grid_state:
            start_grid_state = temp_start_grid_state
            # Update the corresponding graph node in grid_to_node_map for consistency
            start_node_graph = grid_to_node_map[start_grid_state] # Update the graph node too
        else:
            print("Could not find a valid start grid state after finding nearest. Aborting grid creation.")
            messagebox.showerror("Grid Error", "Could not find a valid start grid state for the map. Try a different location or grid size.")
            return None, None, None, None, None, None, None, None

    if end_grid_state in grid_obstacles:
        temp_end_grid_state = find_nearest_mapped_cell(end_grid_state, mapped_grid_cells)
        if temp_end_grid_state:
            end_grid_state = temp_end_grid_state
            # Update the corresponding graph node in grid_to_node_map for consistency
            end_node_graph = grid_to_node_map[end_grid_state] # Update the graph node too
        else:
            print("Could not find a valid end grid state after finding nearest. Aborting grid creation.")
            messagebox.showerror("Grid Error", "Could not find a valid end grid state for the map. Try a different location or grid size.")
            return None, None, None, None, None, None, None, None

    if start_grid_state == end_grid_state:
        print(f"Start and end nodes (or their nearest mapped cells) mapped to the same grid cell: {start_grid_state}. Aborting grid creation.")
        messagebox.showerror("Grid Error", f"Start and end points map to the same grid cell ({start_grid_state}). Please try a different location or grid size.")
        return None, None, None, None, None, None, None, None

    # Ensure start and end are not in obstacles *after* any adjustments
    grid_obstacles.discard(start_grid_state)
    grid_obstacles.discard(end_grid_state)

    print(f"Simplified grid size: {grid_size}x{grid_size}")
    print(f"Start grid state: {start_grid_state}")
    print(f"End grid state: {end_grid_state}")
    print(f"Number of obstacles in grid: {len(grid_obstacles)}")

    return grid_size, start_grid_state, end_grid_state, list(grid_obstacles), grid_to_node_map, node_to_grid_map, (min_x, min_y), (max_x, max_y), start_node_graph, end_node_graph


def find_nearest_mapped_cell(target_cell, mapped_cells):
    """Finds the nearest cell in mapped_cells to the target_cell."""
    if not mapped_cells:
        return None
    target_row, target_col = target_cell
    min_dist_sq = float('inf')
    nearest_cell = None

    for cell in mapped_cells:
        row, col = cell
        dist_sq = (row - target_row)**2 + (col - target_col)**2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            nearest_cell = cell
    return nearest_cell


class App:
    def __init__(self, root):
        self.root = root
        root.title("Q-Learning Pathfinding Simulator")
        root.geometry("800x700") # Initial size

        self.current_q_table = None
        self.current_grid_params = None # (grid_size, start_state, end_state, obstacles)
        self.current_tab_trained = None # "original" or "real_world"

        # For Real-World Map specific data
        self.current_osm_graph = None
        self.real_world_grid_to_node_map = None
        self.real_world_node_to_grid_map = None
        self.real_world_min_coords = None
        self.real_world_max_coords = None
        self.real_world_graph_start_node = None # Store original graph start node
        self.real_world_graph_end_node = None   # Store original graph end node

        self.animation_thread = None
        self.animation_stop_event = threading.Event()

        self.create_main_menu()
        self.setup_hyperparameters_frame()

    def create_main_menu(self):
        # Clear existing widgets if coming back from a sub-menu
        for widget in self.root.winfo_children():
            if widget != self.hp_frame: # Don't destroy hyperparameter frame
                widget.destroy()

        self.main_menu_frame = ttk.Frame(self.root, padding="20")
        self.main_menu_frame.pack(expand=True, fill="both")

        ttk.Label(self.main_menu_frame, text="Choose a Pathfinding Mode:", font=("Arial", 16)).pack(pady=20)

        ttk.Button(self.main_menu_frame, text="Original Grid Pathfinding",
                   command=self.show_original_grid_interface, width=30, style='TButton').pack(pady=10)
        ttk.Button(self.main_menu_frame, text="Real-World Map Pathfinding",
                   command=self.show_real_world_map_interface, width=30, style='TButton').pack(pady=10)
        ttk.Button(self.main_menu_frame, text="Utilities",
                   command=self.show_utilities_interface, width=30, style='TButton').pack(pady=10)


    def setup_hyperparameters_frame(self):
        # Hyperparameters (always visible at the bottom)
        self.hp_frame = ttk.LabelFrame(self.root, text="Hyperparameters", padding="10")
        self.hp_frame.pack(side="bottom", fill="x", pady=5, padx=10)

        # Learning Rate
        ttk.Label(self.hp_frame, text="Learning Rate:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.lr_var = tk.DoubleVar(value=0.1)
        ttk.Entry(self.hp_frame, textvariable=self.lr_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        # Discount Factor
        ttk.Label(self.hp_frame, text="Discount Factor:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        self.df_var = tk.DoubleVar(value=0.99)
        ttk.Entry(self.hp_frame, textvariable=self.df_var, width=10).grid(row=0, column=3, padx=5, pady=2)

        # Epsilon Decay Rate
        ttk.Label(self.hp_frame, text="Epsilon Decay Rate:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.edr_var = tk.DoubleVar(value=0.9995)
        ttk.Entry(self.hp_frame, textvariable=self.edr_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        # Min Epsilon
        ttk.Label(self.hp_frame, text="Min Epsilon:").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        self.min_eps_var = tk.DoubleVar(value=0.01)
        ttk.Entry(self.hp_frame, textvariable=self.min_eps_var, width=10).grid(row=1, column=3, padx=5, pady=2)

        # Total Episodes
        ttk.Label(self.hp_frame, text="Total Episodes:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.episodes_var = tk.IntVar(value=5000)
        ttk.Entry(self.hp_frame, textvariable=self.episodes_var, width=10).grid(row=2, column=1, padx=5, pady=2)

        # Configure columns to expand nicely
        for i in range(4):
            self.hp_frame.grid_columnconfigure(i, weight=1)

    def show_original_grid_interface(self):
        self.main_menu_frame.pack_forget() # Hide main menu

        self.original_grid_frame = ttk.Frame(self.root, padding="10")
        self.original_grid_frame.pack(expand=True, fill="both")

        # Top control frame
        control_frame = ttk.Frame(self.original_grid_frame)
        control_frame.pack(pady=5, fill="x")
        ttk.Button(control_frame, text="< Back to Main Menu", command=self.back_to_main_menu).pack(side="left", padx=5)

        # Configuration frame for grid
        config_frame = ttk.LabelFrame(control_frame, text="Grid Config", padding="10")
        config_frame.pack(side="left", padx=10, fill="x", expand=True)

        ttk.Label(config_frame, text="Size (4-8):").grid(row=0, column=0, padx=2, pady=1, sticky="w")
        self.grid_size_var = tk.IntVar(value=8)
        ttk.Spinbox(config_frame, from_=4, to=8, textvariable=self.grid_size_var, width=4).grid(row=0, column=1, padx=2, pady=1, sticky="ew")

        ttk.Label(config_frame, text="Obstacles:").grid(row=1, column=0, padx=2, pady=1, sticky="w")
        self.num_obstacles_var = tk.IntVar(value=8)
        ttk.Spinbox(config_frame, from_=0, to=31, textvariable=self.num_obstacles_var, width=4).grid(row=1, column=1, padx=2, pady=1, sticky="ew")

        ttk.Button(config_frame, text="Generate Grid & Train", command=self.run_original_grid_training).grid(row=0, column=2, rowspan=2, padx=5, pady=2)
        config_frame.grid_columnconfigure(1, weight=1)

        # Display buttons for Q-Table and Graph
        display_buttons_frame = ttk.Frame(control_frame)
        display_buttons_frame.pack(side="right", padx=5)
        ttk.Button(display_buttons_frame, text="Show Grid Map", command=self.show_original_grid_map).pack(side="top", pady=2)
        ttk.Button(display_buttons_frame, text="Show Q-Table", command=lambda: self.show_q_table_popup("original")).pack(side="top", pady=2)


        # Progress Bar
        self.progress_label_og = ttk.Label(self.original_grid_frame, text="Training Progress: 0%")
        self.progress_label_og.pack(pady=2)
        self.progress_og = ttk.Progressbar(self.original_grid_frame, orient="horizontal", length=400, mode="determinate")
        self.progress_og.pack(pady=5, fill="x", padx=5)

        # Canvas for grid visualization
        self.grid_canvas = tk.Canvas(self.original_grid_frame, bg="white", borderwidth=2, relief="groove")
        self.grid_canvas.pack(expand=True, fill="both", pady=10)
        self.grid_canvas_size = 500 # Max size for grid visualization
        self.grid_canvas.bind("<Configure>", self.on_grid_canvas_resize)

        # Matplotlib Figure for Plotting (Learning Curve)
        self.fig_og, self.ax_og = plt.subplots(figsize=(6, 2.5), dpi=100) # Smaller figure for learning curve
        self.canvas_og_mpl = FigureCanvasTkAgg(self.fig_og, master=self.original_grid_frame)
        self.canvas_og_mpl_widget = self.canvas_og_mpl.get_tk_widget()
        self.canvas_og_mpl_widget.pack(side="bottom", fill="x", pady=5)
        self.plot_learning_curve([], self.ax_og, self.canvas_og_mpl, "Original Grid") # Initial blank plot

        self.last_drawn_grid_params = None # To track if grid needs redrawing

    def on_grid_canvas_resize(self, event):
        # Redraw the grid if the canvas size changes and we have grid parameters
        if self.last_drawn_grid_params:
            self.draw_grid_on_canvas(self.last_drawn_grid_params[0],
                                     self.last_drawn_grid_params[1],
                                     self.last_drawn_grid_params[2],
                                     self.last_drawn_grid_params[3])

    def show_original_grid_map(self):
        """Shows the current grid configuration on the canvas."""
        if self.current_grid_params:
            grid_size, start_state, end_state, obstacles = self.current_grid_params
            self.draw_grid_on_canvas(grid_size, start_state, end_state, obstacles)
            messagebox.showinfo("Original Grid Map", "Current grid configuration displayed on canvas.")
        else:
            messagebox.showwarning("No Grid", "Please generate a grid first by clicking 'Generate Grid & Train'.")


    def show_real_world_map_interface(self):
        self.main_menu_frame.pack_forget() # Hide main menu

        self.real_world_frame = ttk.Frame(self.root, padding="10")
        self.real_world_frame.pack(expand=True, fill="both")

        # Top control frame
        control_frame_rw = ttk.Frame(self.real_world_frame)
        control_frame_rw.pack(pady=5, fill="x")
        ttk.Button(control_frame_rw, text="< Back to Main Menu", command=self.back_to_main_menu).pack(side="left", padx=5)

        # Place name input
        place_frame = ttk.Frame(control_frame_rw)
        place_frame.pack(side="left", padx=10, fill="x", expand=True)

        ttk.Label(place_frame, text="Place Name (e.g., 'Berkeley, CA' or 'Random Location'):").grid(row=0, column=0, sticky="w", padx=5)
        self.place_name_var = tk.StringVar(value="San Francisco, CA") # Changed default for better testing
        self.place_name_entry = ttk.Entry(place_frame, textvariable=self.place_name_var, width=40)
        self.place_name_entry.grid(row=0, column=1, padx=5, sticky="ew")
        ttk.Button(place_frame, text="Load Map & Train", command=self.run_real_world_training).grid(row=0, column=2, padx=5)
        place_frame.grid_columnconfigure(1, weight=1)

        # Display buttons for Q-Table and Graph
        display_buttons_frame_rw = ttk.Frame(control_frame_rw)
        display_buttons_frame_rw.pack(side="right", padx=5)
        ttk.Button(display_buttons_frame_rw, text="Show OSMnx Graph", command=self.show_osmnx_graph).pack(side="top", pady=2)
        ttk.Button(display_buttons_frame_rw, text="Show Q-Table", command=lambda: self.show_q_table_popup("real_world")).pack(side="top", pady=2)

        # Progress Bar
        self.progress_label_rw = ttk.Label(self.real_world_frame, text="Training Progress: 0%")
        self.progress_label_rw.pack(pady=2)
        self.progress_rw = ttk.Progressbar(self.real_world_frame, orient="horizontal", length=400, mode="determinate")
        self.progress_rw.pack(pady=5, fill="x", padx=5)

        # Matplotlib Figure for Map and Learning Curve
        # We'll use one figure, potentially with subplots, or just the map and a smaller curve.
        self.fig_rw, (self.ax_rw_map, self.ax_rw_curve) = plt.subplots(2, 1, figsize=(30, 25), gridspec_kw={'height_ratios': [3, 1]}, dpi=100)
        self.canvas_rw_mpl = FigureCanvasTkAgg(self.fig_rw, master=self.real_world_frame)
        self.canvas_rw_mpl_widget = self.canvas_rw_mpl.get_tk_widget()
        self.canvas_rw_mpl_widget.pack(expand=True, fill="both", pady=10)

        # Initial blank map and curve
        self.ax_rw_map.set_title("Real-World Map Visualization")
        self.ax_rw_map.set_xticks([])
        self.ax_rw_map.set_yticks([])
        self.plot_learning_curve([], self.ax_rw_curve, self.canvas_rw_mpl, "Real-World Grid") # Initial blank plot

    def show_osmnx_graph(self):
        if not self.current_osm_graph:
            messagebox.showwarning("No Map Loaded", "Please load a real-world map first by clicking 'Load Map & Train'.")
            return

        # Plot the graph in a new window using osmnx's plotting capabilities
        fig, ax = ox.plot_graph(self.current_osm_graph, show=False, close=False,
                                bgcolor='lightgray', edge_color='white', node_size=10,
                                node_color='blue', edge_linewidth=0.5,
                                figsize=(20, 20)) # Fixed size for pop-up

        # Highlight start and end nodes if available
        if self.real_world_graph_start_node and self.real_world_graph_end_node:
            s_node = self.real_world_graph_start_node
            e_node = self.real_world_graph_end_node

            if s_node in self.current_osm_graph.nodes:
                sx, sy = self.current_osm_graph.nodes[s_node]['x'], self.current_osm_graph.nodes[s_node]['y']
                ax.scatter(sx, sy, color='green', s=200, zorder=5, edgecolors='black', label='Start Node')
                ax.text(sx, sy, 'S', fontsize=12, ha='center', va='center', color='white', zorder=6)

            if e_node in self.current_osm_graph.nodes:
                ex, ey = self.current_osm_graph.nodes[e_node]['x'], self.current_osm_graph.nodes[e_node]['y']
                ax.scatter(ex, ey, color='red', s=200, zorder=5, edgecolors='black', label='End Node')
                ax.text(ex, ey, 'E', fontsize=12, ha='center', va='center', color='white', zorder=6)

        ax.set_title(f"OSMnx Graph for {self.place_name_var.get()}")
        plt.show() # Display the matplotlib figure in a new window


    def show_utilities_interface(self):
        self.main_menu_frame.pack_forget() # Hide main menu

        self.utilities_frame = ttk.Frame(self.root, padding="10")
        self.utilities_frame.pack(expand=True, fill="both")

        # Back button
        ttk.Button(self.utilities_frame, text="< Back to Main Menu", command=self.back_to_main_menu).pack(pady=5, anchor="nw")

        # Frame for Utilities
        utilities_content_frame = ttk.Frame(self.utilities_frame, padding="10")
        utilities_content_frame.pack(pady=10, padx=10, expand=True, fill="both")

        # Working Directory Button
        ttk.Button(utilities_content_frame, text="Print Working Directory", command=self.print_working_dir).pack(pady=5)
        self.working_dir_label = ttk.Label(utilities_content_frame, text="")
        self.working_dir_label.pack(pady=2)

        # File Upload Simulation Button
        ttk.Button(utilities_content_frame, text="Simulate File Upload", command=self.handle_file_upload).pack(pady=5)

        # Text entry for string manipulation
        ttk.Label(utilities_content_frame, text="String 1:").pack(pady=2)
        self.string1_entry = ttk.Entry(utilities_content_frame, width=50)
        self.string1_entry.insert(tk.END, "This is the first string.")
        self.string1_entry.pack(pady=2)

        ttk.Label(utilities_content_frame, text="String 2 (chars to remove):").pack(pady=2)
        self.string2_entry = ttk.Entry(utilities_content_frame, width=50)
        self.string2_entry.insert(tk.END, "aeiou")
        self.string2_entry.pack(pady=2)

        ttk.Button(utilities_content_frame, text="Remove Chars", command=self.remove_chars_action).pack(pady=5)

        ttk.Label(utilities_content_frame, text="Result:").pack(pady=2)
        self.remove_chars_result_label = ttk.Label(utilities_content_frame, text="")
        self.remove_chars_result_label.pack(pady=2)


    def show_q_table_popup(self, mode):
        if self.current_q_table is None:
            messagebox.showwarning("No Q-Table", "No Q-Table available. Please run a training simulation first.")
            return

        popup = tk.Toplevel(self.root)
        popup.title(f"{mode.replace('_', ' ').title()} Q-Table")
        popup.geometry("600x400")

        # Create a scrollable text widget
        text_frame = ttk.Frame(popup)
        text_frame.pack(expand=True, fill="both", padx=10, pady=10)

        text_widget = tk.Text(text_frame, wrap="none", font=("Consolas", 10))
        text_widget.pack(side="left", expand=True, fill="both")

        vsb = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        vsb.pack(side="right", fill="y")
        text_widget.configure(yscrollcommand=vsb.set)

        hsb = ttk.Scrollbar(popup, orient="horizontal", command=text_widget.xview)
        hsb.pack(side="bottom", fill="x")
        text_widget.configure(xscrollcommand=hsb.set)

        # Format Q-table as a string
        q_table_str = "Q-Table:\n"
        grid_size = self.current_q_table.shape[0]
        for r in range(grid_size):
            for c in range(grid_size):
                q_table_str += f"\nState ({r}, {c}):\n"
                for action_idx, action_q_value in enumerate(self.current_q_table[r, c]):
                    action_desc = f"Action {action_idx} {ACTIONS[action_idx]}"
                    q_table_str += f"  {action_desc}: {action_q_value:.4f}\n"
            q_table_str += "-" * 50 + "\n" # Separator between rows

        text_widget.insert(tk.END, q_table_str)
        text_widget.config(state="disabled") # Make text read-only


    def back_to_main_menu(self):
        if hasattr(self, 'original_grid_frame') and self.original_grid_frame.winfo_exists():
            self.original_grid_frame.destroy()
        if hasattr(self, 'real_world_frame') and self.real_world_frame.winfo_exists():
            self.real_world_frame.destroy()
        if hasattr(self, 'utilities_frame') and self.utilities_frame.winfo_exists():
            self.utilities_frame.destroy()

        # Stop any ongoing animation threads
        self.animation_stop_event.set()
        if self.animation_thread and self.animation_thread.is_alive():
            print("Joining animation thread...")
            self.animation_thread.join(timeout=0.5) # Give it a moment to stop gracefully
            print("Animation thread joined.")
        self.animation_stop_event.clear() # Reset for next use

        self.current_q_table = None
        self.current_grid_params = None
        self.current_tab_trained = None
        self.current_osm_graph = None
        self.real_world_grid_to_node_map = None
        self.real_world_node_to_grid_map = None
        self.real_world_min_coords = None
        self.real_world_max_coords = None
        self.real_world_graph_start_node = None
        self.real_world_graph_end_node = None

        # Close all matplotlib figures
        plt.close('all')

        self.create_main_menu() # Re-create the main menu

    def print_working_dir(self):
        directory = get_working_directory()
        self.working_dir_label.config(text=f"Current Working Directory: {directory}")
        messagebox.showinfo("Working Directory", f"Current Working Directory: {directory}")

    def handle_file_upload(self):
        uploaded_files = simulate_file_upload()
        if uploaded_files:
            file_name = list(uploaded_files.keys())[0]
            content = uploaded_files[file_name].decode('utf-8')
            messagebox.showinfo("Simulated Upload Result", f"File '{file_name}' uploaded. Content preview:\n{content[:100]}...")
        else:
            messagebox.showinfo("Simulated Upload Result", "No file simulated for upload.")

    def remove_chars_action(self):
        s1 = self.string1_entry.get()
        s2 = self.string2_entry.get()
        result = remove_dirty_chars(s1, s2)
        self.remove_chars_result_label.config(text=f"Removed: '{result}'")
        messagebox.showinfo("Remove Characters", f"Original: '{s1}'\nChars to remove: '{s2}'\nResult: '{result}'")

    def update_progress(self, current_episode, total_episodes, tab_name="original"):
        progress_percentage = (current_episode / total_episodes) * 100
        if tab_name == "original":
            self.progress_og['value'] = progress_percentage
            self.progress_label_og['text'] = f"Training Progress: {progress_percentage:.1f}%"
        elif tab_name == "real_world":
            self.progress_rw['value'] = progress_percentage
            self.progress_label_rw['text'] = f"Training Progress: {progress_percentage:.1f}%"
        self.root.update_idletasks() # Update the GUI

    def plot_learning_curve(self, episode_rewards, ax, canvas, title):
        ax.clear()
        if episode_rewards:
            ax.plot(episode_rewards)
            ax.set_title(f'{title} Learning Curve')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Total Reward')
        else:
            ax.text(0.5, 0.5, 'No episode rewards to plot', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f'{title} Learning Curve')
        ax.grid(True)
        canvas.draw()

    def run_original_grid_training(self):
        # Stop any ongoing animation
        self.animation_stop_event.set()
        if self.animation_thread and self.animation_thread.is_alive():
            self.animation_thread.join(timeout=0.2) # Short timeout
        self.animation_stop_event.clear() # Reset for next use

        self.clear_original_grid_canvas() # Clear previous grid path and agent
        self.progress_og['value'] = 0
        self.progress_label_og['text'] = "Training Progress: 0%"
        self.plot_learning_curve([], self.ax_og, self.canvas_og_mpl, "Original Grid") # Clear learning curve plot

        try:
            grid_size = self.grid_size_var.get()
            num_obstacles = self.num_obstacles_var.get()

            if not (4 <= grid_size <= 8):
                raise ValueError("Grid size must be between 4 and 8.")
            max_allowed_obstacles = (grid_size * grid_size) // 2
            if not (0 <= num_obstacles <= max_allowed_obstacles):
                raise ValueError(f"Number of obstacles cannot exceed {max_allowed_obstacles} (half of grid boxes).")

            # Get hyperparameters from GUI
            learning_rate = self.lr_var.get()
            discount_factor = self.df_var.get()
            epsilon_decay_rate = self.edr_var.get()
            min_epsilon = self.min_eps_var.get()
            total_episodes = self.episodes_var.get()
            if not (0 < learning_rate <= 1 and 0 <= discount_factor <= 1 and 0 < epsilon_decay_rate <= 1 and 0 <= min_epsilon < 1 and total_episodes > 0):
                raise ValueError("Hyperparameters out of valid range.")

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
            return

        # Generate a new grid
        start_state = (0, 0)
        end_state = (grid_size - 1, grid_size - 1)
        obstacles = self.generate_obstacles(grid_size, num_obstacles, start_state, end_state)

        # If generate_obstacles could not find a path, it will return an empty list of obstacles
        # or fewer than requested, but still guarantee a path if possible.
        # The `train_q_learning` function itself has a `has_path` check to confirm.
        if not has_path(grid_size, start_state, end_state, obstacles):
             messagebox.showerror("Grid Generation Error", "Could not generate a solvable grid with the specified number of obstacles. Please reduce obstacle count or increase grid size.")
             return # Abort if no path is guaranteed

        self.current_grid_params = (grid_size, start_state, end_state, obstacles)
        self.draw_grid_on_canvas(grid_size, start_state, end_state, obstacles)
        self.last_drawn_grid_params = self.current_grid_params

        # Run training in a separate thread
        def train_thread():
            self.root.after(0, lambda: messagebox.showinfo("Training Started", "Starting Q-Learning training for Original Grid. GUI will remain responsive."))
            q_table, episode_rewards = train_q_learning(
                grid_size=grid_size,
                start_state=start_state,
                end_state=end_state,
                obstacles=obstacles,
                total_episodes=total_episodes,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon_decay_rate=epsilon_decay_rate,
                min_epsilon=min_epsilon,
                progress_callback=lambda current, total: self.update_progress(current, total, "original")
            )

            if q_table is not None:
                self.current_q_table = q_table
                self.current_tab_trained = "original"
                # Clear real-world map data as we switched mode
                self.current_osm_graph = None
                self.real_world_grid_to_node_map = None
                self.real_world_node_to_grid_map = None
                self.real_world_min_coords = None
                self.real_world_max_coords = None
                self.real_world_graph_start_node = None
                self.real_world_graph_end_node = None

                self.root.after(0, lambda: self.plot_learning_curve(episode_rewards, self.ax_og, self.canvas_og_mpl, "Original Grid"))
                self.root.after(0, lambda: messagebox.showinfo("Training Complete", "Q-Learning training finished for Original Grid. Visualizing path..."))
                self.root.after(0, self.animate_original_grid_path) # Start animation
            else:
                 self.root.after(0, lambda: messagebox.showerror("Training Failed", "Q-Learning training for Original Grid failed or no path found. Check console for details."))
                 self.root.after(0, lambda: self.plot_learning_curve([], self.ax_og, self.canvas_og_mpl, "Original Grid")) # Clear plot


        self.animation_thread = threading.Thread(target=train_thread)
        self.animation_thread.daemon = True # Allow thread to exit with app
        self.animation_thread.start()

    def generate_obstacles(self, grid_size, num_obstacles, start_state, end_state):
        all_possible_cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]
        available_cells_for_obstacles = [cell for cell in all_possible_cells if cell != start_state and cell != end_state]

        obstacles = set()
        tried_cells = set() # Keep track of cells we've already tried to place an obstacle

        # Prioritize placing obstacles that don't immediately block the path
        # Loop up to a certain number of attempts, or until enough obstacles are placed
        for _ in range(num_obstacles * 5): # Give it more attempts than requested obstacles
            if len(obstacles) >= num_obstacles:
                break
            if not available_cells_for_obstacles:
                print("No more available cells to place obstacles.")
                break

            chosen_obstacle = random.choice(available_cells_for_obstacles)
            if chosen_obstacle in tried_cells: # Avoid retrying the same cell if it constantly blocks
                continue
            tried_cells.add(chosen_obstacle)

            temp_obstacles = list(obstacles | {chosen_obstacle}) # Test with new obstacle added
            if has_path(grid_size, start_state, end_state, temp_obstacles):
                obstacles.add(chosen_obstacle)
                # No need to remove from available_cells_for_obstacles unless we are strict about unique picks,
                # random.choice will eventually pick others. But for efficiency, we can remove.
                available_cells_for_obstacles.remove(chosen_obstacle)
            else:
                # If adding this obstacle blocks the path, don't add it
                # and remove it from available_cells so we don't try it again often
                if chosen_obstacle in available_cells_for_obstacles:
                    available_cells_for_obstacles.remove(chosen_obstacle)

        if len(obstacles) < num_obstacles:
            messagebox.showwarning("Obstacle Placement", f"Could not place all {num_obstacles} obstacles while ensuring a path. Only {len(obstacles)} placed.")
        return list(obstacles)


    def clear_original_grid_canvas(self):
        self.grid_canvas.delete("all")

    def draw_grid_on_canvas(self, grid_size, start_state, end_state, obstacles, current_agent_pos=None, optimal_path=None):
        self.clear_original_grid_canvas()

        canvas_width = self.grid_canvas.winfo_width()
        canvas_height = self.grid_canvas.winfo_height()

        cell_size = min(canvas_width / grid_size, canvas_height / grid_size)
        offset_x = (canvas_width - grid_size * cell_size) / 2
        offset_y = (canvas_height - grid_size * cell_size) / 2

        for r in range(grid_size):
            for c in range(grid_size):
                x1 = offset_x + c * cell_size
                y1 = offset_y + r * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                fill_color = "white"
                text = ""
                tag = f"cell_{r}_{c}"

                if (r, c) == start_state:
                    fill_color = "green"
                    text = "S"
                elif (r, c) == end_state:
                    fill_color = "red"
                    text = "E"
                elif (r, c) in obstacles:
                    fill_color = "gray"
                    text = "#"

                self.grid_canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="black", tags=tag)
                if text:
                    self.grid_canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=text, fill="black", tags=tag, font=("Arial", int(cell_size/3)))

        # Draw optimal path if provided (after initial grid to overlay)
        if optimal_path:
            for i, (r, c) in enumerate(optimal_path):
                if (r, c) != start_state and (r, c) != end_state:
                    x_center = offset_x + c * cell_size + cell_size / 2
                    y_center = offset_y + r * cell_size + cell_size / 2
                    radius = cell_size * 0.2
                    self.grid_canvas.create_oval(x_center - radius, y_center - radius,
                                                 x_center + radius, y_center + radius,
                                                 fill="blue", outline="blue", tags="path_marker")

        # Draw current agent position if provided
        if current_agent_pos:
            r, c = current_agent_pos
            x_center = offset_x + c * cell_size + cell_size / 2
            y_center = offset_y + r * cell_size + cell_size / 2
            radius = cell_size * 0.35 # Slightly larger for agent
            # Draw a circle for the agent
            self.grid_canvas.create_oval(x_center - radius, y_center - radius,
                                         x_center + radius, y_center + radius,
                                         fill="gold", outline="black", tags="agent")
        self.grid_canvas.update_idletasks() # Ensure canvas updates

    def animate_original_grid_path(self):
        if self.current_q_table is None or self.current_grid_params is None or self.current_tab_trained != "original":
            messagebox.showwarning("No Data", "Please train the Q-Learning model for the Original Grid first.")
            return

        grid_size, start_state, end_state, obstacles = self.current_grid_params
        optimal_path = find_optimal_path(self.current_q_table, grid_size, start_state, end_state, obstacles)

        if not optimal_path:
            messagebox.showwarning("Path Not Found", "Optimal path could not be determined after training. Agent might be stuck or no path exists.")
            return

        self.root.after(0, lambda: self.draw_grid_on_canvas(grid_size, start_state, end_state, obstacles)) # Redraw base grid

        # Animation loop
        def animate():
            for i, state in enumerate(optimal_path):
                if self.animation_stop_event.is_set():
                    print("Animation stopped by user.")
                    break
                # Only draw agent for current step, not the full path yet
                self.root.after(0, lambda s=state: self.draw_grid_on_canvas(grid_size, start_state, end_state, obstacles, current_agent_pos=s))
                time.sleep(0.2) # Control animation speed

            if not self.animation_stop_event.is_set():
                # After animation, draw the final path clearly and remove agent
                self.root.after(0, lambda: self.draw_grid_on_canvas(grid_size, start_state, end_state, obstacles, optimal_path=optimal_path))
                messagebox.showinfo("Animation Complete", "Agent has finished traversing the optimal path.")

        self.animation_thread = threading.Thread(target=animate)
        self.animation_thread.daemon = True
        self.animation_thread.start()


    def run_real_world_training(self):
        # Stop any ongoing animation
        self.animation_stop_event.set()
        if self.animation_thread and self.animation_thread.is_alive():
            self.animation_thread.join(timeout=0.2) # Short timeout
        self.animation_stop_event.clear() # Reset for next use

        self.clear_real_world_map_plot() # Clear previous map and plot
        self.progress_rw['value'] = 0
        self.progress_label_rw['text'] = "Training Progress: 0%"

        place_name = self.place_name_var.get().strip()
        if not place_name:
            messagebox.showwarning("Input Error", "Please enter a place name or 'Random Location'.")
            return

        # Get hyperparameters from GUI
        try:
            learning_rate = self.lr_var.get()
            discount_factor = self.df_var.get()
            epsilon_decay_rate = self.edr_var.get()
            min_epsilon = self.min_eps_var.get()
            total_episodes = self.episodes_var.get()
            if not (0 < learning_rate <= 1 and 0 <= discount_factor <= 1 and 0 < epsilon_decay_rate <= 1 and 0 <= min_epsilon < 1 and total_episodes > 0):
                raise ValueError("Hyperparameters out of valid range.")
        except ValueError as e:
            messagebox.showerror("Input Error", f"Please enter valid numeric values for hyperparameters. Error: {e}")
            return

        # Run graph loading and training in a separate thread
        def real_world_thread():
            self.root.after(0, lambda: messagebox.showinfo("Loading Map & Training", f"Starting to load map for '{place_name}' and train Q-Learning. This may take some time depending on map size and internet speed."))
            self.root.after(0, lambda: self.ax_rw_map.text(0.5, 0.5, "Loading Map...", ha='center', va='center', transform=self.ax_rw_map.transAxes))
            self.root.after(0, self.canvas_rw_mpl.draw)

            graph = get_real_world_graph(place_name=place_name)
            if not graph:
                self.root.after(0, lambda: self.update_progress(0, 1, "real_world"))
                self.root.after(0, self.clear_real_world_map_plot)
                return

            start_node_graph, end_node_graph = get_start_end_nodes(graph)
            if not start_node_graph or not end_node_graph:
                self.root.after(0, lambda: messagebox.showerror("Error", "Could not select valid start/end nodes from the map. Map might be too small or disconnected."))
                self.root.after(0, lambda: self.update_progress(0, 1, "real_world"))
                self.root.after(0, self.clear_real_world_map_plot)
                return

            # Store the original graph
            self.current_osm_graph = graph
            self.real_world_graph_start_node = start_node_graph
            self.real_world_graph_end_node = end_node_graph

            # Convert graph to a simplified grid
            REAL_WORLD_GRID_SIZE = 30 # Can be adjusted, user requested "low grid size"
            grid_size_real, start_state_real, end_state_real, obstacles_real, \
            grid_to_node_map, node_to_grid_map, min_coords, max_coords, _, _ = \
                graph_to_simplified_grid(graph, start_node_graph, end_node_graph, grid_size=REAL_WORLD_GRID_SIZE)

            if grid_size_real is None:
                self.root.after(0, lambda: messagebox.showerror("Error", "Could not create a valid grid from the map. Check console for details."))
                self.root.after(0, lambda: self.update_progress(0, 1, "real_world"))
                self.root.after(0, self.clear_real_world_map_plot)
                return

            self.current_grid_params = (grid_size_real, start_state_real, end_state_real, obstacles_real)
            self.real_world_grid_to_node_map = grid_to_node_map
            self.real_world_node_to_grid_map = node_to_grid_map
            self.real_world_min_coords = min_coords
            self.real_world_max_coords = max_coords

            # Train Q-Learning on the real-world simplified grid
            q_table_real, episode_rewards_real = train_q_learning(
                grid_size=grid_size_real,
                start_state=start_state_real,
                end_state=end_state_real,
                obstacles=obstacles_real,
                total_episodes=total_episodes,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon_decay_rate=epsilon_decay_rate,
                min_epsilon=min_epsilon,
                progress_callback=lambda current, total: self.update_progress(current, total, "real_world")
            )

            if q_table_real is not None:
                self.current_q_table = q_table_real
                self.current_tab_trained = "real_world"
                self.root.after(0, lambda: self.plot_learning_curve(episode_rewards_real, self.ax_rw_curve, self.canvas_rw_mpl, f"Real-World Grid ({place_name})"))
                self.root.after(0, lambda: messagebox.showinfo("Training Complete", f"Q-Learning training finished for Real-World Grid ('{place_name}'). Visualizing path..."))
                self.root.after(0, self.animate_real_world_path) # Start animation
            else:
                 self.root.after(0, lambda: messagebox.showerror("Training Failed", f"Q-Learning training for Real-World Grid ('{place_name}') failed or no path found. Check console for details."))
                 self.root.after(0, lambda: self.plot_learning_curve([], self.ax_rw_curve, self.canvas_rw_mpl, f"Real-World Grid ({place_name})"))
                 self.root.after(0, self.clear_real_world_map_plot)


        self.animation_thread = threading.Thread(target=real_world_thread)
        self.animation_thread.daemon = True
        self.animation_thread.start()

    def clear_real_world_map_plot(self):
        self.ax_rw_map.clear()
        self.ax_rw_map.set_title("Real-World Map Visualization")
        self.ax_rw_map.set_xticks([])
        self.ax_rw_map.set_yticks([])
        self.ax_rw_curve.clear()
        self.ax_rw_curve.set_title("Real-World Grid Learning Curve")
        self.ax_rw_curve.set_xlabel('Episode')
        self.ax_rw_curve.set_ylabel('Total Reward')
        self.ax_rw_curve.grid(True)
        self.canvas_rw_mpl.draw()

    def draw_real_world_map(self, current_agent_pos_grid=None, optimal_path_grid=None):
        if not self.current_osm_graph or not self.current_grid_params:
            return

        grid_size, start_state_grid, end_state_grid, obstacles_grid = self.current_grid_params
        graph = self.current_osm_graph
        grid_to_node_map = self.real_world_grid_to_node_map
        min_x, min_y = self.real_world_min_coords
        max_x, max_y = self.real_world_max_coords

        self.ax_rw_map.clear()
        self.ax_rw_map.set_title("Real-World Map Pathfinding")
        self.ax_rw_map.set_xticks([])
        self.ax_rw_map.set_yticks([])

        # Plot the base map with minimal objects (edges and relevant nodes)
        ox.plot_graph(graph, ax=self.ax_rw_map, show=False, close=False,
                      bgcolor='white', edge_color='lightgray', node_size=0, edge_linewidth=0.5)

        # Plot the obstacles as shaded regions
        cell_width_map = (max_x - min_x) / grid_size
        cell_height_map = (max_y - min_y) / grid_size

        def grid_to_map_center_coords(r, c):
            # Map grid cell (r, c) to center of its corresponding real-world coordinate box
            map_x = min_x + c * cell_width_map + cell_width_map / 2
            map_y = max_y - r * cell_height_map - cell_height_map / 2 # Invert row for map Y (higher row = lower lat)
            return map_x, map_y

        # Plot obstacles on the map (as semi-transparent rectangles)
        for r, c in obstacles_grid:
            map_x_rect_origin = min_x + c * cell_width_map
            map_y_rect_origin = max_y - (r + 1) * cell_height_map # y-origin for rectangle is bottom-left corner
            self.ax_rw_map.add_patch(plt.Rectangle((map_x_rect_origin, map_y_rect_origin),
                                                 cell_width_map, cell_height_map, color='darkgrey', alpha=0.5))

        # Plot Start and End Nodes (larger, distinct colors)
        # Using the original graph nodes for more accurate positioning
        if self.real_world_graph_start_node and self.real_world_graph_start_node in graph.nodes:
            s_node = self.real_world_graph_start_node
            s_x, s_y = graph.nodes[s_node]['x'], graph.nodes[s_node]['y']
            self.ax_rw_map.scatter(s_x, s_y, color='green', s=150, zorder=5, label='Start', edgecolors='black')
            self.ax_rw_map.text(s_x, s_y, 'S', fontsize=10, ha='center', va='center', color='white', zorder=6)

        if self.real_world_graph_end_node and self.real_world_graph_end_node in graph.nodes:
            e_node = self.real_world_graph_end_node
            e_x, e_y = graph.nodes[e_node]['x'], graph.nodes[e_node]['y']
            self.ax_rw_map.scatter(e_x, e_y, color='red', s=150, zorder=5, label='End', edgecolors='black')
            self.ax_rw_map.text(e_x, e_y, 'E', fontsize=10, ha='center', va='center', color='white', zorder=6)


        # Plot the optimal path if provided
        if optimal_path_grid:
            # Convert grid path to a sequence of real-world coordinates for plotting
            real_path_coords = []
            for grid_cell in optimal_path_grid:
                # Get the map coordinates for the center of this grid cell
                map_x, map_y = grid_to_map_center_coords(grid_cell[0], grid_cell[1])
                real_path_coords.append((map_x, map_y))

            if len(real_path_coords) > 1:
                # Draw lines connecting these map coordinates
                xs = [p[0] for p in real_path_coords]
                ys = [p[1] for p in real_path_coords]
                self.ax_rw_map.plot(xs, ys, color='blue', linewidth=3, solid_capstyle='round', zorder=4, label='Optimal Path')
            elif len(real_path_coords) == 1: # Just a point if path is only start/end or very short
                 x, y = real_path_coords[0]
                 self.ax_rw_map.scatter(x, y, color='blue', s=50, zorder=4, alpha=0.7)


        # Plot current agent position
        if current_agent_pos_grid:
            agent_x, agent_y = grid_to_map_center_coords(current_agent_pos_grid[0], current_agent_pos_grid[1])
            self.ax_rw_map.scatter(agent_x, agent_y, color='gold', s=200, zorder=10, marker='o', edgecolors='black') # Agent marker

        self.fig_rw.tight_layout(rect=[0, 0.1, 1, 1]) # Adjust layout to make space for title/footer
        self.canvas_rw_mpl.draw()

    def animate_real_world_path(self):
        if self.current_q_table is None or self.current_grid_params is None or self.current_tab_trained != "real_world":
            messagebox.showwarning("No Data", "Please load a map and train the Q-Learning model for the Real-World Map first.")
            return

        grid_size, start_state_grid, end_state_grid, obstacles_grid = self.current_grid_params
        optimal_path_grid = find_optimal_path(self.current_q_table, grid_size, start_state_grid, end_state_grid, obstacles_grid)

        if not optimal_path_grid:
            messagebox.showwarning("Path Not Found", "Optimal path could not be determined for the real-world map. Agent might be stuck or no path exists.")
            return

        # Ensure base map is drawn before animation
        self.root.after(0, lambda: self.draw_real_world_map(current_agent_pos_grid=None, optimal_path_grid=None))

        # Animation loop
        def animate():
            for i, state_grid in enumerate(optimal_path_grid):
                if self.animation_stop_event.is_set():
                    print("Animation stopped by user.")
                    break
                # Update map with current agent position
                self.root.after(0, lambda s_grid=state_grid: self.draw_real_world_map(current_agent_pos_grid=s_grid, optimal_path_grid=None))
                time.sleep(0.3) # Control animation speed

            if not self.animation_stop_event.is_set():
                # After animation, draw the final path clearly and remove agent marker
                self.root.after(0, lambda: self.draw_real_world_map(current_agent_pos_grid=None, optimal_path_grid=optimal_path_grid))
                messagebox.showinfo("Animation Complete", "Agent has finished traversing the optimal path on the real-world map.")

        self.animation_thread = threading.Thread(target=animate)
        self.animation_thread.daemon = True
        self.animation_thread.start()


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()