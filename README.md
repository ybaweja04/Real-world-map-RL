
# Q-Learning Pathfinding Simulator with Real-World Map Integration

> ⚠️ **DISCLAIMER**  
> This code is **strictly prohibited** from being used, copied, modified, distributed, or published in any form without the **explicit written permission** of the original author. **All rights reserved.**

## 🧭 Overview

This Python-based project provides a **graphical simulation of Q-Learning pathfinding** in two environments:

- 🧱 **Original Grid-Based Simulation**
- 🌐 **Real-World Map Simulation** using OpenStreetMap data (via OSMnx)

Users can configure training parameters, train an agent using reinforcement learning, and visualize its learning process and optimal path.

---

## 🎯 Features

- 🔲 Configurable Grid Environment (4x4 to 8x8)
- 🗺 Real-World Map Integration (OpenStreetMap + OSMnx)
- 📊 Live Learning Curve Plotting
- 🤖 Agent Animation and Path Visualization
- ⚙️ Adjustable Hyperparameters
- 🧰 Extra Utility Tools (e.g., simulated file upload, string filtering)

---

## 🛠 Tech Stack

- **Python 3.x**
- **Tkinter** – GUI framework
- **NumPy** – Numerical Q-table operations
- **Matplotlib** – Plotting training performance and paths
- **NetworkX** – Graph structure support
- **OSMnx** – Real-world street map data integration
- **Threading** – Asynchronous training animations
- **Pillow (PIL)** – Image support in GUI

---

## 🚀 Getting Started

### 📦 Installation

Run the following command to install all required packages:

```bash
pip install osmnx matplotlib networkx geopandas descartes
```

> The script will also attempt auto-installation if dependencies are missing.

### ▶️ Run the Application

```bash
python Final_Project_RL.py
```

---

## ⚙️ Adjustable Hyperparameters

All hyperparameters can be changed via the GUI:

- **Learning Rate (α)** – Determines how much new information overrides old
- **Discount Factor (γ)** – Determines future reward importance
- **Epsilon Decay Rate** – Controls exploration rate over time
- **Minimum Epsilon** – Floor for exploration probability
- **Total Episodes** – Number of training iterations

---

## 🗂 Project Structure (Main Modules)

### 1. Original Grid Simulation

- Set grid size and number of obstacles
- Run Q-Learning training
- Visualize learned optimal path

### 2. Real-World Map Training

- Enter a location (e.g., "Mumbai, India") or use a "Random Location"
- Convert graph nodes to grid format
- Train and visualize path on real-world data

### 3. Utilities

- View current working directory
- Simulate file uploads
- Remove selected characters from strings

---

## 🛑 License & Usage

**🚫 Strict License and Usage Restriction**

This project is the intellectual property of its author and is protected under copyright.

> **You are not allowed to:**
> - Use this code in your own project or platform.
> - Reproduce, redistribute, or host this code publicly or privately.
> - Publish any derivative work based on this code.
> 
> **Any unauthorized use is a violation of intellectual property rights.**

---

## 🙏 Acknowledgements

- Developed as a standalone academic demonstration project.
- Built using open-source tools: OSMnx, Matplotlib, NetworkX, etc.

---

**📧 For permissions or collaboration inquiries, please contact the author directly.**
