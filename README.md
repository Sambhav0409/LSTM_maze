# 🧠 NeuroMap: LSTM-Based Maze Navigation Simulator

NeuroMap is a Python-based machine learning and robotics simulation where a virtual robot learns to navigate a 2D maze using **LSTM (Long Short-Term Memory)** neural networks — mimicking **human spatial memory** and decision-making.

---

## 📌 Key Highlights

- ✅ Pure Python (no ROS, no hardware dependencies)
- ✅ Realtime 2D maze simulator
- ✅ Brain-inspired path planning using LSTM
- ✅ Visualized memory-based decision making
- ✅ Internship & interview-ready ML + Robotics project

---

## 🧪 Demo Preview

<p align="center">
  <img demo/demo/src="neuro_map_demo.png" width="600"/>
</p>

> *(You can update the image path once you add screenshots or a .gif to `demo/` folder.)*

---

## 🏗️ Project Structure

```bash
LSTM_maze/
├── maze.py             # Maze generation and environment
├── model.py            # LSTM-based decision model
├── neuromap_model.pth  # Trained model weights
├── train.py            # Training script
├── visualize.py        # GUI visualizer
├── demo/               # Screenshots / gif
└── README.md           # Project description
