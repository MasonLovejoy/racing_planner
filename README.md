# Autonomous Racing Trajectory Planner
## Project Overview:
This system predicts future waypoints for autonomous racing vehicles using:

Track boundary information (left/right track edges)  
Raw visual input (RGB camera images)  
Real-time trajectory optimization  

The planners are trained and evaluated on the SuperTuxKart racing simulator, demonstrating robust performance across various track configurations.

## Architecture:
Three distinct model architectures are implemented:
1. MLP Planner

Multi-layer perceptron baseline  
Processes flattened track boundary points  
Fast inference with minimal computational overhead  

2. Transformer Planner

Attention-based architecture for trajectory prediction  
Learns complex spatial relationships between track features  
Superior performance on challenging track geometries  

3. CNN Planner

Vision-based approach using ResNet-style architecture  
End-to-end learning from raw RGB images  
Robust to varying lighting and track conditions  

## Key Features:
Multi-modal input processing: Handles both geometric (track boundaries) and visual (camera) data  
Real-time prediction: Optimized for low-latency trajectory generation  
Evaluation metrics: Comprehensive longitudinal and lateral error tracking  
Data augmentation: Extensive augmentation pipeline for robust training  
Visualization tools: Built-in video generation for qualitative analysis  

## Quick Start
Training a Model:
bash# Train MLP planner  
python3 -m racing_planner.train_planner --model_name mlp_planner --num_epoch 50

# Train Transformer planner
python3 -m racing_planner.train_planner --model_name transformer_planner --lr 1e-4

# Train CNN planner
python3 -m racing_planner.train_planner --model_name cnn_planner --batch_size 128

## Evaluation
bash# Evaluate on different tracks  
python3 -m racing_planner.supertux_utils.evaluate --model mlp_planner --track lighthouse  
python3 -m racing_planner.supertux_utils.evaluate --model transformer_planner --track snowmountain  
python3 -m racing_planner.supertux_utils.evaluate --model cnn_planner --track cornfield_crossing  

## Technical Stack

PyTorch: Deep learning framework  
TensorBoard: Training visualization and logging  
PySTK: SuperTuxKart Python bindings for simulation  
OpenCV: Image processing and visualization  
NumPy: Numerical computations  

## Project Structure
racing_planner/  
├── models.py              # Model architectures  
├── train_planner.py       # Training pipeline  
├── metrics.py             # Evaluation metrics  
├── datasets/  
│   ├── road_dataset.py    # Data loading and processing  
│   ├── road_transforms.py # Augmentation pipeline  
│   └── road_utils.py      # Utility functions  
└── supertux_utils/  
    ├── evaluate.py        # Evaluation framework  
    └── video_visualization.py  # Visualization tools  

## Data Augmentation
Robust augmentation pipeline includes:

Horizontal flipping for left/right symmetry  
Lateral shifts to simulate track position variance  
Gaussian noise injection for robustness  
Dynamic track boundary perturbations  

## Model Details
Input Specifications

Track-based models: 10 points per track boundary (left/right)  
Vision-based models: 96x128 RGB images  
Output: 3 future waypoints in ego-centric coordinates  

## Training Configuration

Optimizer: AdamW with weight decay  
Scheduler: Cosine annealing learning rate  
Loss: Mean Squared Error (MSE) with masked computation  
Gradient clipping: Max norm of 1.0  
