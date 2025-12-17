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
