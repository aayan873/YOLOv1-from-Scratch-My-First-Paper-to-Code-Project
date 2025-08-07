# YOLOv1 from scratch (Paper Reproduction)


A full Pytorch implementation of the **YOLOv1 (You look only Once)** object detection model, reproduced from the original paper:

> **"You Only Look Once: Unified, Real-Time Object Detection"**  
> Joseph Redmon et al. (CVPR 2016)  
> [Paper Link](https://arxiv.org/abs/1506.02640)

---

## Motivation
While learning PyTorch and exploring research in Computer Vision, I decided to reproduce **YOLOv1** from scratch to:
- Deepen my understanding of the **YOLO architecture**
- Refine my **PyTorch skills**
- Gain experience translating **research papers into working code**
  
I had previously used pre-built YOLO models for object detection, but this project pushed me to build everything from the ground up.

## Overview 
This project reproduces **YOLOv1**, one of the most influential object detection model, entirely from scratch using PyTorch.
It includes the full model architecure, custom loss function, and data pipeline, the goal was to deeply understand and implemeting a research paper from scratch, which also helped in improving my PyTorch skills

> Training has not been completed due to compute limitations.  
> The code includes a test forward pass and modular structure to support future training and visualization.

## Project Structure

├── model.py # YOLOv1 architecture
├── loss.py # Custom YOLOv1 loss function
├── utils.py # Utility functions (e.g., IOU, grid conversions)
├── dataset.py # Data pipeline placeholder
├── training.py # Training script (untrained)
├── inference.py # Inference on test set
├── README.md # This file!

## 🔧 Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- torchvision  
- numpy  
- tqdm

Install dependencies:

```bash
pip install -r requirements.txt

