# PointNet Optimization for 3D Point Cloud Classification
This repository contains the implementation of the paper "Advancing Point Cloud Classification with Deep Learning by Optimizing PointNet through Transformations for Superior Accuracy", presented at the 10th International Conference on Information and Communication Technology (ICICT 2025), London, UK. The project achieves 92% accuracy on the ModelNet40 dataset by enhancing PointNet with advanced preprocessing and architectural optimizations.

## Key Features
- Optimized PointNet Architecture: Improves 3D point cloud classification through spatial transformations and feature extraction.
- High Performance: Achieves 92% accuracy, 91% precision/recall, and 96% sensitivity/specificity on ModelNet40.
- Modular Codebase: Clean separation of data preprocessing, model training, and utilities.
- Reproducible Results: Includes scripts for dataset preparation, training, and evaluation.

## Repository Structure

pointnet-optimization/
├── dataset.py # Prepares and preprocesses ModelNet40 dataset
├── utils.py # Installs dependencies and provides helper functions
├── model.py # Implements the enhanced PointNet architecture
├── train.py # Handles model training and evaluation
├── README.md # Project documentation
└── requirements.txt # Python dependencies


---

## Setup & Installation
1. Clone the repository:
   ```bash
   git clone our repository link
   cd pointnet-optimization
   Install dependencies:

   python utils.py --install
   Preprocess Dataset:
   python dataset.py --data_dir /path/to/ModelNet40 --output_dir ./processed_data
   Train the Model:#
   python train.py --data_dir ./processed_data --epochs 50 --batch_size 32
   Evaluate:
   python utils.py --plot_metrics --log_dir ./logs

## Results

The optimized PointNet model achieves the following performance metrics on the ModelNet40 dataset:

| Metric               | Score |
|----------------------|-------|
| Accuracy             | 92%   |
| Precision            | 91%   |
| Recall               | 91%   |
| F1-Score             | 89%   |
| Sensitivity (TPR)    | 96%   |
| Specificity (TNR)    | 96%   |

Confusion matrix for selected classes:

| Actual \ Predicted | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 |
|--------------------|---------|---------|---------|---------|---------|
| **Class 1**        | 99      | 1       | 0       | 0       | 0       |
| **Class 2**        | 0       | 98      | 2       | 0       | 0       |
| **Class 3**        | 0       | 1       | 97      | 2       | 0       |
| **Class 4**        | 0       | 0       | 1       | 98      | 1       |
| **Class 5**        | 0       | 0       | 0       | 1       | 99      |

   @inproceedings{akbar2025pointnet,
  title={Advancing Point Cloud Classification with Deep Learning by Optimizing PointNet through Transformations for Superior Accuracy},
  author={Akbar, Muhammad Sufyan and Jiandong, Guo and Khan, Muhammad Irfan and Iqbal, Asif and Salim},
  booktitle={10th International Conference on Information and Communication Technology (ICICT)},
  year={2025},
  location={London, UK}
}
