# FaceRecognition

A desktop application for face recognition using the ORL (Olivetti Research Lab) face dataset. The system implements multiple pattern recognition algorithms.

## Algorithms

- **NN (Nearest Neighbor)** - Basic nearest neighbor classification
- **kNN (k-Nearest Neighbors)** - K-nearest neighbor with configurable K value
- **Eigenfaces** - Principal Component Analysis (PCA) for face recognition
- **Eigenfaces with Class Representative** - PCA with class representative averaging
- **Lanczos** - Lanczos algorithm for dimensionality reduction

## Features

- GUI built with CustomTkinter
- Support for multiple distance metrics: Manhattan, Euclidean, Infinity, Cosine
- Configurable training/testing split
- Statistics generation with accuracy and timing metrics
- Image visualization

## Tech Stack

- Python
- NumPy
- Matplotlib
- CustomTkinter

## Usage

1. Select the ORL dataset folder
2. Choose the number of training/testing pictures per person
3. Select normalization method (Manhattan, Euclidean, Infinity, Cosine)
4. Choose algorithm (NN, kNN, Eigenfaces, Eigenfaces with CR, Lanczos)
5. Select a test picture
6. Click "Test" to run recognition
7. Click "Generate Statistics" to run full evaluation

