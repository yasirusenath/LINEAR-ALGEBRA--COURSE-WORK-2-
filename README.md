# Linear Algebra for Machine Learning: Image Classification

## Project Overview

This project demonstrates how linear algebra concepts are applied in machine learning through grayscale image classification. It is designed as a beginner-friendly guide for understanding the relationship between mathematical operations and practical machine learning models.

Using three classes of grayscale images, the project explores how **Principal Component Analysis (PCA)** and **Singular Value Decomposition (SVD)** reduce image dimensionality before applying **Multinomial Logistic Regression** for multi-class classification.


<img width="1536" height="1024" alt="From Pixels to Student Predictions" src="https://github.com/user-attachments/assets/494f0594-b4da-4ae3-a95d-e764f7e5c106" />

## Project Workflow

1. **Image Preprocessing:** Load grayscale images, resize them to 300 × 300 pixels, and convert them into numerical vectors.
2. **Dimensionality Reduction:** Apply PCA to reduce the original 90,000 pixel features to 50 principal components.
3. **Model Training:** Train a multinomial logistic regression classifier using the reduced feature representation.
4. **Model Evaluation:** Evaluate classification performance using accuracy, a classification report, and a confusion matrix.
5. **Mathematical Interpretation:** Explore how vectors, matrices, eigenfaces, PCA, and matrix multiplication contribute to the classification process.

## Technologies Used

- Python
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- Jupyter Notebook / Google Colab

## Learning Objectives

- Understand why linear algebra is important in machine learning.
- Learn how image data can be represented as vectors and matrices.
- Explore PCA and SVD for dimensionality reduction.
- Understand how logistic regression performs multi-class classification.
- Evaluate a machine learning model using standard classification metrics.

## Getting Started

Open the `CW2_Linear_Algebra_Image_Classification_Guide.ipynb` notebook in Jupyter Notebook or Google Colab, prepare the three image-class folders, and follow the code cells and explanations step by step.

**Note:** Classification results depend on the image dataset, its quality, and the distribution of images across the classes.
