# TMNIST-94: Deep Learning Character Recognition

A deep learning project using a 7-layer Convolutional Neural Network (CNN) to classify 94 different characters from the TMNIST Alphabet dataset, achieving 93.99% accuracy.

## Overview

**TMNIST-94** extends the classic MNIST challenge to a diverse set of 94 characters, including:
- Uppercase and lowercase letters
- Digits (0-9)
- Special characters

With over 281,000 grayscale images (28x28 pixels), this project demonstrates robust multi-class character recognition using deep learning.

**Dataset:**  
TMNIST Alphabet 94 Characters  
*Available on Kaggle:*  
https://www.kaggle.com/datasets/nikbearbrown/tmnist-alphabet-94-characters

## Features

- **Custom 7-layer CNN** built with TensorFlow/Keras
- **Data preprocessing** and exploratory data analysis (EDA)
- **Hyperparameter tuning** (batch size, epochs, learning rate, early stopping)
- **Visualization** of predictions and training metrics
- **High accuracy**: 93.99% on the test set
- **Ready for OCR and document digitization applications**

## Model Architecture

| Layer Type         | Output Shape          | Parameters |
|--------------------|----------------------|------------|
| Input (28x28x1)    | (28, 28, 1)          | 0          |
| Conv2D (32 filters)| (26, 26, 32)         | 320        |
| MaxPooling2D       | (13, 13, 32)         | 0          |
| Conv2D (64 filters)| (11, 11, 64)         | 18,496     |
| MaxPooling2D       | (5, 5, 64)           | 0          |
| Flatten            | (1600)               | 0          |
| Dense (128 units)  | (128)                | 204,928    |
| Dropout (0.5)      | (128)                | 0          |
| Dense (94 units)   | (94)                 | 12,126     |

**Total trainable parameters:** 235,870

## Training Details

- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Batch Size:** 32
- **Epochs:** 20 (with early stopping)
- **Validation Split:** Used for performance monitoring

## Results

- **Test Accuracy:** 93.99%
- **Precision/Recall:** High across most classes (see detailed evaluation in notebook)
- **Visualization:** Side-by-side plots of true vs. predicted labels for sample test images

## Usage

1. **Clone the repository**
2. **Install requirements:**  
   `pip install numpy pandas matplotlib scikit-learn tensorflow keras`
3. **Run the notebook:**  
Open `tmnist-malini-janaki-sankaran.ipynb` in Jupyter Notebook or Colab.
4. **Train or evaluate the model** as per instructions in the notebook.

## Requirements

The following Python libraries are required (as used in the project notebook):
- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow
- keras

## Example Prediction Visualization

> True: F  
> Pred: F

> True: /  
> Pred: /

*...and more, as visualized in the notebook.*

## Applications

- Optical Character Recognition (OCR)
- Document digitization
- Accessibility tools
- CAPTCHA solving research

## Credits

Developed by Malini Janaki Sankaran as part of Data Science Engineering Methods and Tools course project.  
**Dataset:** [TMNIST Alphabet 94 Characters on Kaggle](https://www.kaggle.com/datasets/nikbearbrown/tmnist-alphabet-94-characters)

*For detailed code, model evaluation, and further instructions, see the notebook in this repository.*

