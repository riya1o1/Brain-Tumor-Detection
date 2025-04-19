# Brain Tumor Detection using CNN

This project uses Convolutional Neural Networks (CNN) to detect brain tumors from MRI images. The goal is to assist in fast and accurate diagnosis using deep learning.

## 🧠 Overview
- Classifies MRI brain scans as **Tumor** or **No Tumor**
- Built using TensorFlow/Keras
- Trained on labeled MRI datasets

## 🔧 Tech Stack
- Python
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib
- OpenCV
- Google Colab

## 📁 Dataset
- [Google Drive Dataset](https://drive.google.com/drive/u/0/folders/1IkGdDJMzTyT1cmkfxUxrQnE7hKA4FCVf)
- Contains MRI images labeled as `Tumor` and `No Tumor`
- Includes preprocessing steps: resizing, normalization, and augmentation

## 🧱 Model Architecture
- Multiple Conv2D + MaxPooling layers
- Flatten + Dense layers
- Output: Sigmoid for binary classification
- Optimizer: Adam
- Loss: Binary Crossentropy

## 📊 Results
- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~95%
- Uses EarlyStopping to avoid overfitting

## ▶️ How to Run
1. Open the [Colab Notebook](https://colab.research.google.com/drive/1etk6n3h0Jpa6A10AVFFrXn2lEe9H8SHC#scrollTo=R2UvH8_Fw2hW)
2. Upload the dataset to Colab
3. Run all cells step by step
4. Model will train and show evaluation metrics

