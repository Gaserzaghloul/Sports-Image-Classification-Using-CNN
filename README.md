# Sports Image Classification Using CNN with 98.34% Accuracy

Built a deep CNN for multi-class sports image classification using a Kaggle dataset.
Tuned hyperparameters (layers, dropout, optimizer, batch size, learning rate, weight decay) and trained with early stopping and evaluation metrics.
Achieved 98.34% validation accuracy on a validation set of 4,937 images out of a total of 24,681 augmented images.
Visualized results with confusion matrix, PCA projection, and training curves.

## 🧠 Project Overview

I developed a deep learning pipeline that includes:
- Data preprocessing (cleaning, encoding, standardization, and PCA)
- **Data Augmentation** (Flip, Rotation, Brightness)
- CNN model training with dynamic architecture and hyperparameter tuning

## 🗂️ Dataset

- Original dataset: 8,227 training images  
- After augmentation (×3): **24,681 images**
  - **Training set**: 19,744 images  
  - **Validation set**: 4,937 images  
- Test set: 2,056 images

## 📈 Final Results

- **Validation Accuracy:** 98.34%
- **Architecture:** 5 Conv Blocks + BatchNorm + Dropout + Dense layers
- **Training Epochs:** 50 (with early stopping)
- **Loss Function:** Sparse Categorical Crossentropy
- **Best Parameters:**
  - Layers: 5
  - Dropout: 0.2
  - Batch Size: 64
  - Optimizer: Adam
  - Learning Rate: 0.0005
  - Weight Decay: 1e-4
    
## 🚀 Deployment

The final trained model was deployed using **Flask** to serve predictions through a simple web interface.  
Users can upload an image of a sport, and the system will return the predicted class with high confidence.

## 🛠️ Technologies Used

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)

