# Retinal Blindness (Diabetic Retinopathy) Detection

## 📌 Description
A deep learning-based classification model to detect diabetic retinopathy from retinal fundus images using transfer learning with MobileNet. This project leverages TensorFlow and Keras to implement the model pipeline with data augmentation, performance optimization, and model saving for real-world deployment.

---

## 🧠 Model Overview
- **Base Model**: MobileNet (pretrained on ImageNet)
- **Custom Layers**: GlobalAveragePooling, Flatten, Dense, Dropout
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy with label smoothing
- **Callbacks**: EarlyStopping, ReduceLROnPlateau
- **Accuracy Metric**: Categorical Accuracy

---

## 📂 Dataset
- Source: aptos2019-blindness-detection
- Format: Fundus images in `.png`
- Labels: 0 - No DR, 1 - Mild, 2 - Moderate, 3 - Severe, 4 - Proliferate

---

## 🔧 How to Run

1. Clone the repository.
2. Place dataset inside the `dataset` folder.
3. Run the script:

```bash
python diabetic_retinopathy_detection.py
```

The model will train and save as `model.h5`.

---

## 📊 Results
- Accuracy: 82%
- Confusion matrix and class distribution visualizations are generated using `seaborn`.

---

## ✅ Requirements

```bash
pip install tensorflow pandas numpy matplotlib seaborn pillow scikit-learn
```

---

## 📌 Future Work
- Integrate the model with a web app using Flask.
- Add Grad-CAM visualizations to highlight decision areas.
- Expand the dataset for better generalization.
----
