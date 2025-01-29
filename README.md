# 🚢 Caltech Deep Learning Project - Automating Port Operations

---

## 📌 Project Overview
This is a **Caltech Deep Learning Project**, focused on **automating boat classification at a port** to eliminate human error in misidentifications. The goal is to develop **two models**:

1. **CNN Model (Model 1)** – A custom deep learning model for boat classification.
2. **MobileNetV2 Transfer Learning Model (Model 2)** – A **lightweight** pre-trained model optimized for mobile deployment.

However, **Model 1 performed extremely poorly (33% accuracy), while Model 2 significantly improved results (66% accuracy)**. The findings suggest that **pre-trained models drastically improve performance** over custom-built CNNs.

This repository includes **detailed documentation, source code, and performance evaluations**.

---

## 🏆 Project Objectives

### 🚢 **Part 1: CNN-Based Boat Classification**
- **Problem Statement**: Marina Pier Inc. seeks to automate boat classification at the **San Francisco port** to avoid errors caused by human oversight.
- **Objective**: Develop a **CNN model** to classify boats into **9 categories**.

### 📲 **Part 2: Lightweight Model for Mobile Deployment**
- **Problem Statement**: The CNN model was **not accurate enough**, and **too large** for mobile use.
- **Objective**: Use **MobileNetV2** to improve classification accuracy while keeping the model lightweight for **real-time deployment on mobile devices**.

---

## 🛠️ Tools & Technologies Used
- **Python**
- **TensorFlow / Keras** – Deep learning framework
- **NumPy, Pandas** – Data manipulation
- **Matplotlib, Seaborn** – Data visualization
- **OpenCV** – Image processing
- **Scikit-learn** – Model evaluation metrics

---

## 📂 Dataset Description

### 📸 **Boat Image Classification Dataset**
The dataset contains **1,162 images** of boats, categorized into **9 classes**:

| Boat Type          |
|--------------------|
| Buoy              |
| Cruise Ship       |
| Ferry Boat        |
| Freight Boat      |
| Gondola           |
| Inflatable Boat   |
| Kayak             |
| Paper Boat        |
| Sailboat          |

- Images are **split into training and testing sets**.
- The dataset is **loaded using** `tf.keras.preprocessing.image_dataset_from_directory`.

---

## 📊 Steps to Perform

### **🛠 Part 1: CNN-Based Model (Model 1)**
#### 1️⃣ **Data Preparation**
- **Split dataset** into **80% training, 20% testing**.
- Normalize images using **image scaling (1./255)**.
- Load data in **batches of 32**.

#### 2️⃣ **Building the CNN Model**
- **Architecture**:
  - Conv2D (32 filters, 3×3, ReLU) → MaxPool2D
  - Conv2D (32 filters, 3×3, ReLU) → MaxPool2D
  - GlobalAveragePooling2D
  - Dense (128 neurons, ReLU)
  - Dense (128 neurons, ReLU)
  - Dense (9 neurons, Softmax)
- **Compilation**:
  - **Optimizer**: Adam
  - **Loss function**: Categorical Crossentropy
  - **Metrics**: Accuracy, Precision, Recall
- **Training**:
  - Model trained for **20 epochs**.
  - Tracked **loss and accuracy** over epochs.

#### 3️⃣ **Evaluation & Model Performance**
- **Model 1 (CNN) performed poorly** with **only 33% accuracy**.
- The model **failed to recognize most boat types**, especially **inflatable_boat, cruise_ship, and freight_boat**.
- **Data augmentation and resampling attempts did not improve performance**.

---

### **📲 Part 2: MobileNetV2 Transfer Learning Model (Model 2)**
#### 1️⃣ **Data Preprocessing**
- **Split dataset** into **70% training, 30% testing**.
- Normalize images using **image scaling (1./255)**.
- Load data in **batches of 32**.

#### 2️⃣ **Building the Transfer Learning Model**
- **Pre-trained Model**: **MobileNetV2** (pre-trained on ImageNet).
- **Architecture**:
  - MobileNetV2 (pre-trained on ImageNet)
  - GlobalAveragePooling2D
  - Dropout (0.2)
  - Dense (256 neurons, ReLU) → BatchNormalization → Dropout (0.1)
  - Dense (128 neurons, ReLU) → BatchNormalization → Dropout (0.1)
  - Dense (9 neurons, Softmax)
- **Compilation**:
  - **Optimizer**: Adam
  - **Loss function**: Categorical Crossentropy
  - **Metrics**: Accuracy, Precision, Recall
- **Training**:
  - Model trained for **50 epochs** with **Early Stopping (monitoring validation loss)**.

#### 3️⃣ **Evaluation & Model Performance**
- **Model 2 (MobileNetV2) significantly improved accuracy to 66%**.
- **Cruise_ship, Kayak, and Sailboat classifications saw major improvements**.
- **Precision and Recall scores were much higher**, indicating **better class balance**.

---

## 📌 Key Findings & Observations

### 🚢 **CNN-Based Model (Model 1)**
❌ **Failed to classify most boat types**.  
❌ **Extremely low accuracy (33%)**.  
❌ **Struggled with underrepresented classes like Freight Boat and Inflatable Boat**.  

---

### 📲 **MobileNetV2 Transfer Learning Model (Model 2)**
✅ **Improved accuracy to 66%**.  
✅ **Performed much better on larger classes like Cruise Ship, Kayak, and Sailboat**.  
✅ **Reduced false positives and improved recall**.  

⚠️ **Remaining Challenges**:
- Some **boat types (Inflatable Boat, Freight Boat) still had poor performance** due to limited training samples.
- The dataset needs **more diversity and balance** for underrepresented classes.
- **66% accuracy is still not production-ready** – further tuning required.

### 🔥 **Comparison of Both Models**
| Model                  | Training Time | Accuracy | Suitable for Mobile? |
|------------------------|--------------|----------|----------------------|
| **CNN (Custom Model)** | Longer       | 33%      | ❌ No               |
| **MobileNetV2 (TL)**  | Faster       | 66%      | ✅ Yes |

---

## 🚀 How to Use This Project
1. Clone the repository:
   ```bash
   git clone https://github.com/lthornqu/caltech-deep-learning-automating-port-operations.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the **Jupyter Notebook** to explore the models and perform inference.

---

## 📌 Future Improvements
🔹 **Increase Dataset Size**: More images needed for **underrepresented classes**.  
🔹 **Better Data Augmentation**: Improve diversity in **training samples**.  
🔹 **Fine-Tune Transfer Learning**: Experiment with **different architectures (ResNet, EfficientNet)**.  
🔹 **Optimize for Edge Deployment**: Convert the model to **TFLite for mobile inference**.  

---

## 👨‍💻 Contributors
- **Lee Thornquist** - Deep Learning Engineer & Model Developer
