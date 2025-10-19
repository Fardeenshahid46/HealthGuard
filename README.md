# 🩺 HealthGuard – AI-Based Disease Prediction System

**HealthGuard** is a Machine Learning–powered project that predicts the likelihood of diabetes based on medical data.  
It uses advanced data preprocessing, exploratory data analysis (EDA), and machine learning algorithms to achieve **high accuracy** in predictions.

## 🚀 Features
✅ Data Cleaning & Preprocessing (handle missing values, normalize data, remove outliers)  
✅ Exploratory Data Analysis (EDA) using charts and correlations  
✅ Model Training using **Random Forest Classifier** for high accuracy  
✅ Performance Evaluation using **Accuracy, Confusion Matrix, Classification Report**  
✅ Model Saving using **Joblib** for real-world use  
✅ Ready for integration with **Streamlit or Flask** web app  

## 🧠 Tech Stack
- **Language:** Python 3.12+  
- **Libraries Used:**  
  - pandas  
  - numpy  
  - matplotlib  
  - seaborn  
  - scikit-learn  
  - joblib  

## 📊 Dataset
**Source:** [Kaggle - Diabetes Dataset](https://www.kaggle.com/datasets)  
**Attributes:**
- Pregnancies  
- Glucose  
- BloodPressure  
- SkinThickness  
- Insulin  
- BMI  
- DiabetesPedigreeFunction  
- Age  
- Outcome (Target variable: 0 = No Diabetes, 1 = Diabetes)

## ⚙️ Project Setup & Usage Guide

### 🧩 Step 1: Clone the Repository
git clone https://github.com/Fardeenshahid46/HealthGuard.git

### 🧩 Step 2: Navigate to the Project Directory
cd HealthGuard

### 🧩 Step 3: Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate        # For Windows
# OR
source venv/bin/activate     # For macOS/Linux

### 🧩 Step 4: Install Required Dependencies
pip install -r requirements.txt

### 🧩 Step 5: Add Dataset
Place your diabetes.csv file inside the data/ folder:
HealthGuard/
 ├── data/
 │   └── diabetes.csv

### 🧩 Step 6: Run the Model Training Script
python train_model.py
After training, the system will:
Clean and preprocess the data
Perform EDA (optional in eda.ipynb)
Train a Random Forest model
Save the model as model.pkl

### 🧩 Step 7: View Model Performance
After training, your terminal will display:
Accuracy score
Confusion matrix
Classification report

Example Output:
Model Accuracy: 0.89
Precision: 0.87
Recall: 0.86
F1 Score: 0.86
### 🧩 Project Structure
HealthGuard/
│
├── data/
│   └── diabetes.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   └── evaluate.py
│
├── model.pkl
├── requirements.txt
├── README.md
└── .gitignore
### 📦 requirements.txt
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
