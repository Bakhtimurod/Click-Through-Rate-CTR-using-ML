# Click-Through Rate (CTR) Prediction Using Machine Learning

This repository contains a comprehensive machine learning project that predicts the probability of users clicking on advertisements, also known as Click-Through Rate (CTR). The project addresses challenges such as class imbalance and uses advanced modeling techniques to enhance prediction accuracy.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Dataset](#dataset)
4. [Setup and Installation](#setup-and-installation)
5. [Project Workflow](#project-workflow)
    - [1. Data Preprocessing](#1-data-preprocessing)
    - [2. Feature Engineering](#2-feature-engineering)
    - [3. Model Training](#3-model-training)
    - [4. Evaluation](#4-evaluation)
6. [Key Results](#key-results)
7. [Future Enhancements](#future-enhancements)


---

## Project Overview
Predicting Click-Through Rate (CTR) is critical in digital advertising for optimizing ad placement and targeting strategies. This project uses machine learning models to predict whether a user will click on an advertisement based on features like user behavior, ad characteristics, and platform data.

---

## Technologies Used
- **Programming Language**: Python
- **Libraries**: pandas, numpy, scikit-learn, seaborn, matplotlib, XGBoost
- **Tools**: Google BigQuery, Jupyter Notebook
- **Techniques**:
    - SMOTE for class imbalance handling
    - Hyperparameter tuning using GridSearchCV
    - Visualization of feature importance and data distribution

---

## Dataset
The dataset was sourced from [Kaggle](https://www.kaggle.com/), containing clickstream data with features such as:
- **Categorical Variables**: Gender, product, and campaign ID.
- **Numerical Variables**: Age, user depth, and city development index.
- **Target Variable**: `is_click` (1 for clicked, 0 for not clicked).

---

## Setup and Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/YourUsername/CTR-Prediction.git
    cd CTR-Prediction
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Ensure Google BigQuery credentials are configured for fetching the dataset.

---

## Project Workflow

### **1. Data Preprocessing**
- Handled missing values using random sampling and imputation techniques.
- Encoded categorical features using **LabelEncoder**.
- Scaled numerical features using **StandardScaler** for consistent range.
- Addressed class imbalance with **SMOTE**, generating synthetic samples for the minority class.

### **2. Feature Engineering**
- Extracted time-based features (e.g., hour of the day from timestamps).
- Engineered interaction-based features (e.g., product interactions and campaign engagement).
- Performed exploratory data analysis (EDA) to understand feature distributions and correlations.

### **3. Model Training**
- Trained the following machine learning models:
    - Logistic Regression
    - Random Forest
    - Gradient Boosted Trees
    - XGBoost
- Hyperparameter tuning was performed using **GridSearchCV**.

### **4. Evaluation**
- Evaluated models using:
    - **Accuracy**
    - **Precision**
    - **Recall**
    - **F1-Score**
    - **ROC-AUC**
- Random Forest and XGBoost were the top-performing models.

---

## Key Results
- The best-performing model achieved:
    - **Accuracy**: 0.91
    - **Precision**: 0.89
    - **Recall**: 0.87
    - **F1-Score**: 0.88
    - **ROC-AUC**: 0.93
- Insights:
    - Clicks are highly influenced by product type and campaign ID.
    - Click activity peaks at certain times of the day, as seen in time-based feature analysis.

---

## Future Enhancements
- Incorporate deep learning models (e.g., Neural Networks) for better feature representation.
- Add real-time clickstream data integration using tools like Kafka or AWS Kinesis.
- Automate model deployment using frameworks like TensorFlow Serving or Flask.

---

