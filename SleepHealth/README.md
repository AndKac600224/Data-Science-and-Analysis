# 🌙 Sleep Disorder Predictive Analytics: Lifestyle & Health Correlation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![ML-Accuracy](https://img.shields.io/badge/Accuracy-96%25-green.svg)]()

## 📌 Project Overview
This project focuses on identifying the key lifestyle and physiological drivers behind sleep disorders such as **Insomnia** and **Sleep Apnea**. Using a dataset of 374 individuals, I developed a machine learning pipeline that predicts the presence of a sleep disorder with **96% accuracy**.

The analysis highlights how critical factors like blood pressure and BMI directly correlate with sleep quality and clinical diagnoses.

**Dataset source:** https://www.kaggle.com/datasets/siamaktahmasbi/insights-into-sleep-patterns-and-daily-habits

## 📊 Key Highlights
* **Predictive Power:** Achieved 96% accuracy using Random Forest and XGBoost.
* **Feature Engineering:** Engineered high-impact features by decomposing blood pressure into Systolic and Diastolic components.
* **Dimensionality Reduction:** Optimized model efficiency by applying a correlation threshold (0.15), removing noise while retaining 95% of the variance.
* **Medical Logic:** The model correctly prioritized high-risk physiological markers (BP, BMI) over subjective survey data.

## 🛠 Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn
* **Tools:** Jupyter Notebook / Google Colab

## 🚀 The Data Pipeline

### 1. Data Preprocessing & Engineering
* Handled categorical variables via **One-Hot Encoding**.
* Performed **Feature Splitting**: Extracted `Systolic_BP` and `Diastolic_BP` from compound string data.
* Applied **StandardScaler** to ensure distance-based models (k-NN, SVM) perform optimally.

### 2. Model Benchmarking
I compared 5 different classification algorithms to find the most stable predictor:

| Model | Accuracy | F1-Score (Weighted) |
| :--- | :---: | :---: |
| **Random Forest** | **96%** | **0.96** |
| **XGBoost** | **96%** | **0.96** |
| Logistic Regression | 93% | 0.93 |
| k-Nearest Neighbors | 93% | 0.93 |
| SVM (RBF Kernel) | 93% | 0.93 |

### 3. Insights & Interpretation
Using **Random Forest Feature Importance**, the analysis revealed that the most significant predictors are:
1.  **Systolic & Diastolic Blood Pressure**
2.  **BMI Category (Overweight)**
3.  **Sleep Duration**

> [!TIP]
> **Conclusion:** Physiological indicators are much stronger predictors of sleep disorders than subjective "Stress Level" ratings.

## 📈 Visualizing Results

### Feature Importance
*This chart shows which variables had the most impact on the model's decisions.*
<img width="985" height="547" alt="image" src="https://github.com/user-attachments/assets/b8626cb3-0539-445a-9f3e-9470f0f6490c" />

### Confusion Matrix
*The model's performance on unseen data—only 3 misclassifications out of 75 cases.*
<img width="611" height="455" alt="image" src="https://github.com/user-attachments/assets/65329fc1-4239-4361-aaf4-b3998300109b" />

## 💡 How to Run
1. Clone the repository: `git clone https://github.com/[Andkac600224]/SleepHealth.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Open the `.ipynb` file in Jupyter Notebook or Google Colab.
