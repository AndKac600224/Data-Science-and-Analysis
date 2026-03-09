# ✈️ Indian Domestic Flight Price Prediction | Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20Random%20Forest-orange)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-yellow)
![Data Science](https://img.shields.io/badge/Data%20Science-EDA%20%7C%20Feature%20Engineering-success)

<img src="https://images.unsplash.com/photo-1530521954074-e64f6810b32d?q=80&w=2070&auto=format&fit=crop" width="100%" alt="Aviation Background">

## 📌 Project Overview
Flight ticket prices are highly dynamic and depend on a complex interplay of factors such as the airline's pricing strategy, time left to departure, and travel class. This project aims to analyze these pricing patterns within the Indian domestic aviation market and build a robust Machine Learning pipeline capable of accurately predicting ticket prices.

The ultimate goal is to provide a highly accurate regression model that could be utilized by consumers to find the optimal time to buy a ticket, or by travel agencies for price forecasting.

## 📊 The Dataset
The dataset contains over **300,000 data points** and **11 features** regarding flight booking options from India's top 6 metro cities. 
* **Target Variable:** `price` (in INR)
* **Key Features:** `airline`, `source_city`, `destination_city`, `departure_time`, `arrival_time`, `stops`, `class`, `duration`, `days_left`.

## 🛠️ Methodology & Technical Approach
This project demonstrates a complete Data Science lifecycle, emphasizing best practices such as strict separation of training/testing data to avoid **Data Leakage**.

1. **Exploratory Data Analysis (EDA):**
   * Analyzed price distributions across different airlines (Full-service vs. Low-cost carriers).
   * Investigated the non-linear relationship between `days_left` and ticket `price`.
2. **Data Preprocessing & Feature Engineering:**
   * Dropped non-predictive columns (e.g., flight numbers).
   * Applied **One-Hot Encoding** for nominal categorical variables (dropping the first category to prevent multicollinearity).
   * Applied **Ordinal Encoding** for hierarchical features (`class`, `stops`).
   * Used **StandardScaler** explicitly fitted only on training data to ensure robust scaling without data leakage.
   * Evaluated composite features (like creating a `route` feature), prioritizing model simplicity (Occam's razor) based on correlation matrices.
3. **Machine Learning Modeling:**
   * **Baseline:** Linear Regression (Revealed struggles with non-linear pricing tiers and Business Class outliers).
   * **Intermediate:** Random Forest Regressor (Successfully captured non-linearity, significantly dropping the error rate).
   * **Advanced:** XGBoost Regressor optimized via **GridSearchCV** and validated with **Early Stopping** mechanisms.

## 🏆 Key Findings & Model Performance
The tree-based ensemble models significantly outperformed linear approaches. The **XGBoost Regressor**, fine-tuned via GridSearchCV, yielded the best overall metrics, proving that the dataset holds highly predictable and logical patterns.

| Model | R² Score | MAE (Mean Absolute Error) | RMSE |
|-------|----------|---------------------------|------|
| Linear Regression | 0.904 | ~4,500 INR | ~7,000 INR |
| Random Forest | 0.939 | 3,361 INR | 5,583 INR |
| **XGBoost (Optimized)** | **0.950** | **2,946 INR** | **5,038 INR** |

**Business Insights:**
* **Travel Class is King:** The `class` feature (Economy vs. Business) dictates the baseline price more than any other variable.
* **Booking Window:** The `days_left` feature confirms that booking 2-3 weeks in advance typically yields the most stable prices, with exponential spikes occurring within 5 days of departure.
* **Airline Premium:** Full-service airlines like *Vistara* and *Air India* consistently show higher price floors compared to low-cost carriers like *Indigo* or *GO FIRST*.

## 💻 Technologies Used
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Linear Models, Ensembles, Preprocessing, Metrics), XGBoost

## 🚀 How to Run the Project
1. Clone the repository:
   ```bash
   git clone [https://github.com/AndKac600224/flightIndia.git](https://github.com/AndKac600224/flightIndia.git)
   ```
2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```
3. Open and execute the Jupyter Notebook `FlightIndia.ipynb`.
