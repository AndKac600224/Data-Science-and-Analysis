# ERA5 Climate Reanalysis & Predictive Modeling (2010–2024)

![Project Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Machine Learning](https://img.shields.io/badge/Focus-EDA%20%26%20Predictive%20Modeling-orange?style=flat-square)
![Institution](https://img.shields.io/badge/Institution-AGH%20UST-red?style=flat-square)

> **ℹ️ Language Note / Uwaga Językowa:**
> While this documentation is presented in English for professional accessibility, the source code (`.ipynb`) and the comprehensive engineering report (`.docx`) are written in **Polish**.
> *Kod źródłowy oraz szczegółowe sprawozdanie inżynierskie znajdują się w języku polskim.*

---

## Executive Summary

This project delivers a comprehensive **Exploratory Data Analysis (EDA)** and **Machine Learning (ML)** study focusing on meteorological trends in **Lublin, Poland**, over a 15-year period (2010–2024). Utilizing high-fidelity reanalysis data from the **ERA5 dataset** (Copernicus Climate Change Service), the project aims to identify climatic anomalies and build robust regression models for temperature forecasting.

The study was conducted as part of the advanced **Machine Learning curriculum** at the **AGH University of Science and Technology (Akademia Górniczo-Hutnicza)** in Kraków. It demonstrates end-to-end data science capabilities: from raw data ingestion and feature engineering to model evaluation and environmental impact assessment.

## Business & Scientific Objectives

* **Climatic Trend Analysis:** To statistically validate the impact of global warming on local weather patterns (identifying warming trends and temperature inversions).
* **Predictive Modeling:** To develop algorithms capable of estimating Air Temperature (`t2m`) based on atmospheric pressure, wind vectors, and solar radiation.
* **Urban Planning Insights:** To provide data-driven recommendations for city planning (Blue-Green Infrastructure) to mitigate Urban Heat Islands.

## Data Architecture

The dataset is derived from **ECMWF ERA5 Reanalysis**, providing a consistent view of the atmosphere over decades.

* **Temporal Scope:** January 2010 – December 2024.
* **Granularity:** 6-hour intervals (00:00, 06:00, 12:00, 18:00).
* **Format:** Processed from GRIB to structured CSV.
* **Key Features:**
    * `t2m`: 2-meter Temperature (Target Variable).
    * `sp` / `msl`: Surface Pressure / Mean Sea Level Pressure.
    * `u10` / `v10`: 10-meter Wind Components (Zonal/Meridional).
    * `tp`: Total Precipitation.
    * `tcc`: Total Cloud Cover.
    * `ssrd`: Surface Solar Radiation Downwards.

## Methodology & Tech Stack

The project follows the standard **CRISP-DM** lifecycle tailored for scientific research.

### 1. Data Engineering & Preprocessing
* **Unit Standardization:** Conversion of Kelvin to Celsius (`t2m`) and Pascals to hPa (`sp`).
* **Vector Transformation:** Engineered a scalar `Wind Speed` feature from `u10` and `v10` vectors using Euclidean norm.
* **Temporal Features:** Extracted cyclical time features (Year, Month, Day, Hour) to capture seasonality.
* **Outlier Management:** Applied Interquartile Range (IQR) method to detect and handle anomalies in pressure and wind readings.

### 2. Exploratory Data Analysis (EDA)
* **Correlation Matrices:** Utilized Pearson correlation heatmaps to identify multicollinearity (e.g., strong `sp` vs. `msl` relationship) and feature importance.
* **Distribution Analysis:** KDE plots and Histograms to assess normality (Gaussian distribution checks).
* **Time-Series Decomposition:** Visualized long-term temperature progression and seasonal fluctuations.

### 3. Machine Learning Strategy
The core task was **Regression Analysis** to predict Temperature (`t2m`). The dataset was split into training and testing sets with **StandardScaler** applied for normalization.

**Models Implemented & Evaluated:**
* Linear Regression (Baseline)
* Decision Tree Regressor
* **Random Forest Regressor** (Ensemble)
* **Gradient Boosting Regressor** (Ensemble - Best Performer)
* Support Vector Regressor (SVR)

**Evaluation Metrics:**
* $R^2$ (Coefficient of Determination)
* MSE (Mean Squared Error)
* MAE (Mean Absolute Error)

## Key Insights & Results

* **Model Performance:** Ensemble methods (Random Forest, Gradient Boosting) significantly outperformed linear models, capturing the non-linear interactions between solar radiation (`ssrd`), cloud cover (`tcc`), and temperature.
* **Climate Observations:** The analysis confirmed a rising temperature trend in the studied period, aligning with global climate change models.
* **Feature Importance:** `ssrd` (Solar Radiation) and `d2m` (Dew Point) were identified as the most critical predictors for air temperature.

## Technologies

The project utilizes the Python Data Science ecosystem:

```python
# Core Analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning (Scikit-Learn)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
```

## Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/YourUsername/fitness-stream-project.git
    cd fitness-stream-project
    ```
2.  **Prerequisites:**
    Install required packages:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
3.  **Run the Analysis:**
    Launch Jupyter Notebook:
    ```bash
    jupyter notebook projektEDA_ML_rob.ipynb
    ```

## 📄 Repository Structure

| File | Description |
| :--- | :--- |
| `projektEDA_ML_rob.ipynb` | **Main Kernel**. Contains data cleaning, EDA visualizations, and ML modeling code. |
| `projektEDA_ML.docx` | **Engineering Report (PL)**. Full academic documentation, theoretical background, and detailed conclusions. |
| `dane_grib_era5.csv` | **Dataset**. Processed ERA5 meteorological data used for training. |

## 👤 Author

**Kacper Andrzejewski**
* **Institution:** AGH University of Science and Technology
* **Course:** Machine Learning & Data Analysis
* **Contact:** [GitHub Profile](https://github.com/AndKac600224)

---
*© 2026 Kacper Andrzejewski. Project created for educational purposes.*
```
