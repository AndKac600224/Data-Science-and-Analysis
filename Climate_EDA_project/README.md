# ERA5 Climate Analysis: Trends & ML Feasibility Study (2010–2024)

![Project Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Focus](https://img.shields.io/badge/Focus-Advanced%20EDA%20%26%20Strategy-orange?style=flat-square)
![Institution](https://img.shields.io/badge/Institution-AGH%20UST-red?style=flat-square)
![Data Source](https://img.shields.io/badge/Data-CDS%20Copernicus%20ERA5-green?style=flat-square)

> **ℹ️ Language Note / Uwaga Językowa:**
> While this documentation is presented in English for professional accessibility, the source code (`.ipynb`) and the comprehensive engineering report (`.docx`) are written in **Polish**.
> *Kod źródłowy oraz szczegółowe sprawozdanie inżynierskie znajdują się w języku polskim.*

---

## Executive Summary

This project delivers a rigorous **Exploratory Data Analysis (EDA)** of meteorological conditions in **Lublin, Poland**, spanning a 15-year period (2010–2024). Utilizing high-fidelity **ERA5 Reanalysis data** from the **Copernicus Climate Change Service (CDS)**, the study identifies critical climatic trends, anomalies, and correlations.

Beyond statistical analysis, the accompanying engineering report provides a **theoretical framework and feasibility assessment** for implementing Machine Learning models. It outlines how historical data patterns can be leveraged to build predictive systems for temperature forecasting and extreme weather alert systems in future iterations.

## Business & Scientific Objectives

* **Climatic Trend Validation:** Statistical confirmation of global warming impacts on the local microclimate, including the frequency of temperature inversions.
* **Data-Driven Urban Planning:** Providing empirical evidence to support **Blue-Green Infrastructure** initiatives (e.g., increasing biologically active areas) to mitigate Urban Heat Islands.
* **ML Readiness Assessment:** Evaluating data quality and feature correlations (e.g., Solar Radiation vs. Temperature) to propose a robust architecture for future regression modeling (Random Forest/Gradient Boosting).

## Data Source & Architecture

The dataset is derived from the **ERA5 hourly data on single levels**, provided by the **Copernicus Climate Data Store (CDS)**.

* **Source:** [Copernicus Climate Data Store (ERA5)](https://cds.climate.copernicus.eu/)
* **Temporal Scope:** January 2010 – December 2024.
* **Granularity:** 6-hour intervals (Synoptic hours: 00:00, 06:00, 12:00, 18:00).
* **Format:** Processed from raw GRIB to structured CSV.
* **Key Variables:**
    * `t2m`: 2-meter Temperature (Target Variable).
    * `sp` / `msl`: Surface Pressure / Mean Sea Level Pressure.
    * `u10` / `v10`: 10-meter Wind Components (Vector decomposition).
    * `tp`: Total Precipitation.
    * `tcc`: Total Cloud Cover.
    * `ssrd`: Surface Solar Radiation Downwards.

## Methodology

The project follows the **Data Understanding** and **Data Preparation** phases of the CRISP-DM lifecycle.

### 1. Data Engineering & Preprocessing
* **Unit Standardization:** Conversion of Kelvin to Celsius (`t2m`) and Pascals to hPa (`sp`).
* **Feature Extraction:**
    * Calculation of scalar `Wind Speed` from `u10` and `v10` vectors.
    * Extraction of cyclical temporal features (Year, Month, Day, Hour) to capture seasonality.
* **Quality Assurance:** Detection of outliers using the Interquartile Range (IQR) method, specifically for pressure anomalies and wind extremes.

### 2. Advanced Exploratory Data Analysis (EDA)
* **Correlation Mapping:** Pearson correlation heatmaps revealed strong dependencies between Solar Radiation (`ssrd`), Dew Point (`d2m`), and Air Temperature (`t2m`), establishing a solid baseline for future feature selection.
* **Distribution Analysis:** Histograms and KDE plots verified the non-Gaussian distribution of precipitation and cloud cover.
* **Trend Analysis:** Visual decomposition of time series confirmed a rising mean temperature trend over the 2010–2024 interval.

### 3. Proposed ML Architecture (Feasibility Study)
*As detailed in the final section of the Engineering Report (`.docx`), a predictive strategy was designed based on the EDA findings.*

* **Proposed Goal:** Regression analysis to forecast `t2m`.
* **Recommended Algorithms:** Ensemble methods (**Random Forest**, **Gradient Boosting**) are suggested over linear models due to the non-linear nature of atmospheric interactions (e.g., cloud cover vs. radiation).
* **Expected Metrics:** The study defines success criteria based on $R^2$ and MAE (Mean Absolute Error) for future implementation.

## Key Findings

* **Warming Trend:** A statistically significant increase in average annual temperatures was observed, aligning with global climate change models.
* **Urban Heat Island Mitigation:** The analysis supports the hypothesis that increasing urban vegetation and water bodies (evapotranspiration) is critical for temperature regulation.
* **Data Quality for ML:** The dataset demonstrates high consistency and strong signal-to-noise ratio in key features (`ssrd`, `d2m`), making it highly suitable for training supervised learning models.

## Technologies

The project utilizes the Python Data Science ecosystem for analysis and visualization:

```python
# Core Data Processing
import pandas as pd
import numpy as np

# Visualization & Reporting
import matplotlib.pyplot as plt
import seaborn as sns
```

## Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/AndKac600224/Data-Science-and-Analysis.git](https://github.com/AndKac600224/Data-Science-and-Analysis.git)
    cd Data-Science-and-Analysis/Climate_EDA_project
    ```
2.  **Prerequisites:**
    Install required packages:
    ```bash
    pip install pandas numpy matplotlib seaborn
    ```
3.  **Run the Analysis:**
    Launch Jupyter Notebook:
    ```
    jupyter notebook projektEDA_ML_rob.ipynb
    ```

## Repository Structure

| File | Description |
| :--- | :--- |
| `projektEDA_ML_rob.ipynb` | **Main Kernel**. Code for data cleaning, preprocessing, and extensive EDA visualizations. |
| `projektEDA_ML.docx` | **Engineering Report (PL)**. Contains the full academic study, theoretical background, conclusions, and the **Machine Learning implementation proposal**. |
| `dane_grib_era5.csv` | **Dataset**. Processed ERA5 meteorological data used for the analysis. |

## Author

**Kacper Andrzejewski**
* **Institution:** AGH University of Science and Technology
* **Project Type:** EDA & Machine Learning Feasibility Study
* **Contact:** [GitHub Profile](https://github.com/AndKac600224)

---
*Attribution: This project contains modified Copernicus Climate Change Service information [2025]. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.*
```
