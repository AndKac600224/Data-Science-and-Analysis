# Student Performance Analysis: Clustering & Prediction

![Project Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)
![R](https://img.shields.io/badge/Language-R-blue?style=flat-square)
![Machine Learning](https://img.shields.io/badge/Focus-Clustering%20%26%20kNN-orange?style=flat-square)
![Institution](https://img.shields.io/badge/Institution-AGH%20UST-red?style=flat-square)

> **ℹ️ Language Note / Uwaga Językowa:**
> While this `README` provides a professional overview in English, the core analysis report (`.Rmd`, `.html`) and comments within the source code are written in **Polish**.
> *Główny raport analityczny oraz komentarze w kodzie zostały sporządzone w języku polskim.*

---

## Executive Summary

This project conducts a rigorous statistical analysis of factors influencing academic performance using a dataset of 10,000 student records. The study applies both **Unsupervised Learning** (to discover hidden patterns in student behavior) and **Supervised Learning** (to predict future performance scores).

Developed as part of the **Data Analysis & Machine Learning** curriculum at **AGH University of Cracow**, this project demonstrates the application of clustering algorithms (k-Means, Hierarchical) and regression models (k-Nearest Neighbors) to solve educational data mining problems.

## Objectives

* **Pattern Recognition:** To identify distinct groups of students based on study habits and lifestyle choices using clustering techniques.
* **Methodological Comparison:** To evaluate different linkage methods in hierarchical clustering (Ward, Complete, Average, Single) using statistical metrics (Silhouette, Dunn Index).
* **Performance Prediction:** To build and validate a kNN model capable of estimating the `Performance Index` based on behavioral features.

## Data Source & Structure

The analysis is based on the **Student Performance Dataset**, a synthetic dataset widely used for benchmarking regression and clustering algorithms.
* **Origin:** [Kaggle: Student Performance Prediction](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)
* **Size:** 10,000 observations.
* **Features:**
    * `Hours Studied`: Number of hours spent studying.
    * `Previous Scores`: Scores from previous tests.
    * `Extracurricular Activities`: Binary (Yes/No).
    * `Sleep Hours`: Average daily sleep duration.
    * `Sample Question Papers Practiced`: Number of practice papers solved.
    * **Target:** `Performance Index` (Integer 10-100).

## Methodology

The project follows a structured data science workflow implemented in **R**:

### 1. Preprocessing & EDA
* **Data Cleaning:** Verification of missing values (`NA`) and data types.
* **Feature Engineering:** Encoding binary categorical variables (`Extracurricular Activities`) for Euclidean distance calculations.
* **Visualization:** Distribution analysis using Boxplots to detect outliers and understand variable spread.
* **Scaling:** Standardization (`scale()`) of numerical variables to ensure equal weight in distance-based algorithms.

### 2. Unsupervised Learning (Clustering)
The study explored how students naturally group together without using the target label.
* **Algorithms:**
    * **k-Means Clustering:** Iterative centroid-based grouping.
    * **Hierarchical Clustering:** Comparison of linkage methods (*Ward.D2, Complete, Average, Single*).
* **Validation Metrics:**
    * **Silhouette Width:** Measuring how similar an object is to its own cluster (cohesion) compared to other clusters (separation).
    * **Dunn Index:** Ratio of the smallest distance between observations not in the same cluster to the largest intra-cluster distance.

### 3. Supervised Learning (Prediction)
* **Algorithm:** **k-Nearest Neighbors (kNN)** Regression.
* **Training:** 80/20 Train-Test split.
* **Optimization:** Selection of optimal `k` parameter.
* **Evaluation:** RMSE (Root Mean Square Error) and MSE (Mean Square Error).

## Key Insights & Results

* **The "Clustering Paradox":**
    
    The analysis revealed a distinct mathematical separation of students into two major clusters (confirmed by a high Dunn Index). However, detailed profiling showed that this separation was driven almost exclusively by the binary variable `Extracurricular Activities`. Surprisingly, the average `Performance Index` between these two distinct groups was nearly identical (~54.7 vs ~55.4).
    * *Business Conclusion:* While the algorithm successfully segregated students, "Extracurricular Activities" alone is not a differentiator for academic success in this specific dataset.

* **Prediction Accuracy:**
    The kNN model achieved high accuracy in predicting the `Performance Index`, proving that the combination of continuous variables (`Previous Scores`, `Hours Studied`) has strong predictive power, even if they didn't form distinct isolated clusters in lower dimensions.

* **Best Clustering Method:**
    Based on the **Dunn Index** and **Silhouette Coefficient**, the **Ward.D2** method proved to be the most effective for Hierarchical Clustering on this data structure.

## Tech Stack

The project utilizes the **R** statistical programming environment.

```r
# Key Libraries Used
library(dplyr)      # Data Manipulation
library(cluster)    # Clustering Algorithms
library(clValid)    # Cluster Validation (Dunn Index)
library(caret)      # Machine Learning Workflow (kNN)
library(MLmetrics)  # Model Evaluation (RMSE, MSE)
```

## How to Run

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/AndKac600224/Data-Science-and-Analysis.git
    cd Data-Science-and-Analysis/Project_Student_Performance_ML
    ```
2.  **Load the Project:**
    Open `project_code.R` in **RStudio**.
3.  **Install Dependencies:**
    ```r
    install.packages(c("dplyr", "cluster", "clValid", "caret", "MLmetrics"))
    ```
4.  **Execute:**
    Run the script or **knit the `projekt.Rmd` file to generate the full HTML report**.

## Repository Structure

| File | Description |
| :--- | :--- |
| `project_code.R` | **Source Code**. Raw R script containing the full analysis pipeline. |
| `projekt.Rmd` | **R Markdown**. The source for the generated academic report. |
| `projekt.md` | **Final Report**. Compiled MD document with plots and commentary (PL). |
| `Student_Performance.csv` | **Dataset**. The input data used for analysis. |
| `students_performance.txt` | **Description**. Describtion of dataset. |

## Author

**Kacper Andrzejewski**
* **Institution:** AGH University of Cracow
* **Focus:** Data Science & Machine Learning
* **Contact:** [GitHub Profile](https://github.com/AndKac600224)

---
*Project created for educational purposes.*
```
