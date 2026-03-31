# 📊 Leakage-Aware Retail Sales Forecasting

This project focuses on large-scale retail demand forecasting using time-series data mining techniques. It implements a leakage-aware pipeline and evaluates multiple modeling approaches under a consistent experimental framework.

---

## 🚀 Project Overview

Retail demand forecasting is a challenging problem due to:

- temporal dependencies  
- data sparsity (≈31% zero sales)  
- non-stationarity  
- heterogeneous behavior across stores and product families  

This project addresses these challenges by building a **robust, leakage-aware forecasting pipeline** and comparing different model families.

---

## 🎯 Objectives

- Prevent data leakage in time-series preprocessing  
- Build a reproducible forecasting pipeline  
- Compare different modeling approaches:  
  - Ridge Regression (baseline)  
  - LightGBM (tree-based)  
  - LSTM (sequence-based)  
- Evaluate models under consistent temporal splits  

---

## 📦 Dataset

- Source: Kaggle Store Sales Forecasting  
- Records: ~3 million  
- Features: 33  
- Time Range: 2013–2017  

### Key Characteristics
- zero-heavy target (≈31%)  
- strong seasonality  
- promotion-driven demand  
- heterogeneous store behavior  

---

## ⚙️ Methodology

### 🔹 Preprocessing
- leakage-aware time-based splitting  
- deterministic feature engineering  
- missing value handling  
- log transformation for skewed variables  

### 🔹 Feature Engineering
- calendar features (month, weekday, etc.)  
- cyclical encoding (sin/cos)  
- event-based features (holiday, payday)  
- sparsity-aware statistics  

### 🔹 Models
- Ridge Regression  
- LightGBM  
- Tuned LightGBM  
- LSTM  
- Advanced LSTM  

### 🔹 Evaluation Metrics
- MAE  
- RMSE  
- MAPE  
- RMSLE (primary metric)  

---

## 📊 Results

- Tuned LightGBM achieved the best performance  
- LSTM models showed overfitting and instability  
- Feature engineering significantly improved performance  
- Model complexity did not guarantee better results  

---

## 🧠 Key Insight

> Structured tabular interactions dominate over pure sequential dependencies in retail forecasting.

---

## 📁 Project Structure

scripts/     → data pipeline and model implementations  
data/        → raw and processed datasets  
outputs/     → experiment outputs and results  
figures/     → visualizations used in the report  
docs/        → project documentation  
report/      → final academic report  

---

## ▶️ How to Run

Run the full pipeline:

python scripts/main_pipeline.py

Train models individually:

python scripts/ridge_model.py  
python scripts/lightgbm_model.py  
python scripts/tuned_lightgbm_model.py  
python scripts/lstm_model.py  
python scripts/advanced_lstm_model.py  

---

## 🔁 Reproducibility

- time-based splitting is preserved  
- preprocessing is deterministic  
- no data leakage  
- random seeds are fixed where applicable  

---

## 📄 Report

report/zehra_atalay_data_mining_report.pdf

---

## 👩‍💻 Author

Zehra Atalay  
TOBB University of Economics and Technology  

---

## ⭐ Notes

This repository contains only the **final, cleaned version** of the project.  
All intermediate experiments and discarded approaches were removed to ensure clarity and reproducibility.
