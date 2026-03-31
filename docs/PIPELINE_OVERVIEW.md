# Pipeline Overview

The project follows a leakage-aware end-to-end time-series forecasting pipeline.

## Step 1: Data Integration
The raw sales table is merged with:
- store metadata
- oil prices
- holiday data
- transaction counts

## Step 2: Deterministic Preprocessing
The following preprocessing steps are applied:
- missing value handling
- holiday cleanup
- calendar feature generation
- cyclical feature generation
- promotion transformation

## Step 3: Temporal Splitting
The dataset is split chronologically:
- train: 2013-2015
- validation: 2016
- test: 2017

A separate expanding-window time-series cross-validation setup is also created.

## Step 4: Modeling-Ready Transformation
Leakage-safe preprocessing is applied separately for:
- train to validation evaluation
- train plus validation to test evaluation
- time-series CV folds

## Step 5: Model Training
The following models are trained and evaluated:
- Ridge Regression
- LightGBM
- Tuned LightGBM
- LSTM
- Advanced LSTM

## Step 6: Evaluation
All models are evaluated using:
- MAE
- RMSE
- MAPE
- RMSLE

RMSLE is treated as the primary metric due to skewness and zero-heavy sales.