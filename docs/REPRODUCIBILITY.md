# 🔁 Reproducibility

This project was designed to ensure that all results can be reproduced in a consistent and leakage-free manner.

---

## Core Principles

### 1. Time-Series Awareness

- no random shuffling  
- strictly chronological splits  
- future data is never used in training  

---

### 2. Leakage Prevention

- preprocessing is applied separately for each split  
- scaling is fitted only on training data  
- feature engineering does not use future information  

---

### 3. Deterministic Pipeline

- preprocessing steps are deterministic  
- transformations produce consistent outputs  
- same inputs always produce the same datasets  

---

### 4. Structured Workflow

The project follows a fixed pipeline:

1. data integration  
2. preprocessing  
3. feature engineering  
4. temporal splitting  
5. model-ready transformation  
6. model training  
7. evaluation  

---

## How to Reproduce Results

### Step 1: Install Dependencies

pip install -r requirements.txt

### Step 2: Run Data Pipeline

python scripts/main_pipeline.py

### Step 3: Train Models

python scripts/ridge_model.py  
python scripts/lightgbm_model.py  
python scripts/tuned_lightgbm_model.py  
python scripts/lstm_model.py  
python scripts/advanced_lstm_model.py  

---

## Notes

- Some experiments use sampling to reduce computational cost  
- all models are evaluated on the same temporal splits  
- RMSLE is used as the primary metric  

---

## Conclusion

The project ensures reproducibility by combining:

- leakage-aware design  
- deterministic preprocessing  
- structured pipeline execution  

This allows results to be replicated and validated reliably.