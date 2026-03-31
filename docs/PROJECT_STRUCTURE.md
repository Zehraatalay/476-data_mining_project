# 📁 Project Structure

This repository contains the final, cleaned version of the retail forecasting project. Only the final pipeline and validated components are included.

---

## Directory Overview

### data/

Contains all datasets used in the project.

- raw/ → original Kaggle files  
- processed/ → cleaned, split, and model-ready datasets  

---

### scripts/

Contains all executable scripts used in the pipeline.

Includes:

- data integration  
- preprocessing and feature engineering  
- temporal splitting  
- model training scripts  
- utility modules  

These scripts represent the final, leakage-aware pipeline.

---

### outputs/

Contains generated results:

- model outputs  
- evaluation metrics  
- predictions  
- intermediate experiment results  

---

### figures/

Contains visualizations used in the report:

- time-series plots  
- correlation heatmaps  
- training curves  
- distribution plots  

---

### docs/

Contains documentation files explaining:

- pipeline design  
- model choices  
- reproducibility  
- experiment setup  

---

### report/

Contains the final academic report submitted for the course.

---

## Design Principles

The repository is structured according to the following principles:

- clarity → only final, relevant files are included  
- modularity → scripts are separated by functionality  
- reproducibility → pipeline can be rerun from scratch  
- consistency → all experiments follow the same structure  

---

## Important Note

All intermediate experiments, failed attempts, and temporary files have been removed.  
The repository reflects only the **final validated workflow**.