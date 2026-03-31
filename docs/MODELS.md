# Model Descriptions

## Ridge Regression
Ridge Regression is used as a linear baseline. It provides a simple reference point for comparing more flexible nonlinear methods.

## LightGBM
LightGBM is the main tree-based model in this project. It is well suited for structured tabular data and can model nonlinear feature interactions efficiently.

## Tuned LightGBM
This is the final best-performing model. It extends the standard LightGBM setup with:
- feature selection using ExtraTrees
- top-k feature comparison
- grid-based hyperparameter tuning

## LSTM
The standard LSTM model is used to test whether explicit sequence modeling improves forecasting performance.

## Advanced LSTM
The advanced LSTM model includes:
- stacked LSTM layers
- store embeddings
- family embeddings
- additional engineered features

This model was included to evaluate whether increased deep learning complexity improves results.