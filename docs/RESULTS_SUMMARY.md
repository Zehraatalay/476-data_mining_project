# Results Summary

## Best Model
The tuned LightGBM model achieved the best overall performance.

## Main Observations

- Ridge Regression was too limited for the nonlinear structure of the data.
- LightGBM performed strongly because the dataset is primarily structured tabular data.
- LSTM showed overfitting.
- Advanced LSTM became unstable and generalized poorly.
- Feature engineering and feature selection played a major role in performance.

## Final Interpretation
The experiments suggest that retail forecasting in this dataset depends more on heterogeneous structured feature interactions than on pure sequential dependence.