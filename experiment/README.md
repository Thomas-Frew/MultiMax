# Experiment Tools

This folder contains the tools to test MultiMax's model-creation method independent of QuantConnect's platform.

## Tools

- [Cross-Validation](cross_validation.ipynb): Selects optimal parameters for a Random Forest or Gradient Boosting Machine based on a grid of combinations.
- [Experiments](experiments.ipynb): Trains ML models to predict stock prices, then uses SHAP values to select features for effective multi-factor models. Calculates MAE and similarity between the models. Visualises the selected features and their ratios.
- [Backtesting Stats](backtesting_statistics.ipynb): Calculates the average Sharpe ratio, drawdown and compounding annual return from a set of backtests. Data is manually copied and entered from QuantConnect.
