![Multimax Banner](/images/banner.png)

# MultiMax
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

MultiMax is a [QuantConnect](https://www.quantconnect.com/) app that combines the insights of machine learning models with the interpretability and control of multi-factor models. By using [SHAP values](https://dl.acm.org/doi/10.5555/3295222.3295230) for feature selection, MultiMax trains multi-factor models with the best possible subset of fundamental and technical features.

MultiMax trains multi-factor models using the following algorithm:
```
1. Train a machine learning model to predict stock prices using economic factors.
2. Calculate SHAP values for each prediction of the machine learning model.
3. Sort features by their SHAP value's mean magnitude. Features with a higher magnitude are more important.
4. Perform linear regression on the N most important features, training a multi-factor model.
5. Use the multi-factor model to trade using a value investing strategy.
```

Random forests (RFs), gradient boosting machines (GBMs), and deep neural networks (DNNs) are all available as base models for this construction process.

## Backtesting Results
Constructed models can achieve reasonably high profits with great consistency. DNN-based MultiMax models can even achieve better Sharpe ratios than published multi-factor models, like Fama-French 3-factor and Carhart 4-factor models.

The following results were taken over 5 trials of each model:

| Model             | Type     | Sharpe Ratio | Max. Drawdown (%) | Compounding Annual Return (%) |
| ----------------- | -------- | ------------ | ----------------- | ----------------------------- |
| Buy and Hold      | Baseline | 0.224	      | 55.50             |	6.150                         |
| Fama-French       | Baseline | 0.606        | 77.10             |	21.26                         |
| **Carhart**       | Baseline | 0.643	      | 75.60	            | **23.12**                     |
| **RF-based MFM**  | Multimax | 0.538	      | **30.54**         | 10.73                         |
| GBM-based MFM     | Multimax | 0.492	      | 39.60	            | 10.18                         |
| **DNN-based MFM** | Multimax | 0.779	      | 39.16	            | 18.90                         |

![A chart of the returns of MultiMax models](/images/returns_chart.png)

## Features 
In MultiMax, users can:
- Establish a trading environment with their own dataset and parameters.
- Train a random forest, gradient boosting machine or DNN to predict stock prices with their desired hyperparameters.
- Construct a multi-factor model using their ML model’s N most important factors.
- Perform backtesting using their multi-factor model with the TopKDrop strategy.
- Save their models as pre-trained trading objects (PTOs) that can be shared and reused for live trading.
- Analyse and visualise the behaviour, accuracy, profitability, and risk of their ML and multi-factor models.

## Usage
Setting up MultiMax is easy: copy all files from the [app directory](./app) into a QuantConnect project!

To run a backtest, open the [main app](./app/main.py) and press the "Play" button. 

![Debug logs for a backtest](/images/run_main.png)
Training output will appear in QuantConnect's debug logs, while the final results will be available as a backtest on your Quantconnect account.

![Debug logs for a backtest](/images/backtest_logs.png)
![A chart of showing the results of backtesting](/images/backtest_graph.png)

To analyse an existing backtest, open the [results tool](./app/results.ipynb), enter the key of the saved pre-trained trading object (PTO), and press the "Run All" button.

![Setting up the results tool](/images/setup_results.png)
![Running the results tool](/images/run_results.png)

Visualisations of the results will appear in the rest of the notebook.

![A graph from our results tool](/images/coefficients_plot.png)


## Parameters
The following parameters can be configured to control the main app's training and backtesting environment:

### Global Parameters

- `self.method`: MultiMax's modelling method:
  - Automatic: Train an ML model and use its SHAP values for feature selection.
  - Manual: Specify the factors and betas of the multi-factor model manually.
  - Pretrained: Load a saved Pretrained Trading Object (PTO) and use that.

- `self.source`: A Dropbox link to the training and backtesting data. Note that this must have the following structure:
  - Stock Name, Date (dd/MM/yyyy), Factor 1, Factor 2 ...
  - See our [dataset directory](./dataset) for more details on how to create a dataset.

- `self.num_features`: The number of features in our multi-factor model (N).

### Trading Parameters

- `self.start_date`: The date our MFM should start trading.
- `self.end_date`: The data our MFM should finish trading.
- `self.start_cash`: Our portfolio's initial value (in $).
- `self.rebalance_days`: The number of days between portfolio reallocations (D).
- `self.num_long`: The number of stocks our trading strategy holds at one time (K).

### Automatic Mode

- `self.model_type`: The ML model to train and perform feature selection from:
  - RF: Random forest
  - GBM: Gradient boosting machine
  - DNN: Deep neural network
- `self.num_trees`: The number of trees/boosting rounds for RF or GBM models.
- `self.max_depth`: The maximum depth of trees in RF and GBM models.
- `self.batch_size`: The batch size for DNN models.
- `self.epochs`: The number of training epochs from DNN models.

### Manual Mode
- `self.factors`: The names of the factors in our multi-factor model.
- `self.betas`: The coefficients of the factors in our multi-factor model.

### Pretrained Mode
- `key`: The name of our pre-trained trading object (PTO).

## Further Reading 
More information on MultiMax, please read our [research paper](link).
