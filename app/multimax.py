
# region Imports

# QuantConnect
from AlgorithmImports import *

# Data processing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# Randomisation
from random import choice, seed

# LASSO regressor (baseline)
from sklearn.linear_model import Lasso

# Random forest
from sklearn.ensemble import GradientBoostingRegressor

# Gradient boost
from sklearn.ensemble import RandomForestRegressor

# Machine learning
import tensorflow as tf
from tensorflow import keras
import shap

# Serialising
import joblib
import base64
import pickle
import io

# endregion

# region Multi-Factor Model Generator

class ModelGenerator:

    def __init__(self, algo, source, classification):

        # Store the QC algorithm instnce
        self.qc = algo

        # Store the dataset instance
        self.dataset = None
        self.source = source

        # Store the models
        self.model = None
        self.lr_model = None

        # Store predictions
        self.ml_preds = None
        self.mfm_pred = None


        # Log that we have started up successfully
        self.qc.Log("Multi-Factor Model Generator created")

        # Get random seeds
        seed()
        self.seed_1 = choice(range(1, 10000))
        self.seed_2 = choice(range(1, 10000))

        # Store the train and test sets
        self.X_test = None
        self.X_train = None
        self.y_test = None
        self.y_train = None

        # Load the dataset
        self.GetDataset(source, classification)

    def GetDataset(self, source, classification):

        # Load dataset
        qb = QuantBook()
        data = qb.Download(source)
        arr = [x.strip().split(',') for x in data.split('\n')]
        df = pd.DataFrame(arr)

        # Define the columns of the dataset
        df.columns = df.iloc[0] 
        df = df[1:]

        # Clean up the source data
        df = df.iloc[:, 2:]
        df = df[:-1].astype(float)

        # Drop NaNs and infs
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)

        # Save the dataset
        self.dataset = df
        self.qc.Debug(f"Successfully loaded dataset of shape {df.shape}")

        # If we are predicting the direction of a stock's movement...
        if (classification):

            # Define a function to map values to 1 or 0
            def mapDirection(value):
                return 1 if value >= 0 else 0

            # Apply the mapping function to the 'numbers' column
            df["Return"] = df["Return"].astype(float).apply(mapDirection)

        # Split into test and train sets
        y = df["Return"].astype(float)
        X = df.drop(["Return"], axis=1).astype(float)

        # Normalising the input space
        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Split the data into training and testing sets (9:1) and store them
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=self.seed_2)
        self.qc.Debug(f"Successfully loaded train set {self.X_train.shape} and test set {self.X_test.shape}")

        self.labels = self.dataset.columns[1:]

    def PrintModel(self, model):

        # Prints out all factors from the model, and their coefficients
        for key, value in model.items():
            self.qc.Debug(f"{key}: {value:.2f}")

    def MakeDataStats(self):

        # Create source stats
        train = {
            'num_features': len(self.labels),
            'split_ratio': len(self.y_train)/len(self.y_test),
            'mean': np.mean(self.y_train),
            'std': np.std(self.y_train)
        }

        # Create prediction stats
        ml_stats = {
            'mean': np.mean(self.ml_preds),
            'std': np.std(self.ml_preds),
            'error': np.mean(abs(self.y_test - self.ml_preds)),
            'alpha': np.std(self.y_train) - np.mean(abs(self.y_test - self.ml_preds)),
            'preds': list(self.ml_preds)
        }

        mfm_stats = {
            'mean': np.mean(self.lr_preds),
            'std': np.std(self.lr_preds),
            'error': np.mean(abs(self.y_test - self.lr_preds)),
            'alpha:': np.std(self.y_train) - np.mean(abs(self.y_test - self.lr_preds)),
            'preds': list(self.lr_preds),
            'distance': np.mean(abs(self.ml_preds - self.lr_preds))
            
        }
    
        # Package into data
        data_stats = {
            'train': train,
            'ML': ml_stats,
            'MFM': mfm_stats
        }

        return data_stats

    def TrainLasso(self, alpha, num_features):

        self.qc.Debug(f"Training LASSO regressor with alpha {alpha}")

        # Create a LASSO
        self.model = Lasso(alpha=alpha, fit_intercept=False, random_state=self.seed_1, max_iter=2000)
        self.model.fit(self.X_train, self.y_train)

        # Get data the lasso coefficients
        coeffs = self.model.coef_

        # Create a dictionary representing the factors and their coefficients
        features = dict(zip(self.labels, coeffs))
        features = dict(sorted(features.items(), key=lambda item: -abs(item[1])))

        # Extract the top N features to create a multi-factor model
        features = {A:N for (A,N) in [x for x in features.items()][:num_features]}

        # Test MAE
        self.ml_preds = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, self.ml_preds)
        self.qc.Debug(F"LASSO complete with MAE {mae}.")

        # Print the multi-factor model
        self.PrintModel(features)

        return features

    def TrainRandomForest(self, trees, depth, num_features):
        
        self.qc.Debug(f"Training Random Forest regressor with {trees} trees {depth} levels deep")

        # Create a Random Forest model and train it
        self.model = RandomForestRegressor(n_estimators=trees, max_depth=depth, random_state=self.seed_2)
        self.model.fit(self.X_train, self.y_train)

        # Test the base model
        self.ml_preds = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test,self.ml_preds)

        self.qc.Debug(F"Random Forest trained with MAE {mae}.")

        # Do SHAP on random forest
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test)

        # Sort by SHAP values and extract the N most important features
        feature_importance = np.abs(shap_values).mean(axis=0)
        selected_features = self.X_train.columns[np.argsort(feature_importance)[::-1][:num_features]]

        # Prepare the data with the selected features
        X_train_selected = self.X_train[selected_features]
        X_test_selected = self.X_test[selected_features]

        # Train a linear regression model with the selected features
        self.lr_model = LinearRegression()
        self.lr_model.fit(X_train_selected, self.y_train)

        # Evaluate the performance of the linear regression model
        self.lr_preds = self.lr_model.predict(X_test_selected)
        mae = mean_absolute_error(self.lr_preds, self.y_test)

        self.qc.Debug(f"Linear Regression model distilled with MAE: {mae}")


        # Create the multi-factor model
        features = dict(zip(selected_features, self.lr_model.coef_))
        features = dict(sorted(features.items(), key=lambda item: -abs(item[1])))

        # Print the multi-factor model
        self.PrintModel(features)

        return features

    def TrainGradientBoost(self, trees, depth, num_features):
        
        self.qc.Debug(f"Training Gradient Boosting Machine with {trees} trees {depth} levels deep")

        # Create a Random Forest model and train it
        self.model = GradientBoostingRegressor(n_estimators=trees, max_depth=depth, random_state=self.seed_2)
        self.model.fit(self.X_train, self.y_train)

        # Calculate error
        self.ml_preds = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, self.ml_preds)

        self.qc.Debug(F"Gradient Booster complete with MAE {mae}.")


        # Do SHAP on random forest
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test)

        # Sort by SHAP values and extract the N most important features
        feature_importance = np.abs(shap_values).mean(axis=0)
        selected_features = self.X_train.columns[np.argsort(feature_importance)[::-1][:num_features]]

        # Prepare the data with the selected features
        X_train_selected = self.X_train[selected_features]
        X_test_selected = self.X_test[selected_features]

        # Train a linear regression model with the selected features
        self.lr_model = LinearRegression()
        self.lr_model.fit(X_train_selected, self.y_train)

        # Evaluate the performance of the linear regression model
        self.lr_preds = self.lr_model.predict(X_test_selected)
        mae = mean_absolute_error(self.lr_preds, self.y_test)

        self.qc.Debug(f"Linear Regression model distilled with MAE: {mae}")


        # Create the multi-factor model
        features = dict(zip(selected_features, self.lr_model.coef_))
        features = dict(sorted(features.items(), key=lambda item: -abs(item[1])))

        # Print the multi-factor model
        self.PrintModel(features)    

        return features

    def TrainDNN(self, batch_size, epochs, num_features):

        # Establish the nerual netwrok input shape
        input_shape = len(self.X_train.columns)

        # Create the DNN model (best case, MAE = 14%)
        self.model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])
        
        # Compile the model and specify the optimizer, loss function, and metrics
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

        # Print the model's structure
        self.qc.Debug("Created DNN with the following structure:")
        self.qc.Debug(self.model.summary())

        # Train the model
        self.model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=epochs, validation_data=(self.X_test, self.y_test))

        # Calculate error
        self.ml_preds = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, self.ml_preds)

        self.qc.Debug(F"DNN complete with MAE {mae}.")

        # Do SHAP on random forest
        explainer = shap.KernelExplainer(self.model, self.X_test[0:30])
        shap_values = explainer.shap_values(self.X_test[0:30])

        # Sort by SHAP values and extract the N most important features
        feature_importance = np.abs(shap_values[0]).mean(axis=0)
        selected_features = self.X_train.columns[np.argsort(feature_importance)[::-1][:num_features]]

        # Prepare the data with the selected features
        X_train_selected = self.X_train[selected_features]
        X_test_selected = self.X_test[selected_features]

        # Train a linear regression model with the selected features
        self.lr_model = LinearRegression()
        self.lr_model.fit(X_train_selected, self.y_train)

        # Evaluate the performance of the linear regression model
        self.lr_preds = self.lr_model.predict(X_test_selected)
        mae = mean_absolute_error(self.lr_preds, self.y_test)

        self.qc.Debug(f"Linear Regression model distilled with MAE: {mae}")


        # Create the multi-factor model
        features = dict(zip(selected_features, self.lr_model.coef_))
        features = dict(sorted(features.items(), key=lambda item: -abs(item[1])))

        # Print the multi-factor model
        self.PrintModel(features)       

        return features

# endregion

# region Live Data Parser

class DataSource(PythonData):

    def __init__(self): 
        self.headers = []

    def GetSource(self, config, date, isLiveMode):
        source = "[SOURCE]"
        return SubscriptionDataSource(source, SubscriptionTransportMedium.RemoteFile)

    def Reader(self, config, line, date, isLiveMode):

        # If the stock cannot be extracted, simply ignore it
        if not (line.strip()):
            return None

        # Extract the line and create a new stock instance
        data = line.split(',')
        stock = DataSource()

        try:

            # Create a stock symbol from the first entry
            stock.Symbol = Symbol.Create(data[0], SecurityType.Equity, Market.USA)
            stock.Name = str(data[0])

            # Extract the time from the second entry
            stock.Time = datetime.strptime(data[1], "%m/%d/%Y")

            # Extract all other data
            stock.Data = data[2:]
            stock.Headers = self.headers

            # Zip the data up into a dictionary with meaningful labels
            stock.Props = dict(zip(stock.Headers, stock.Data))

        except:

            # If an error is thrown, then it must be the header column. Extract and store the headers.
            if (data[0] == ""):
                self.headers = data[2:]
            
            return None

        return stock

# endregion

# region Trading Algorithm

class MultiMax(QCAlgorithm):

    def Initialize(self):
        """ Trading algorithm setup. """

        # ============ PARAMETERS ============== #

        # Trading method (manual/automatic/pretrained)
        self.method = 'Automatic'

        # Trading parameters
        self.source = "[SOURCE]"

        self.start_date = datetime(2000, 1, 1)
        self.end_date = datetime(2023, 1, 1)

        self.start_cash = 100000
        self.rebalance_days = 300
        self.num_long = 10

        # ML model parameters (LASSO, trees, DNN)
        self.alpha = 0.05

        self.num_trees = 32
        self.max_depth = 5

        self.batch_size = 32
        self.epochs = 16

        # Multi-factor model parameters
        self.num_features = 10

        # ============ PARAMETERS ============== #


        # Features, factors and betas
        self.generator = None
        self.features = None
        self.factors = None
        self.betas = None

        # Manual mode (features and betas are specified)
        if (self.method == "Manual"):

            # ============ PARAMETERS ============== #
            self.factors = ["Return on Equity (ROE)", "Revenue", "Price-to-Book (PB)"]
            self.betas = [9.09, -1.74, -8.00]

            self.features = dict(zip(self.factors, self.betas))

        # Automatic mode (an ML model is trained)
        elif (self.method == 'Automatic'):

            # ============ PARAMETERS ============== #
            self.model_type = "RF"

            self.generator = ModelGenerator(self, self.source, False)

        # Pretrained mode (loading an earlier version)
        elif (self.method == 'Pretrained'):

            # ============ PARAMETERS ============== #
            key = "[KEY]"

            # Load the string from the Object Store
            data_string = self.ObjectStore.Read(key)
            print(f"Loading serialised trading model {data_string}.")

            # Unpack the model as a dict
            model_data = json.loads(data_string)

            # Load parameters
            start_arr = model_data["setup_stats"]["start_date"].split('/')
            end_arr = model_data["setup_stats"]["end_date"].split('/')

            self.start_date = datetime(int(start_arr[2]), int(start_arr[1]), int(start_arr[0]))
            self.end_date = datetime(int(end_arr[2]), int(end_arr[1]), int(end_arr[0]))

            self.start_cash = model_data["setup_stats"]["start_cash"]
            self.rebalance_days = model_data["setup_stats"]["rebalance_days"]
            self.num_long = model_data["setup_stats"]["num_long"]

            self.num_features = model_data["setup_stats"]["num_features"]

            self.features = model_data["features"]
            self.factors = list(self.features.keys())
            self.betas = list(self.features.values())


        # Otherwise, an error has occured
        else:
            self.Error("Error: Please choose a valid trading method.")
            

        # Define trading range
        self.SetStartDate(self.start_date)
        self.SetEndDate(self.end_date)

        # Define strategy cash
        self.SetCash(self.start_cash)

        # Containers for our current/historical investment
        self.longSymbols = []
        self.investments = {}

        # Portfolio update setup
        self.nextLiquidate = self.Time

        # Baseline setup
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.Schedule.On(self.DateRules.WeekEnd("SPY"), self.TimeRules.BeforeMarketClose("SPY", 30), self.PlotBaseline)
        self.price_history = []

        # Stock selection universe
        self.AddUniverse(DataSource, "MultiMaxUniverse", Resolution.Daily, self.Selection)

    def Selection(self, data: List[DataSource]) -> List[Symbol]:
        """ Stock selection (TopKDrop) strategy. """

        # Skip if it is too soon to update
        if self.Time < self.nextLiquidate:
            return Universe.Unchanged

        # Create our model if it doesnt exist
        if (self.method == "Automatic" and self.features == None):

            # Train LASSO
            if (self.model_type == "LASSO"):
                self.features = self.generator.TrainLasso(self.alpha, self.num_features)

            # Train random forest
            elif (self.model_type == "RF"):
                self.features = self.generator.TrainRandomForest(self.num_trees, self.max_depth, self.num_features)

            # Train GBM
            elif (self.model_type == "GBM"):
                self.features = self.generator.TrainGradientBoost(self.num_trees, self.max_depth, self.num_features)

            # Train DNN
            elif (self.model_type == "DNN"):
                self.features = self.generator.TrainDNN(self.batch_size, self.epochs, self.num_features)

            # Error
            else:
                self.Error("Please select an avaiable ML model type: RF/GBM/DNN.")

            # Store all factors and their betas
            self.factors = list(self.features.keys())
            self.betas = list(self.features.values())

        # Filter our stocks that don't have the right factors
        filtered = [x for x in data if [hasattr(x.Props, f) for f in self.factors]]

        # Sort our stocks by each of the factors
        sorts = [sorted(filtered, key=lambda x: float(x.Props[f]), reverse=True) for f in self.factors]
        stockBySymbol = {}

        # For every stock...
        for stock in filtered:

            # Create weighted ranks based each stock's factors and betas
            ranks = np.empty(0)

            for i in range (0, len(sorts)):

                rank = self.betas[i] * (len(sorts[i]) - sorts[i].index(stock))
                ranks = np.append(ranks, [rank])

            # Store the average rank
            avgRank = np.mean(ranks)
            stockBySymbol[stock.Symbol] = avgRank

        # Sort stocks by their rank
        sorted_dict = sorted(stockBySymbol.items(), key = lambda x: x[1], reverse=True)
        symbols = [x[0] for x in sorted_dict]

        # Set the Liquidate Date
        self.nextLiquidate = self.Time + timedelta(self.rebalance_days)

        # Pick the stocks with the N highest ranks to buy
        self.longSymbols = symbols[:self.num_long]

        return self.longSymbols

    def PackageModel(self):
        """ Model saving system. """

        # ============ PARAMETER ============== #
        key = "Chocolate"

        # If we want to save our model...
        if (self.method != "Pretrained"):

            # Serialize the model to a string
            model_buffer = io.BytesIO()

            # Get universe stats
            setup_stats = self.MakeSetupStats()
            backtest_stats = self.MakeBacktestStats()
            encoded_model = None
            data_stats = None

            # Serialise model if applicable
            if (self.generator != None):

                joblib.dump(self.generator.model, model_buffer)
                model_buffer.seek(0)
                encoded_model = base64.b64encode(model_buffer.read()).decode('utf-8')

                data_stats = self.generator.MakeDataStats()

            # Define the model's data
            model_data = {
                'model': encoded_model,
                'features': self.features,
                'source': self.source,
                'data_stats': data_stats,
                'setup_stats': setup_stats,
                'backtest_stats': backtest_stats,
                'investments': self.investments
            }
                
            # Serialise the PTO
            model_data_str = json.dumps(model_data)

            # Save the string to the Object Store
            self.ObjectStore.Save(key + ".pto", str(model_data_str))
            self.Debug(f"Packaged and Exported Data as {key}.pto.")

    def OnData(self, data):
        """ Value investment strategy."""

        # Liquidate stocks in the end of every year
        if self.Time >= self.nextLiquidate:
            for holding in self.Portfolio.Values:

                # If the holding is in the long list for the next year, don't liquidate
                if holding.Symbol in self.longSymbols:
                    continue
                
                # If the holding is not in the list, liquidate
                if holding.Invested:
                    self.Liquidate(holding.Symbol)

        # Count the number of stocks to invest in
        count = len(self.longSymbols)

        # Return if no stocks to invet in
        if count == 0: 
            return

        # Open long position at the start of every year
        for symbol in self.longSymbols:
            self.SetHoldings(symbol, 1/count)

        # Update our stock tracking
        if self.Portfolio.Invested:
            for holding in self.Portfolio.Values:
                symbol = holding.Symbol.Value
                if symbol not in self.investments:
                    self.investments[symbol] = 0
                self.investments[symbol] += 1

        # Set the liquidate Date
        self.nextLiquidate = self.Time + timedelta(self.rebalance_days)

        # After opening positions, clear the long symbol lists until next universe selection
        self.longSymbols.clear()

    def MakeSetupStats(self):

        # Create source stats
        stats = {
            'start_date': self.start_date.strftime('%m/%d/%Y'),
            'end_date': self.end_date.strftime('%m/%d/%Y'),
            'start_cash': self.start_cash,
            'rebalance_days': self.rebalance_days,
            'num_long': self.num_long,
            'num_features': self.num_features
        }

        return stats

    def MakeBacktestStats(self):

        # Create source stats
        stats = {
            'sharpe_ratio': self.Statistics.TotalPerformance.PortfolioStatistics.SharpeRatio,
            'drawdown': self.Statistics.TotalPerformance.PortfolioStatistics.Drawdown,
            'annual_return': self.Statistics.TotalPerformance.PortfolioStatistics.CompoundingAnnualReturn,
            'alpha': self.Statistics.TotalPerformance.PortfolioStatistics.Alpha,
            'beta': self.Statistics.TotalPerformance.PortfolioStatistics.Beta,
            'win_rate': self.Statistics.TotalPerformance.PortfolioStatistics.WinRate,
            'ir': self.Statistics.TotalPerformance.PortfolioStatistics.InformationRatio,
            'total_return': self.Statistics.TotalPerformance.PortfolioStatistics.TotalNetProfit
        }

        return stats

    def OnEndOfAlgorithm(self):
        """ Prints the algorithm's stats and saves it in the ObjectStore. """

        # Print the the statistics
        stats = self.MakeBacktestStats()
        self.Debug(f"Algorithm complete with a total return of {stats['total_return']} and Sharpe Ratio {stats['sharpe_ratio']}.")
        
        # Print the top 10 investments and their frequencies
        top_investments = {A:N for (A,N) in [x for x in self.investments.items()][:10]} 
        self.Debug("Top 10 Investments:")
        for key, value in top_investments.items():
            self.Debug('{0}: {1}'.format(key, value)) 
            
        self.PackageModel()

    def PlotBaseline(self):  
        """ Plots baseline values at the end of each week. """

        # Get the price of our crypto over the past 2 days
        history = self.History(self.spy, 2, Resolution.Hour)

        if ('close' in history):
            price = history['close'].unstack(level= 0).iloc[-1]
            self.price_history.append(price)

            # Plot the price change of this crypto
            price_change = 100000 * self.price_history[-1] / self.price_history[0] 
            self.Plot("Strategy Equity", "Buy and Hold", price_change[::10])

# endregion