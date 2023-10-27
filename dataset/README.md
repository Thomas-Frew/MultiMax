# Dataset Scraping

This folder contains all the tools necessary to create a dataset of fundamental factors, fundamental growth rates and technical factors for all stocks in the Wilshire 5000 between 1/1/2000 and 1/1/2023.

Our tools gather information from [FinanceModellingPrep](link). They require a Starter API key from FinanceModellingPrep to use.

## Usage

1. Visit [FinanceModellingPrep](link) and get a Starter API key.
2. Replace all instances of `[API_KEY]` in every notebook with your API key.
3. Run [Make Fundamentals](make_fundamentals.ipynb), [Make Fundamental Growth](make_fundamental_growth.ipynb), [Make Stock Growth](make_stock_growth.ipynb) and [Make Technicals](make_technicals.ipynb) to create the datasets containing our input and output features.
4. Run [Combine Datasets](combine_datasets.ipynb) to combine the datasets into a master dataset!
5. Upload this dataset to Dropbox if you want to use it in MultiMax
