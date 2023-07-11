# Project-2-Bootcamp

# Overview
This is a project in which we attempt to use machine learning to predict housing prices over a period from 2008-2021. 

The areas of interest are:
* New York
* Los Angeles
* Miami
* Phoenix
* Seattle
* Boston
* Houston
* Chicago

For features to predict the housing prices, we utilized:
* the monthly mean SPY price
* the monthly mean VIX price
* the monthly mean price of a lumber index
* the local unemployment rate in each market
* the local Median Household Income (MHI)
* the 30 year fixed mortgage rate
* the Federal Funds Rate

*Note: We also collected data on local crime but found it impossible to integrate



For each city, we utilize a Facebook Prophet model, a neural net (one with one hidden layer and another with two hidden layers), and a KNN Regressor to predict housing prices in 8 key markets.

The neural net turned out to have very unreliable predictive power and is not worth discussing in detail


# Usage

##Technologies
*pandas* - https://pandas.pydata.org/
*numpy* - https://numpy.org/
*pathlib* - https://docs.python.org/3/library/pathlib.html
*tensorflow* - https://www.tensorflow.org/resources/libraries-extensions
*tensorflow.keras* - https://www.tensorflow.org/guide/keras
*sklearn* - https://scikit-learn.org/stable/user_guide.html
*matplotlib* - https://matplotlib.org/stable/plot_types/index.html
*seaborn* - https://seaborn.pydata.org/
*streamlit* - https://streamlit.io/

## Raw Data
Raw data was collected from a variety of sources and is largely stored in [raw_data](https://github.com/moic10x/Project-2/tree/main/raw_data).
Housing prices for each market was gathered from [zillow](https://www.zillow.com/research/data/).
Unemployment and MHI data was collected from the [St.Louis Fed](https://fred.stlouisfed.org)
VIX and SPY data were also collected from the [St.Louis Fed] (https://fred.stlouisfed.org/series/VIXCLS), (https://fred.stlouisfed.org/series/SP500)
Lumber data was found on [Macro Trends](https://www.macrotrends.net/2637/lumber-prices-historical-chart-data)

## Notebooks that clean raw data
Notebooks to clean raw data are found in respective folders. For example, the [spy_lumber_vix](https://github.com/moic10x/Project-2/tree/main/spy_lumber_vix) folder contains the ipynb which cleaned the raw data for SPY, VIX, and Lumber in such a way that it was able to be merged together to feed our ML regressors. Same goes for [fed_funds_&_interest_rates](https://github.com/moic10x/Project-2/tree/main/fed_funds_%26_interest_rates) and [unemployment](https://github.com/moic10x/Project-2/tree/main/unemployment).

## Clean Notebooks
Clean csvs can be found in [clean_data](https://github.com/moic10x/Project-2/tree/main/clean_data). These are the csvs that ultimately get fed into the [CribPredict.ipynb]() notebook which contains the Neural Net and the KNNeighbors Regressor.

## ML and Facebook Prophet

# Findings
Findings were presented during a live class.  
*A summary of the presentation can be found here* - https://docs.google.com/presentation/d/1zsxBzoBtbDNCQ6d5RNz5Oyg6Ib0jm7wf4-0yhQVRHtQ/edit#slide=id.p

We utilized streamlit to present our findings. The streamlit file is run through the streamlit_app.py file. 

Facebook Prophet is run in the Prophet_Forecasting.ipynb file.
