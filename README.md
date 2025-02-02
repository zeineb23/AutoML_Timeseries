﻿# AutoML_Timeseries
AutoML Timeseries is an AutoML API designed for time series forecasting. This project aims to automate the preprocessing, model selection, training, and prediction of time series data. The API supports multiple models and ultimately selects Prophet with exogenous variables for final predictions.

## Features

- **Preprocessing Pipeline**: Automatically cleans, scales, and fills missing values in the dataset.
- **Comparative Study**: Evaluates multiple models including Prophet, Neural Prophet, and ARIMA, selecting the best-performing model.
- **Model Selection**: Chooses Prophet with exogenous variables for time series forecasting.
- **Prediction**: Users can input a range of dates, and the API will predict values for each feature using individual Prophet models, merge them into a single dataframe, and pass them to the trained Prophet model.
- **Model Saving**: The trained model is saved for future use by the user.

## Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/your-username/automl-timeseries.git
cd automl-timeseries
pip install -r requirements.txt
```
