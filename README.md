# Time Series-Based Sales Forecasting

A time series forecasting project that analyzes historical sales data, identifies temporal patterns, and predicts future sales using a SARIMA model. The final model is deployed through an interactive Streamlit web application for visualization and forecasting.

## Project Overview

Sales forecasting helps businesses estimate future demand, understand sales patterns, and support better planning and decision-making.

This project uses historical Superstore sales data to build a time series forecasting pipeline. The workflow covers data preprocessing, missing-date handling, exploratory time series analysis, model development, evaluation, walk-forward validation, and web deployment.

The final forecasting model uses:

**SARIMA(0,0,1)(1,0,1,7)**

The model captures both short-term temporal relationships and weekly seasonality in daily sales.

---

## Objectives

- Prepare and clean historical sales data for time series analysis.
- Convert transaction-level data into a regular daily sales series.
- Identify and handle missing dates in the time series.
- Analyze trends and weekly seasonality.
- Examine stationarity using the Augmented Dickey-Fuller test.
- Analyze autocorrelation using ACF and PACF.
- Compare baseline, ARIMA, and SARIMA models.
- Evaluate models using MAE and RMSE.
- Perform walk-forward validation.
- Generate a 90-day sales forecast.
- Deploy the forecasting model using Streamlit.

---

## Dataset

The project uses the **Superstore Sales Dataset**, containing approximately four years of historical sales transactions.

### Dataset characteristics

- **Rows:** 9,800
- **Columns:** 18
- **Historical period:** January 2015 – December 2018
- **Target variable:** Sales

Important fields include:

- Order ID
- Order Date
- Ship Date
- Ship Mode
- Customer ID
- Customer Name
- Segment
- Country
- City
- State
- Region
- Product ID
- Category
- Sub-Category
- Product Name
- Sales

---

## Methodology

### 1. Data Preprocessing

The transaction-level dataset was converted into a daily sales time series by aggregating sales by `Order Date`.

Missing calendar dates were identified and the series was regularized to a continuous daily frequency.

Days with no recorded sales were assigned a sales value of `0`.

The resulting time series contains:

- **1,458 daily observations**
- **228 dates with no recorded sales**

This regularization ensures that the forecasting model receives a consistent daily time index.

---

### 2. Exploratory Time Series Analysis

Several analyses were performed to understand the behavior of the sales series:

- Daily sales trend
- 30-day rolling average
- Monthly sales patterns
- Sales by day of week
- Stationarity testing
- ACF analysis
- PACF analysis
- Seasonal decomposition

Weekly seasonality was particularly relevant, which motivated the use of a seasonal period of **7 days**.

---

### 3. Stationarity Testing

The Augmented Dickey-Fuller (ADF) test was used to assess stationarity.

The resulting p-value was approximately:

**3.09 × 10⁻⁵**

This provided evidence against the presence of a unit root in the series and supported the use of a model without first-order differencing.

---

## Model Development

Multiple forecasting approaches were evaluated.

### Baseline Models

Two baseline approaches were tested:

- Naive Forecast
- Seasonal Naive Forecast using a 7-day seasonal period

### ARIMA Models

Several ARIMA configurations were compared:

| Model | MAE | RMSE |
|---|---:|---:|
| ARIMA(0,0,1) | 1703.40 | 2632.35 |
| ARIMA(1,0,0) | 1703.62 | 2632.48 |
| ARIMA(1,0,1) | 1700.68 | 2632.61 |
| ARIMA(2,0,0) | 1703.89 | 2632.64 |

The results showed that introducing seasonal components could further improve forecasting performance.

---

## Final Model

The selected model is:

### SARIMA(0,0,1)(1,0,1,7)

The model includes a weekly seasonal period of **7 days**.

On the holdout test set, the final model achieved:

| Metric | Result |
|---|---:|
| MAE | **1549.26** |
| RMSE | **2401.91** |

Compared with the naive baseline, the final model achieved approximately:

- **34.54% improvement in MAE**
- **28.98% improvement in RMSE**

---

## Walk-Forward Validation

Time-based walk-forward validation was performed to evaluate model performance across multiple forecasting windows.

The validation used:

- Initial training size: 800 observations
- Forecast horizon: 30 days
- Validation folds: 12

Average walk-forward performance:

| Metric | Result |
|---|---:|
| MAE | **1439.41** |
| RMSE | **2062.19** |

Walk-forward validation provided an additional assessment of how the model performs when forecasting sequential future periods.

---

## 90-Day Forecast

After evaluation, the final SARIMA model was refitted using the complete historical daily sales series.

A **90-day forecast** was then generated.

### Forecast period

**December 31, 2018 – March 30, 2019**

### Forecast summary

| Metric | Value |
|---|---:|
| Total Forecasted Sales | **$241,453.11** |
| Average Daily Forecast | **$2,682.81** |
| Minimum Daily Forecast | **$608.35** |
| Maximum Daily Forecast | **$4,092.92** |

Prediction intervals are also generated to represent forecast uncertainty.

---

## Streamlit Web Application

The forecasting model is deployed through an interactive Streamlit application.

The application provides:

- Historical sales overview
- Daily sales visualization
- Rolling average analysis
- Model performance metrics
- Walk-forward validation results
- Configurable forecast horizon
- Future sales forecast
- Forecast confidence intervals
- Forecast summary statistics
- Historical vs. forecast visualization
- Monthly forecast analysis
- Forecast by day of week
- Forecast data table
- CSV download functionality
- Methodology and project information

---

## Project Structure

```text
Time-Series-Based-Sales-Forecasting/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/
│   ├── raw/
│   │   └── train.csv
│   │
│   └── processed/
│       └── 90_day_sales_forecast.csv
│
└── notebooks/
    └── 01_sales_forecasting.ipynb